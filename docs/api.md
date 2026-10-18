# API (v1)

Every `/v1` call needs an API key: `Authorization: Bearer <key>` or `X-API-Key: <key>`, `401`
otherwise (`/healthz` and the docs stay open). The API refuses to start without any key. Generate
one with `schemalens-api keys new --env-file .env` (or, with Docker only,
`docker compose run --rm api schemalens-api keys new`). Two kinds:

- **service keys** (`API_KEYS=key1,key2`): for your systems; the daily limit counts per key
  (`KEY_DAILY_DOCUMENTS`, 0 = none).
- **demo keys** (`DEMO_KEYS`, `keys new --demo`): for a public web page, where anyone can read the
  key in the browser. It only says which page calls, so its calls are limited per client address
  (`DEMO_DAILY_DOCUMENTS`) and in total (`DEMO_GLOBAL_DAILY_DOCUMENTS`).

On AWS the template puts API Gateway in front: send the generated gateway key as `x-api-key`; the
function accepts what the gateway checked (`GATEWAY_KEYS`). Other protection: CORS
(`ALLOWED_ORIGINS`) and size limits.

Endpoints live under `/api/v1/` (`https://your-domain/api/v1/extract`; `v1` is the API version);
`/healthz`, `/docs` and `/openapi.json` are at the root. Errors use RFC 9457
(`application/problem+json`). The machine-readable contract is [`openapi.json`](../openapi.json);
a running API serves interactive docs at `/docs`.

The examples read your instance and key from the environment:

```bash
export SCHEMALENS_URL=http://localhost:8000   # or your deployment's URL (API_PORT sets the local port)
export SCHEMALENS_KEY=sl_...                  # one of its API_KEYS
```

| Method | Path | What |
|---|---|---|
| POST | `/api/v1/extract` | multipart (`file`, `preset` and/or `schema`, `options`) or JSON (`document_base64`). Up to `MAX_UPLOAD_BYTES` (4 MB) and `MAX_PAGES` (3). The schema and the pages are checked **before** the daily limit is counted. Result: JSON (default), CSV or XLSX. |
| POST | `/api/v1/render` | `{result, schema?, preset?, labels?}` → JSON, CSV or XLSX from a result the client already has. Nothing is extracted again or stored. |
| GET | `/api/v1/presets` | presets with their fields (`name`, `title`, `description`, `fields[]`). |
| GET | `/api/v1/demo` | the limits of this deployment (`max_upload_bytes`, `max_pages`, `daily_documents`). |
| GET | `/healthz` | liveness. |

**Errors:** `406` unknown format; `413` file or page limit exceeded; `415` unsupported type; `422`
invalid schema or request, or a document OCR could not read (`document-unreadable`, the LLM is not
called); `429` daily limit (`limit-reached`, with `Retry-After`) or all slots busy; `502` the LLM gave
no usable answer; `503` LLM provider rate limit or missing configuration; `504` `SYNC_TIMEOUT_S`
exceeded. Every response carries `X-Request-ID` and `Server-Timing`. The problem's `type` is
`https://schemalens.space/docs#error-<slug>` (e.g. `#error-invalid-schema`), a link to the
description of that error; match on the slug, not on the `title`.

## Extraction

```bash
curl -s -H "Authorization: Bearer $SCHEMALENS_KEY" $SCHEMALENS_URL/api/v1/extract -F file=@card.jpg -F preset=business_card \
  -F 'options={"model_tier":"accurate","include_ocr_text":true}'
```

```json
{
  "data": { "full_name": "Anna Example", "email": "anna@example.com", "phone": "+48600100200" },
  "fields": { "full_name": { "confidence": 0.95, "evidence": ["L001"] } },
  "warnings": ["phone: normalized 'tel. +48 600 100 200' to '+48600100200'"],
  "usage": { "pages": 1, "ocr_ms": 2310, "llm_ms": 640, "model": "openai/gpt-oss-20b",
             "language": "pl", "ocr_confidence": 0.84, "context": "full", "prompt_version": "…" },
  "ocr_text": ["[L001] Anna Example"]
}
```

- `ocr_text` only with `options.include_ocr_text: true` (useful to show where a value came from).
- `usage.language`: ISO 639-1 code of the document language when it can be told (no model involved).
- `options.model_tier`: `fast` (default) or `accurate`; the server picks the LLM profile.

JSON instead of multipart:

```bash
curl -s -H "Authorization: Bearer $SCHEMALENS_KEY" $SCHEMALENS_URL/api/v1/extract -H "Content-Type: application/json" \
  -d "{\"document_base64\": \"$(base64 < receipt.jpg | tr -d '\n')\", \"preset\": \"receipt\"}"
```

## Live progress (`?stream=true`, JSON only)

The answer is `application/x-ndjson`: one JSON event per line as the work goes on. Events carry
counts, timings, the language and the model name, never document text or values:

```json
{"event": "accepted", "bytes": 111156, "pages": 1}
{"event": "reading", "pages": 1}
{"event": "read", "lines": 24, "ocr_ms": 3538, "ocr_confidence": 0.778, "language": "pl"}
{"event": "extracting", "model": "openai/gpt-oss-20b", "fields": 9}
{"event": "result", "result": { "data": {}, "fields": {}, "warnings": [], "usage": {} }}
```

The last event is `result` (the usual JSON body) or `error` with the problem a plain request would
get (`{"event": "error", "problem": {"type": "…#error-llm-failed", "status": 502}}`); the HTTP status is
already 200 by then. Errors found before the work starts (406, 413, 415, a 422 schema, 429) are plain
responses. On AWS the Function URL streams (`RESPONSE_STREAM`), so events arrive live there too.

## Schema or preset

A request has a `schema` (a JSON Schema), a `preset` (see [presets/](../presets/)) or both; then
`schema.properties` extends or overrides the preset's fields and `required` is merged:

```json
{ "preset": "business_card",
  "schema": { "type": "object", "properties": {
      "industry": {"type": "string", "description": "The company's industry", "x-schemalens-derived": true} } } }
```

An unknown preset, or neither, is a 422.

**Limits** (checked on input, 422 when exceeded): at most 50 fields, depth ≤ 3; types
string/number/integer/boolean/array/object/enum; `format: date|date-time|email`; `pattern`,
`minLength`/`maxLength`, `minimum`/`maximum` (validated after extraction). Other keywords (`$ref`,
`anyOf`, …) are a 422. Every field is optional and nullable for the model (missing data = `null`);
`required` is checked after extraction and reported in `warnings`, never forced.

**Extensions:**

- `"x-schemalens-type"`: a domain type for a string field: `nip`, `regon`, `pesel`, `iban`, `phone`,
  `currency`, `country`, `postal_code_pl`, `url`. It turns on cleaning, the checksum where there is
  one, and careful OCR fixes. For `nip`/`regon`/`pesel`, stray non-digit characters are dropped only
  when the result passes the checksum. For `url`, a mangled `www` (`Wwv`, `vvw`) and a dot read as a
  space or comma before the domain ending are fixed. Anything else is left as it was, with a warning.
  Types come **only** from this annotation, never from field names.
- `"x-schemalens-derived": true`: an inferred field (a classification, a summary, the language). The
  model may infer it from the content but must not add facts from outside the document; it writes
  it in the language of the document. `evidence` is optional and the confidence is capped.

## Result formats

`POST /api/v1/extract` and `POST /api/v1/render` answer in JSON by default; ask for a file with
`?format=csv|xlsx` (wins) or an `Accept` header. An unknown format is a 406 before any work.

- **csv**: the values of `data`, UTF-8 with a BOM. Columns follow the schema: objects flattened to
  dotted paths (`seller.name`), the first top-level list of objects as one row per item with scalar
  fields repeated, further lists as JSON in one cell (`X-SchemaLens-Flattening: partial`), lists of
  plain values joined with ` | `. `include_meta=true` adds `<field>__confidence` columns. Text starting
  with `=`, `+`, `-`, `@` is prefixed with an apostrophe (no formula injection).
- **xlsx**: sheet `data` (one row), one sheet per list of objects, sheet `meta` (field, confidence,
  evidence, warnings, usage). Numbers as numbers, dates as dates, text never as a formula; styled for
  reading (header, frozen row, filters, column widths, coloured confidence). `POST /api/v1/render` takes
  optional `labels` (field path → heading, e.g. in the reader's language).

```bash
curl -s -H "Authorization: Bearer $SCHEMALENS_KEY" "$SCHEMALENS_URL/api/v1/extract?format=csv&include_meta=true" -F file=@invoice.jpg -F preset=invoice -o invoice.csv
curl -s -H "Authorization: Bearer $SCHEMALENS_KEY" "$SCHEMALENS_URL/api/v1/render?format=xlsx" -H 'Content-Type: application/json' \
  -d "{\"result\": $(cat result.json), \"preset\": \"invoice\"}" -o result.xlsx
```
