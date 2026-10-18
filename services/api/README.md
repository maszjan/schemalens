# SchemaLens API

Thin FastAPI adapter over `schemalens_core`. Contract: [docs/api.md](../../docs/api.md),
machine-readable in [`openapi.json`](../../openapi.json), interactive docs at `/docs` (Scalar).

## Run

```bash
cp .env.example .env            # an LLM profile (LOCAL_* or GROQ_*)
uv run schemalens-api keys new --env-file .env   # every call needs a key
docker compose up --build       # API on :8000, or API_PORT from .env (docs /docs)
docker compose --profile dev up --build api-dev   # hot reload on :8001
```

Without Docker: `uv run uvicorn schemalens_api.main:app --reload` (needs the EasyOCR models in
`.models/easyocr`; see `OCR_ALLOW_DOWNLOAD` in `.env.example`).

There are no accounts. Every call needs an API key (`API_KEYS`, or `DEMO_KEYS` for a public page,
limited per client address and per day); documents are limited to `MAX_UPLOAD_BYTES` / `MAX_PAGES`.
See [docs/api.md](../../docs/api.md). The server picks the LLM (`ACTIVE_LLM_PROFILE`); a client chooses only
`options.model_tier` (`fast` or `accurate`) and `options.include_ocr_text`.

## Examples

```bash
export SCHEMALENS_URL=http://localhost:8000 SCHEMALENS_KEY=<your key>
curl -s -H "Authorization: Bearer $SCHEMALENS_KEY" $SCHEMALENS_URL/api/v1/presets                                          # ready-made schemas
curl -s -H "Authorization: Bearer $SCHEMALENS_KEY" $SCHEMALENS_URL/api/v1/extract -F file=@card.jpg -F preset=business_card
curl -s -H "Authorization: Bearer $SCHEMALENS_KEY" "$SCHEMALENS_URL/api/v1/extract?stream=true" -F file=@card.jpg -F preset=business_card   # live progress (NDJSON)
```

Your own schema:

```bash
curl -s -H "Authorization: Bearer $SCHEMALENS_KEY" $SCHEMALENS_URL/api/v1/extract \
  -F file=@invoice.pdf \
  -F 'schema={"type":"object","properties":{
        "seller_name":{"type":"string","description":"Name of the seller"},
        "issue_date":{"type":"string","format":"date"},
        "total_gross":{"type":"number"},
        "seller_nip":{"type":"string","x-schemalens-type":"nip"},
        "cost_category":{"type":"string","enum":["fuel","equipment","services","other"],
                         "x-schemalens-derived":true}},
      "required":["seller_name","total_gross"]}'
```

A preset extended with your own fields (`schema` adds to or overrides the preset):

```bash
curl -s -H "Authorization: Bearer $SCHEMALENS_KEY" $SCHEMALENS_URL/api/v1/extract -F file=@card.jpg -F preset=business_card \
  -F 'schema={"type":"object","properties":{
        "industry":{"type":"string","description":"The company industry","x-schemalens-derived":true}}}'
```

Files instead of JSON:

```bash
curl -s -H "Authorization: Bearer $SCHEMALENS_KEY" "$SCHEMALENS_URL/api/v1/extract?format=csv" -F file=@invoice.jpg -F preset=invoice -o invoice.csv
curl -s -H "Authorization: Bearer $SCHEMALENS_KEY" "$SCHEMALENS_URL/api/v1/extract?format=xlsx" -F file=@invoice.jpg -F preset=invoice -o invoice.xlsx
curl -s -H "Authorization: Bearer $SCHEMALENS_KEY" "$SCHEMALENS_URL/api/v1/render?format=xlsx" -H 'Content-Type: application/json' \
  -d "{\"result\": $(cat result.json), \"preset\": \"invoice\"}" -o result.xlsx   # no second extraction
```

## Privacy

Documents are processed in memory and never written to disk, a database or logs. The only state is
rate-limit counters: a salted hash of the client address and a number per day (in memory locally,
DynamoDB with a 2-day TTL on AWS). Logs are JSON lines with metadata only.

## Development

```bash
uv run pytest -q services/api/tests        # ASGI tests with a fake pipeline, no network
uv run python scripts/export_openapi.py    # refresh openapi.json after a contract change
```

On AWS Lambda the same uvicorn app runs behind the Lambda Web Adapter (Function URL in
`RESPONSE_STREAM` mode; the warm-up ping arrives as `POST /internal/warm-up`). Importing the app
does not load the OCR stack; the pipeline is built on the first extraction request.
