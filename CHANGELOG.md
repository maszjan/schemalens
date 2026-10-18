# Changelog

All notable changes to this project are described here.

## [Unreleased]

First open-source release of the SchemaLens engine and API.

- Extraction from printed photos, scans and PDFs into any JSON Schema or a ready-made preset
  (invoice, receipt, CV, business card, contract terms, product label, menu, generic document,
  lecture notes): OCR (EasyOCR) + an LLM with a strict JSON schema + deterministic normalization and
  validation; a confidence score and source lines for every value; `null` instead of a guess.
- Bring your own LLM: profiles `local` (Ollama, nothing leaves the machine), `openai` (any
  OpenAI-compatible endpoint: OpenAI, Azure OpenAI, vLLM, LM Studio, OpenRouter) and `groq`.
- HTTP API (FastAPI): `/api/v1/extract` (JSON, CSV or XLSX; live progress with `?stream=true`),
  `/api/v1/render`, `/api/v1/presets`, `/api/v1/demo`, `/healthz`; daily limits per client address.
- Endpoints under `/api/v1/` (`https://your-domain/api/v1/extract`); `/healthz`, `/docs` and
  `/openapi.json` at the root.
- Dockerfile (OCR models baked in) and a compose file; CI with tests and lint on GitHub Actions.
- AWS SAM template to adapt: API Gateway (API keys, usage plan with throttling and a daily quota)
  in front of a Lambda container image, the LLM of your choice, no database; an optional public
  demo mode adds a streaming Function URL called with a demo key (`DemoKey`) and per-address
  counters in DynamoDB.
- API keys, always required: bearer or `X-API-Key`, constant-time check; the API does not start
  without one. `schemalens-api keys new [--demo] [--env-file .env]` generates them. Service keys
  (`API_KEYS`) have a daily limit per key (`KEY_DAILY_DOCUMENTS`); demo keys (`DEMO_KEYS`), meant
  for a public web page, are limited per client address and in total. On AWS the function accepts
  the calls API Gateway checked (`GATEWAY_KEYS`).
- MCP server (`services/mcp`, `schemalens-mcp`): `list_presets` and `extract_document` tools for
  Claude, Cursor and other agents, for local work with your own instance (`SCHEMALENS_API_URL`,
  localhost by default; `SCHEMALENS_API_KEY`, required).
- Your own presets: a JSON file in `presets/` or a mounted folder (`PRESETS_DIR`), documented with
  an example.
- CLI and an eval harness on synthetic documents.
- English presets (version 2.0.0): field names, descriptions, enum values (`cash`, `card`,
  `food`, …), example documents, few-shot templates and eval cases. `invoice_pl` is now `invoice`
  (any country's invoice; no Polish-only domain types in the presets). Domain types such as `nip`
  or `postal_code_pl` stay available for your own schemas.
- Currencies: any ISO 4217 code, plus unambiguous symbols and names (₹, R$, yen, rupee...);
  ambiguous ones (`kr`, `¥`) are left as written.
