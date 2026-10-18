# SchemaLens MCP server

Lets AI agents (Claude, Cursor, any [MCP](https://modelcontextprotocol.io) client) pull structured
fields out of printed documents: "read the invoices in ~/Downloads and give me the totals".

It is for local work: the MCP server runs on your machine and calls **the SchemaLens you set up
yourself** - `docker compose up` on localhost (the default) or your own server. It holds no OCR and
no models, so it installs in seconds; the processing happens in your instance.

Start your instance first:

```bash
git clone https://github.com/maszjan/schemalens && cd schemalens
cp .env.example .env && uv run schemalens-api keys new --env-file .env
docker compose up --build     # API on http://localhost:8000
```

## Tools

| Tool | What |
|---|---|
| `list_presets` | ready-made schemas (invoice, receipt, CV, business card, contract terms, ...) and their fields |
| `extract_document` | a local file (JPEG, PNG, WEBP, TIFF, PDF) + a `preset` and/or your own `fields` → values with a confidence and source lines |

## Configuration

| Variable | Default | What |
|---|---|---|
| `SCHEMALENS_API_URL` | `http://localhost:8000` | your instance |
| `SCHEMALENS_API_KEY` | none | a key from the API's `API_KEYS`, sent as `Authorization: Bearer` (required) |
| `SCHEMALENS_TIMEOUT_S` | `300` | a local model needs about a minute per document |

## Add it to your client

Claude Code:

```bash
claude mcp add schemalens -e SCHEMALENS_API_URL=http://localhost:8000 -e SCHEMALENS_API_KEY=<your key> -- \
  uvx --from "git+https://github.com/maszjan/schemalens#subdirectory=services/mcp" schemalens-mcp
```

Claude Desktop (`claude_desktop_config.json`) or Cursor (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "schemalens": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/maszjan/schemalens#subdirectory=services/mcp", "schemalens-mcp"],
      "env": { "SCHEMALENS_API_URL": "http://localhost:8000", "SCHEMALENS_API_KEY": "<your key>" }
    }
  }
}
```

From a clone of the repository: `uv run --package schemalens-mcp schemalens-mcp`.

Documents go only to the API you configure; with the `local` LLM profile on your own machine
nothing leaves it.
