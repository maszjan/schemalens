"""Write openapi.json from the app, without starting a server.

    uv run python scripts/export_openapi.py          # update the file
    uv run python scripts/export_openapi.py --check  # fail if the file is out of date

The file is committed: SDKs are generated from it and its diff shows contract changes.
"""

import json
import sys
from pathlib import Path

from schemalens_api.main import create_app
from schemalens_api.settings import ApiSettings

TARGET = Path(__file__).parent.parent / "openapi.json"


def render() -> str:
    # Defaults only, so the output does not depend on the local .env.
    app = create_app(ApiSettings(_env_file=None))
    return json.dumps(app.openapi(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> int:
    rendered = render()
    if "--check" in sys.argv:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != rendered:
            print("openapi.json is out of date: run scripts/export_openapi.py", file=sys.stderr)
            return 1
        return 0
    TARGET.write_text(rendered, encoding="utf-8")
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
