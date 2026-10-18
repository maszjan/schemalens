"""`schemalens-api keys new`: generate an API key, print it once, optionally add it to a .env.

schemalens-api keys new                       # print a service key (API_KEYS)
schemalens-api keys new --env-file .env       # ... and add it to API_KEYS in .env
schemalens-api keys new --demo --env-file .env  # a demo key for a public page (DEMO_KEYS)
"""

import argparse
import re
import secrets
import sys
from pathlib import Path

PREFIX = {"API_KEYS": "sl_", "DEMO_KEYS": "sl_demo_"}


def new_key(variable: str) -> str:
    # 32 random bytes; the prefix only tells the kinds apart in a config file.
    return PREFIX[variable] + secrets.token_urlsafe(32)


def add_to_env(path: Path, variable: str, key: str) -> None:
    """Append the key to `VARIABLE=` in the file (comma-separated), creating the line if needed."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    pattern = re.compile(rf"^\s*{variable}\s*=(.*)$")
    for index, line in enumerate(lines):
        if match := pattern.match(line):
            current = match.group(1).strip().strip('"').strip("'")
            lines[index] = f"{variable}={current},{key}" if current else f"{variable}={key}"
            break
    else:
        lines.append(f"{variable}={key}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="schemalens-api", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    keys = commands.add_parser("keys", help="API keys").add_subparsers(dest="action", required=True)
    new = keys.add_parser("new", help="generate a key")
    new.add_argument(
        "--demo",
        action="store_true",
        help="a demo key (DEMO_KEYS): it may sit in a public web page; limited per address",
    )
    new.add_argument("--env-file", type=Path, help="add the key to this .env file")
    args = parser.parse_args(argv)

    variable = "DEMO_KEYS" if args.demo else "API_KEYS"
    key = new_key(variable)
    if args.env_file:
        add_to_env(args.env_file, variable, key)
        print(
            f"Added to {variable} in {args.env_file}. Restart the API to use it.", file=sys.stderr
        )
    print(key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
