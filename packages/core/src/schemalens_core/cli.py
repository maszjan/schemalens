"""Command line adapter.

    python -m schemalens_core.cli FILE --schema SCHEMA.json
    python -m schemalens_core.cli FILE --preset business_card [--schema EXTRA.json]

Prints the result as JSON on stdout. Nothing is written to disk.
Files under samples/private*/ are processed only with the local LLM profile.
"""

import argparse
import json
import sys
from pathlib import Path

from schemalens_core.config import get_settings
from schemalens_core.errors import InvalidSchema, SchemaLensError
from schemalens_core.models import ExtractionOptions
from schemalens_core.pipeline import build_pipeline


def _is_private(path: Path) -> bool:
    """True for files under `samples/private*/` (real documents, local LLM only)."""
    parts = path.resolve().parts
    return any(
        parent == "samples" and name.startswith("private")
        for parent, name in zip(parts, parts[1:], strict=False)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m schemalens_core.cli",
        description="Extract structured JSON from any document using a JSON Schema or a preset.",
    )
    parser.add_argument("file", type=Path, help="image (JPEG/PNG/WEBP/TIFF) or PDF")
    parser.add_argument("--schema", type=Path, help="JSON Schema file; with --preset it extends it")
    parser.add_argument("--preset", help="name of a schema preset from presets/")
    parser.add_argument("--tier", choices=["fast", "accurate"], default="fast", help="model tier")
    parser.add_argument(
        "--profile",
        choices=["groq", "local", "openai"],
        help="LLM profile (default: ACTIVE_LLM_PROFILE)",
    )
    parser.add_argument(
        "--ocr-text", action="store_true", help="include the numbered OCR lines in the output"
    )
    args = parser.parse_args(argv)
    if args.schema is None and args.preset is None:
        parser.error("one of --schema or --preset is required")

    try:
        document = args.file.read_bytes()
        schema = json.loads(args.schema.read_text(encoding="utf-8")) if args.schema else None
    except (OSError, ValueError) as exc:
        print(f"error: cannot read input: {exc}", file=sys.stderr)
        return 2

    options = ExtractionOptions(model_tier=args.tier, include_ocr_text=args.ocr_text)
    try:
        settings = get_settings()
        profile = settings.llm_profile(args.profile)
        if _is_private(args.file) and profile.name != "local":
            print(
                f"error: {args.file} is a private sample and must not leave this machine; "
                f"refusing to process it with profile '{profile.name}'. Use --profile local.",
                file=sys.stderr,
            )
            return 2
        pipeline = build_pipeline(settings, profile)
        result = pipeline.extract(document, None, schema, options, preset=args.preset)
    except InvalidSchema as exc:
        print("error: invalid schema", file=sys.stderr)
        for error in exc.errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    except SchemaLensError as exc:
        print(f"error ({exc.code}): {exc}", file=sys.stderr)
        return 1

    output = result.model_dump()
    if output["ocr_text"] is None:
        del output["ocr_text"]
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
