"""Eval runner: python -m evals.run [--compare local,groq]

Runs the synthetic golden set through one or more LLM profiles (default: local, which costs
nothing). Hosted profiles are kept cheap for a free tier: at most --max-docs documents,
answers cached in evals/.cache/, a token estimate printed up front, and the whole run stops
with a clear message when the provider's quota is exhausted. OCR runs once per document
and is shared between profiles, so a comparison only measures the LLM. Raw results go to
evals/results/ (gitignored); evals/report.md holds metrics only: no document content, no keys.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from evals.cases import Case, load_cases
from evals.metrics import OUTCOMES, field_outcome, percentile
from schemalens_core.config import LlmProfile, Settings
from schemalens_core.context import estimate_tokens, fit_to_context
from schemalens_core.errors import LlmRateLimited, ProfileUnavailable, SchemaLensError
from schemalens_core.llm import LlmResponse, OpenAICompatibleClient
from schemalens_core.models import ExtractionOptions, ExtractionResult, OcrResult
from schemalens_core.ocr import EasyOcrEngine, OcrEngine, read_document
from schemalens_core.pipeline import Pipeline
from schemalens_core.preprocess import load_document
from schemalens_core.presets import resolve_schema
from schemalens_core.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from schemalens_core.rag import ChromaRetriever
from schemalens_core.schemas import compile_schema

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / ".cache"
DEFAULT_PROFILE = "local"
DEFAULT_MAX_DOCS = 4
# Which documents a capped run takes first: one hard case of each kind, then the rest.
PRIORITY = (
    "invoice-example",
    "receipt-prompt-injection",
    "other-meeting-note",
    "business_card-example",
)
# Rough answer size per reasoning effort, for the estimate printed before a hosted run.
COMPLETION_TOKENS = {None: 500, "low": 500, "medium": 1500, "high": 3000}
NOTES_DIR = ROOT / "notes"  # hand-maintained sections appended to every report
ACCURACY_TARGET = 0.80
CONFIDENCE_SPLIT = 0.8


class SharedOcr:
    """Runs OCR once per distinct page set and replays it for every profile."""

    def __init__(self, engine: OcrEngine) -> None:
        self._engine = engine
        self._cache: dict[str, OcrResult] = {}

    def read(self, images: list[Image.Image]) -> OcrResult:
        key = hashlib.sha256(b"".join(image.tobytes() for image in images)).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._engine.read(images)
        return self._cache[key]


class RecordingLlm:
    """Passes calls through and keeps the raw responses, so scoring can be replayed."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.responses: list[LlmResponse] = []

    def extract(self, **kwargs: Any) -> LlmResponse:
        response = self._client.extract(**kwargs)
        self.responses.append(response)
        return response


def cache_key(profile: LlmProfile, *, system: str, user: str, json_schema: Any, model: str) -> str:
    """Identifies one LLM request: any change to prompt, model or effort is a different key."""
    payload = json.dumps(
        [PROMPT_VERSION, model, profile.reasoning_effort, system, user, json_schema], sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class CachedLlm:
    """Keeps answers of a hosted LLM on disk, so a rerun spends tokens only on what changed.

    Used for evals only: every document here is synthetic, so the cache holds no real data.
    """

    def __init__(self, client: Any, profile: LlmProfile, directory: Path) -> None:
        self._client = client
        self._profile = profile
        self._directory = directory
        self.hits = 0

    def extract(self, **kwargs: Any) -> LlmResponse:
        path = self._directory / f"{cache_key(self._profile, **kwargs)}.json"
        if path.is_file():
            self.hits += 1
            return LlmResponse(**json.loads(path.read_text(encoding="utf-8")))
        response: LlmResponse = self._client.extract(**kwargs)
        self._directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(response.__dict__), encoding="utf-8")
        return response


class QuotaExhausted(Exception):
    """The provider asks to wait far longer than a run can: stop instead of failing every case."""

    def __init__(self, retry_after: float | None) -> None:
        super().__init__("quota exhausted")
        self.retry_after = retry_after


class ReplayLlm:
    def __init__(self, responses: list[LlmResponse]) -> None:
        self._responses = list(responses)

    def extract(self, **kwargs: Any) -> LlmResponse:
        return self._responses.pop(0)


@dataclass
class FieldRow:
    case: str
    category: str
    field: str
    outcome: str
    confidence: float
    confidence_before: float  # without the grounding, evidence-breadth and OCR-noise rules
    grounding_warning: bool
    too_broad: bool
    noise_warning: bool = False


@dataclass
class RunResult:
    name: str
    model: str
    reasoning_effort: str | None
    rows: list[FieldRow] = field(default_factory=list)
    ocr_ms: list[int] = field(default_factory=list)
    llm_ms: list[int] = field(default_factory=list)
    tokens_per_page: list[float] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)
    retries: int = 0
    escalations: int = 0
    errors: list[str] = field(default_factory=list)  # "case: error code"
    raw: list[dict[str, Any]] = field(default_factory=list)
    documents: int = 0
    cache_hits: int = 0

    def accuracy(self, rows: list[FieldRow] | None = None) -> float:
        rows = self.rows if rows is None else rows
        return sum(r.outcome == "correct" for r in rows) / len(rows) if rows else 0.0


def resolve_run(spec: str, settings: Settings) -> tuple[str, LlmProfile]:
    """`groq`, `local` or `groq:low` (profile with another reasoning effort)."""
    name, _, effort = spec.partition(":")
    profile = settings.llm_profile(name)
    if effort:
        profile = profile.model_copy(update={"reasoning_effort": effort})
    return spec, profile


def build_retriever(settings: Settings) -> ChromaRetriever | None:
    """Retrieval as in production, so evals measure what the service does."""
    return ChromaRetriever(settings) if settings.rag_enabled else None


def select_cases(cases: list[Case], profile: LlmProfile, max_docs: int) -> list[Case]:
    """All cases for the local profile; the first `max_docs` by priority for hosted ones."""
    if profile.name == "local":
        return cases
    order = {case_id: index for index, case_id in enumerate(PRIORITY)}
    ranked = sorted(cases, key=lambda case: order.get(case.id, len(order)))
    return ranked[:max_docs]


def estimate_run(
    cases: list[Case], settings: Settings, profile: LlmProfile, ocr: OcrEngine, cache_dir: Path
) -> tuple[int, int]:
    """(tokens a run will roughly spend, documents already answered from the cache)."""
    tokens = cached = 0
    for case in cases:
        compiled = compile_schema(resolve_schema(case.schema, case.preset, settings), settings)
        recognized = read_document(load_document(case.document(), settings), ocr)
        lines, _ = fit_to_context(recognized, profile.direct_limit, settings.token_estimate_chars)
        request = {
            "system": SYSTEM_PROMPT,
            "user": build_user_prompt(lines, compiled.source),
            "json_schema": compiled.llm_schema(strict=profile.strict_schema),
            "model": profile.model_fast,
        }
        if (cache_dir / f"{cache_key(profile, **request)}.json").is_file():
            cached += 1
            continue
        prompt = request["system"] + request["user"] + json.dumps(request["json_schema"])
        tokens += estimate_tokens(prompt, settings.token_estimate_chars)
        tokens += COMPLETION_TOKENS.get(profile.reasoning_effort, 1500)
    return tokens, cached


def run_case(
    case: Case,
    settings: Settings,
    legacy: Settings,
    profile: LlmProfile,
    ocr: OcrEngine,
    run: RunResult,
    client: Any,
    retriever: Any = None,
) -> None:
    schema = resolve_schema(case.schema, case.preset, settings)
    document = case.document()
    llm = RecordingLlm(client)
    options = ExtractionOptions(include_ocr_text=True)
    try:
        result = Pipeline(settings, ocr, llm, profile, retriever).extract(
            document, None, schema, options
        )
    except LlmRateLimited as exc:
        raise QuotaExhausted(exc.retry_after) from None
    except SchemaLensError as exc:
        run.errors.append(f"{case.id}: {exc.code}")
        for name in schema["properties"]:
            run.rows.append(FieldRow(case.id, case.category, name, "miss", 0.0, 0.0, False, False))
        return
    # Same OCR, same LLM answers, scored with the base formula only: the "before" confidence.
    replay = ReplayLlm(llm.responses)
    before: ExtractionResult = Pipeline(legacy, ocr, replay, profile, retriever).extract(
        document, None, schema, options
    )
    ocr_text = "\n".join(result.ocr_text or [])
    for name, node in schema["properties"].items():
        outcome = field_outcome(
            result.data[name], case.expected[name], node, ocr_text, free_text=name in case.free_text
        )
        run.rows.append(
            FieldRow(
                case=case.id,
                category=case.category,
                field=name,
                outcome=outcome,
                confidence=result.fields[name].confidence,
                confidence_before=before.fields[name].confidence,
                grounding_warning=any(
                    w.startswith(name) and w.endswith("value not found in evidence")
                    for w in result.warnings
                ),
                too_broad=f"{name}: evidence too broad" in result.warnings,
                noise_warning=f"{name}: possible OCR noise" in result.warnings,
            )
        )
    usage = result.usage
    run.ocr_ms.append(usage.ocr_ms)
    run.llm_ms.append(usage.llm_ms)
    run.completion_tokens.append(usage.completion_tokens)
    run.tokens_per_page.append((usage.prompt_tokens + usage.completion_tokens) / usage.pages)
    run.retries += usage.llm_calls - 1
    run.escalations += usage.effort_escalations
    run.raw.append({"case": case.id, "result": result.model_dump(exclude={"ocr_text"})})


def _row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _calibration(rows: list[FieldRow], attr: str) -> str:
    """Accuracy of non-null values above and below the confidence split."""
    scored = [r for r in rows if r.outcome != "miss" and getattr(r, attr) > 0]
    high = [r for r in scored if getattr(r, attr) >= CONFIDENCE_SPLIT]
    low = [r for r in scored if getattr(r, attr) < CONFIDENCE_SPLIT]

    def cell(part: list[FieldRow]) -> str:
        if not part:
            return "— (0)"
        return f"{_pct(sum(r.outcome == 'correct' for r in part) / len(part))} ({len(part)})"

    return f"{cell(high)} | {cell(low)}"


def build_report(runs: list[RunResult], cases: list[Case], skipped: list[str]) -> str:
    out: list[str] = ["# Eval report", ""]
    out += [
        f"- Data: {datetime.now():%Y-%m-%d %H:%M}, prompt `{PROMPT_VERSION}`",
        f"- Zestaw: {len(cases)} dokumentów syntetycznych "
        f"({len({c.category for c in cases})} kategorii), OCR liczony raz i współdzielony",
        "- Raport zawiera wyłącznie identyfikatory przypadków, nazwy pól i metryki.",
        "",
    ]
    out += [f"- Pominięto: {reason}" for reason in skipped]

    out += [
        "## Przebiegi",
        "",
        "| Przebieg | Model | reasoning_effort | Dokumenty | Pola | Błędy wywołań | Z cache |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        out.append(
            f"| {run.name} | `{run.model}` | {run.reasoning_effort or '—'} | {run.documents} "
            f"| {len(run.rows)} | {len(run.errors)} | {run.cache_hits} |"
        )

    out += ["", "## Jakość (per pole)", ""]
    out += [_row("Przebieg", *OUTCOMES, "Dokumenty w 100%", "Próg 80%")]
    out += ["|---|---|---|---|---|---|---|---|"]
    for run in runs:
        counts = Counter(r.outcome for r in run.rows)
        by_case: dict[str, list[FieldRow]] = defaultdict(list)
        for row in run.rows:
            by_case[row.case].append(row)
        perfect = sum(all(r.outcome == "correct" for r in rows) for rows in by_case.values())
        verdict = "tak" if run.accuracy() >= ACCURACY_TARGET else "NIE"
        cells = " | ".join(
            f"{counts[o]} ({_pct(run.accuracy())})" if o == "correct" else str(counts[o])
            for o in OUTCOMES
        )
        out.append(f"| {run.name} | {cells} | {perfect}/{len(by_case)} | {verdict} |")

    out += ["", "## Accuracy per kategoria", "", "W nawiasie liczba pól w danym przebiegu.", ""]
    categories = sorted({c.category for c in cases})
    out += ["| Kategoria | " + " | ".join(run.name for run in runs) + " |"]
    out += ["|---|" + "---|" * len(runs)]
    for category in categories:
        cells = []
        for run in runs:
            rows = [r for r in run.rows if r.category == category]
            # A capped run may not cover a category at all.
            cells.append(f"{_pct(run.accuracy(rows))} ({len(rows)})" if rows else "—")
        out.append(f"| {category} | " + " | ".join(cells) + " |")

    out += ["", "## Czas i tokeny", ""]
    out += [
        _row(
            "Przebieg",
            "llm_ms p50",
            "llm_ms p95",
            "tokeny/stronę (śr.)",
            "tokeny odpowiedzi (śr.)",
            "Retry walidacyjne",
            "Eskalacje effort",
        )
    ]
    out += ["|---|---|---|---|---|---|---|"]
    for run in runs:
        n = max(len(run.llm_ms), 1)
        out.append(
            f"| {run.name} | {percentile(run.llm_ms, 50):.0f} | {percentile(run.llm_ms, 95):.0f} "
            f"| {sum(run.tokens_per_page) / n:.0f} | {sum(run.completion_tokens) / n:.0f} "
            f"| {run.retries} | {run.escalations} |"
        )
    ocr = runs[0].ocr_ms if runs else []
    out += [
        "",
        f"OCR (wspólny): p50 {percentile(ocr, 50):.0f} ms, p95 {percentile(ocr, 95):.0f} ms.",
    ]

    out += ["", "## Grounding i limit evidence", ""]
    out += [
        _row(
            "Przebieg",
            "Ostrzeżenia grounding",
            "w tym pole faktycznie błędne",
            "fałszywe alarmy",
            "evidence too broad",
        )
    ]
    out += ["|---|---|---|---|---|"]
    for run in runs:
        flagged = [r for r in run.rows if r.grounding_warning]
        caught = sum(r.outcome != "correct" for r in flagged)
        out.append(
            f"| {run.name} | {len(flagged)} | {caught} | {len(flagged) - caught} "
            f"| {sum(r.too_broad for r in run.rows)} |"
        )

    out += ["", "## Podejrzane tokeny (szum OCR w tekście)", ""]
    out += [
        _row(
            "Przebieg",
            "Ostrzeżenia",
            "w tym pole faktycznie błędne",
            "fałszywe alarmy",
            "błędne pola tekstowe bez ostrzeżenia",
        )
    ]
    out += ["|---|---|---|---|---|"]
    for run in runs:
        flagged = [r for r in run.rows if r.noise_warning]
        caught = sum(r.outcome != "correct" for r in flagged)
        unflagged = sum(r.outcome == "ocr_text_noise" and not r.noise_warning for r in run.rows)
        out.append(
            f"| {run.name} | {len(flagged)} | {caught} | {len(flagged) - caught} | {unflagged} |"
        )

    out += ["", "## Kalibracja confidence", ""]
    out += [
        f"Accuracy wartości nie-null dla confidence ≥ {CONFIDENCE_SPLIT} i < {CONFIDENCE_SPLIT} "
        "(w nawiasie liczba pól). „Przed” = te same odpowiedzi LLM ocenione bez kar za grounding, "
        "bez limitu evidence i bez kary za podejrzane tokeny.",
        "",
        "| Przebieg | przed: ≥0.8 | przed: <0.8 | po: ≥0.8 | po: <0.8 |",
        "|---|---|---|---|---|",
    ]
    for run in runs:
        out.append(
            f"| {run.name} | {_calibration(run.rows, 'confidence_before')} "
            f"| {_calibration(run.rows, 'confidence')} |"
        )

    out += ["", "## Pola z błędem", ""]
    out += ["| Przypadek | Pole | " + " | ".join(run.name for run in runs) + " |"]
    out += ["|---|---|" + "---|" * len(runs)]
    keys = sorted({(r.case, r.field) for run in runs for r in run.rows if r.outcome != "correct"})
    for case_id, name in keys:
        cells = []
        for run in runs:
            row = next((r for r in run.rows if (r.case, r.field) == (case_id, name)), None)
            if row is None:
                cells.append("—")
            else:
                flag = (" ⚑" if row.grounding_warning else "") + (" ~" if row.noise_warning else "")
                cells.append(f"{row.outcome} ({row.confidence:.2f}){flag}")
        out.append(f"| {case_id} | {name} | " + " | ".join(cells) + " |")
    if not keys:
        out.append("| — | — | " + " | ".join("—" for _ in runs) + " |")
    out += [
        "",
        "⚑ = ostrzeżenie grounding, ~ = ostrzeżenie o szumie OCR. W nawiasie confidence po karach.",
        "",
    ]

    failed = [f"{run.name}: {error}" for run in runs for error in run.errors]
    if failed:
        out += ["## Błędy wywołań", ""] + [f"- {error}" for error in failed] + [""]
    # Hand-maintained sections (calibration history, real documents) survive every rerun.
    for note in sorted(NOTES_DIR.glob("*.md")):
        out += [note.read_text(encoding="utf-8").rstrip(), ""]
    return "\n".join(out)


def run_profile(
    name: str,
    profile: LlmProfile,
    cases: list[Case],
    settings: Settings,
    legacy: Settings,
    ocr: OcrEngine,
    client: Any,
    retriever: Any = None,
) -> RunResult:
    """Run `cases` through one profile. Raises QuotaExhausted to stop the whole run."""
    run = RunResult(name=name, model=profile.model_fast, reasoning_effort=profile.reasoning_effort)
    for case in cases:
        run_case(case, settings, legacy, profile, ocr, run, client, retriever)
        run.documents += 1
        print(f"[{name}] {case.id}: done", file=sys.stderr)
    run.cache_hits = getattr(client, "hits", 0)
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__)
    parser.add_argument(
        "--filter", default="", help="only cases whose id or category contains this"
    )
    parser.add_argument(
        "--compare",
        default="",
        help=f"comma-separated runs, e.g. local,groq,groq:low (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=DEFAULT_MAX_DOCS,
        help="documents per hosted (non-local) run; the local profile always runs them all",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=200_000,
        help="daily token limit of the hosted provider, for the estimate printed before a run",
    )
    args = parser.parse_args(argv)

    # Hosted LLMs rate-limit bursts; evals wait out a per-minute limit. Only a quota that asks
    # for a longer wait stops the run.
    settings = Settings().model_copy(
        update={"llm_max_retries": 5, "llm_retry_backoff_s": 4.0, "llm_retry_after_max_s": 240.0}
    )
    legacy = settings.model_copy(
        update={
            "conf_grounding_factor": 1.0,
            "conf_suspicious_factor": 1.0,
            "conf_evidence_max_lines": 10_000,
            "conf_evidence_max_share": 1.0,
        }
    )
    cases = [c for c in load_cases() if args.filter in c.id or args.filter in c.category]
    if not cases:
        print("no cases match the filter", file=sys.stderr)
        return 2

    specs = [s.strip() for s in args.compare.split(",") if s.strip()] or [DEFAULT_PROFILE]
    ocr = SharedOcr(EasyOcrEngine(settings))
    retriever = build_retriever(settings)
    runs: list[RunResult] = []
    skipped: list[str] = []
    for spec in specs:
        try:
            name, profile = resolve_run(spec, settings)
        except ProfileUnavailable as exc:
            skipped.append(f"przebieg `{spec}` — {exc}")
            print(f"skipping {spec}: {exc}", file=sys.stderr)
            continue
        selected = select_cases(cases, profile, args.max_docs)
        client: Any = OpenAICompatibleClient(settings, profile)
        if profile.name != "local":
            client = CachedLlm(client, profile, CACHE_DIR)
            tokens, cached = estimate_run(selected, settings, profile, ocr, CACHE_DIR)
            print(
                f"{name}: {len(selected)} of {len(cases)} documents (--max-docs {args.max_docs}), "
                f"{cached} answered from cache, estimated ~{tokens} tokens "
                f"of a {args.token_budget} daily limit"
            )
        try:
            runs.append(
                run_profile(name, profile, selected, settings, legacy, ocr, client, retriever)
            )
        except QuotaExhausted as exc:
            wait = f"{exc.retry_after:.0f} s" if exc.retry_after else "an unknown time"
            print(
                f"{name}: STOPPED - the provider's rate limit asks to wait {wait}, which "
                "means an exhausted quota. No further documents were sent and this run is "
                "left out of the report. Answers received so far are cached; rerun later.",
                file=sys.stderr,
            )
            skipped.append(
                f"przebieg `{name}` przerwany: wyczerpany limit dostawcy (czekać {wait})"
            )
    if not runs:
        print("no run completed, the report was not changed", file=sys.stderr)
        return 1

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for run in runs:
        path = results_dir / f"{stamp}-{run.name.replace(':', '-').replace('/', '-')}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in run.raw) + "\n")
    (ROOT / "report.md").write_text(build_report(runs, cases, skipped), encoding="utf-8")
    for run in runs:
        print(f"{run.name}: field accuracy {_pct(run.accuracy())}, {len(run.errors)} call errors")
    print(f"report: {ROOT / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
