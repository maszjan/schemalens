"""Runtime configuration. The only place that reads environment variables."""

from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from schemalens_core.errors import ProfileUnavailable

ProfileName = Literal["groq", "local", "openai"]
ReasoningEffort = Literal["low", "medium", "high"]


class LlmProfile(BaseModel):
    """Everything needed to talk to one LLM endpoint. Resolved by `Settings.llm_profile()`."""

    model_config = ConfigDict(frozen=True)

    name: ProfileName
    base_url: str
    api_key: SecretStr
    model_fast: str
    model_accurate: str
    timeout_s: float
    strict_schema: bool  # True: provider does constrained decoding (Groq gpt-oss)
    direct_limit: int  # most document tokens sent in one request
    # Cap on the answer. None for hosted reasoning models, whose limit also counts reasoning.
    max_output_tokens: int | None = None
    # Whole context of the model (prompt + document + answer). None = large enough that
    # only `direct_limit` matters (hosted models).
    context_window: int | None = None
    reasoning_effort: ReasoningEffort | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM profiles . Any OpenAI-compatible endpoint; model names have no defaults.
    active_llm_profile: ProfileName = "local"

    # groq: hosted, constrained decoding. Optional: nothing may depend on it being configured.
    groq_api_key: SecretStr = SecretStr("")
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model_fast: str = ""
    groq_model_accurate: str = ""
    groq_timeout_s: float = 60.0
    groq_strict_schema: bool = True
    groq_direct_limit: int = 24_000  # document tokens per request
    groq_reasoning_effort: ReasoningEffort | None = "medium"

    # local: Ollama on this machine, the default. The only profile allowed to see private documents.
    local_base_url: str = "http://localhost:11434/v1"
    local_model_fast: str = ""
    local_model_accurate: str = ""
    local_timeout_s: float = 180.0
    local_strict_schema: bool = False
    # Leaves room for the prompt and the answer in an 8192-token context (OLLAMA_CONTEXT_LENGTH).
    local_direct_limit: int = 6_000
    # A local model that loses the thread keeps generating until its context is full,
    # which takes minutes. No extraction answer is that long.
    local_max_output_tokens: int = 2_000
    # OLLAMA_CONTEXT_LENGTH of the local model. The document gets what is left after the
    # prompt and the reserved answer, so a long document never overflows the context.
    local_context_window: int = 8_192

    # openai: bring your own LLM - any OpenAI-compatible endpoint (OpenAI, Azure OpenAI, vLLM,
    # LM Studio, OpenRouter...). Optional, like groq.
    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model_fast: str = ""
    openai_model_accurate: str = ""
    openai_timeout_s: float = 60.0
    # json_schema strict mode; turn it off for servers that only take free-form JSON
    openai_strict_schema: bool = True
    openai_direct_limit: int = 24_000
    openai_context_window: int | None = None
    openai_max_output_tokens: int | None = None
    openai_reasoning_effort: ReasoningEffort | None = None

    # reasoning_effort is sent only to models whose name contains one of these fragments
    reasoning_effort_models: tuple[str, ...] = ("gpt-oss",)
    token_estimate_chars: float = 3.5  # characters per token, for the context estimate

    llm_max_retries: int = 2  # retries on 429/5xx/timeout and failed constrained generation
    llm_retry_backoff_s: float = 1.0
    llm_retry_after_max_s: float = 60.0  # longer Retry-After on 429 = quota: fail, don't wait

    # Field confidence (docs/how-it-works.md)
    conf_ocr_weight: float = 0.7  # share of OCR line confidence vs model-reported confidence
    conf_no_evidence_factor: float = 0.5
    conf_normalized_factor: float = 0.9
    conf_retry_factor: float = 0.85
    conf_checksum_factor: float = 0.5
    conf_derived_cap: float = 0.7  # ceiling for derived fields without evidence
    conf_grounding_factor: float = 0.3  # date/number/identifier not found in its evidence
    conf_suspicious_factor: float = 0.6  # plain text with tokens typical of OCR misreads
    conf_evidence_max_lines: int = 3  # per scalar field, or per element of a list
    conf_evidence_max_share: float = 0.5  # of all document lines, for scalar fields

    # Retrieval for documents longer than the model context (docs/how-it-works.md). The embedding
    # model is read from disk only (baked into the image, or saved there once locally).
    rag_enabled: bool = True
    embed_model_dir: Path = Path(".models/embed")
    rag_chunk_lines: int = 3
    rag_chunk_overlap: int = 1
    rag_head_lines: int = 15  # always sent: title, parties, numbers, dates
    rag_tail_lines: int = 5  # sent when the schema has derived fields (summaries)

    # Few-shot: a worked example per preset from templates/ (off until evals show it helps)
    fewshot_enabled: bool = False
    templates_dir: Path = Path("templates")

    # Presets (see presets/)
    presets_dir: Path = Path("presets")

    # Client schema limits (docs/api.md)
    schema_max_fields: int = 50
    schema_max_depth: int = 3

    # OCR
    ocr_langs: tuple[str, ...] = ("pl", "en")
    ocr_model_dir: Path = Path(".models/easyocr")
    ocr_gpu: bool = False
    ocr_allow_download: bool = False
    ocr_line_overlap: float = 0.5
    # Quality gate before the LLM (docs/how-it-works.md). On synthetic scans readable documents
    # score a mean confidence of 0.70-0.93 (0.72 even at a third of the resolution), unreadable ones
    # (heavy blur, tiny photos, scribbles) 0.01-0.36 or yield a single stray line.
    ocr_min_lines: int = 2
    ocr_min_mean_conf: float = 0.5

    # Preprocessing
    max_pages: int = 5
    max_image_side: int = 2000
    max_image_pixels: int = 50_000_000
    pdf_render_dpi: int = 200
    pdf_text_min_chars: int = 20
    pdf_text_min_alnum_ratio: float = 0.5

    def llm_profile(self, name: str | None = None) -> LlmProfile:
        """The named LLM profile, or the active one (`ACTIVE_LLM_PROFILE`) when no name is given."""
        name = name or self.active_llm_profile
        if name == "groq":
            if not self.groq_api_key.get_secret_value():
                raise ProfileUnavailable("LLM profile 'groq' is not configured: set GROQ_API_KEY")
            return LlmProfile(
                name="groq",
                base_url=self.groq_base_url,
                api_key=self.groq_api_key,
                model_fast=self.groq_model_fast,
                model_accurate=self.groq_model_accurate,
                timeout_s=self.groq_timeout_s,
                strict_schema=self.groq_strict_schema,
                direct_limit=self.groq_direct_limit,
                reasoning_effort=self.groq_reasoning_effort,
            )
        if name == "local":
            return LlmProfile(
                name="local",
                base_url=self.local_base_url,
                api_key=SecretStr("ollama"),  # Ollama ignores the key, the SDK requires one
                model_fast=self.local_model_fast,
                model_accurate=self.local_model_accurate,
                timeout_s=self.local_timeout_s,
                strict_schema=self.local_strict_schema,
                direct_limit=self.local_direct_limit,
                max_output_tokens=self.local_max_output_tokens,
                context_window=self.local_context_window,
            )
        if name == "openai":
            if not self.openai_model_fast:
                raise ProfileUnavailable(
                    "LLM profile 'openai' is not configured: set OPENAI_MODEL_FAST "
                    "(and OPENAI_BASE_URL / OPENAI_API_KEY for your endpoint)"
                )
            return LlmProfile(
                name="openai",
                base_url=self.openai_base_url,
                # some self-hosted servers need no key, the SDK requires one
                api_key=self.openai_api_key
                if self.openai_api_key.get_secret_value()
                else SecretStr("none"),
                model_fast=self.openai_model_fast,
                model_accurate=self.openai_model_accurate,
                timeout_s=self.openai_timeout_s,
                strict_schema=self.openai_strict_schema,
                direct_limit=self.openai_direct_limit,
                max_output_tokens=self.openai_max_output_tokens,
                context_window=self.openai_context_window,
                reasoning_effort=self.openai_reasoning_effort,
            )
        raise ProfileUnavailable(f"unknown LLM profile '{name}' (available: groq, local, openai)")


@cache
def get_settings() -> Settings:
    return Settings()
