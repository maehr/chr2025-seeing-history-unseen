#!/usr/bin/env python3
# ruff: noqa
"""
Generate WCAG-compliant alternative texts using OpenRouter models.
- Fetches media metadata JSON from a URL on each run and saves a copy alongside results
- Builds prompts per media id
- Calls multiple models via OpenRouter Chat Completions API with image input
- Persists raw responses and a timestamped, wide-format Pandas table
- Uses Pydantic for strict, typed parsing of responses and metadata

Python: 3.13
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, ValidationError, field_validator


# ---------------------------
# Configuration
# ---------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 60
RUNS_DIR = Path("./runs")  # results will be nested under runs/YYYYmmdd_HHMMSS
METADATA_URL: str = (
    "https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json"
)

# Models to query (order preserved for column layout)
MODELS: list[str] = [
    "google/gemini-2.5-flash-lite",
    "mistralai/pixtral-12b",
    "openai/gpt-4.1-nano",
    "allenai/molmo-7b-d",
]

# Media ids to process
MEDIA_IDS: list[str] = [
    "m94775",
    "m27909_1",
    "m30203_1_1",
    "m29729_1",
    "m29613_1",
    "m29165",
    "m36484",
    "m30572",
    "m35321",
    "m74721",
    "m24753",
    "m36169",
    "m22924",
    "m73508",
    "m20435",
    "m92357",
    "m36710",
    "m29494",
    "m92410",
    "m37716",
    "m92966",
    "m94115",
    "m30112",
    "m40696",
    "m30541",
    "m36182",
    "m93438",
    "m13025",
    "m40043",
    "m37030_1",
    "m90411",
    "m94271",
    "m93409",
    "m37088",
    "m30171",
    "m28481",
    "m39492_1",
    "m90496_1",
    "m39198_1",
    "m40946_1",
    "m92205",
    "m30696",
    "m37490",
    "m74768",
    "m28635",
    "m37274",
    "m93036",
    "m92849",
    "m14227",
    "m28739",
    "m34278_1",
    "m91636_1",
    "m77000",
    "m82972",
    "m24255_1",
    "m16026",
    "m10333",
    "m19156",
    "m11092",
    "m94047",
    "m25143",
    "m21867",
    "m19033",
    "m13176",
    "m15630",
    "m23902",
    "m16683",
    "m12965",
    "m35209",
    "m95804",
    "m33188",
    "m91000_1",
    "m95513",
    "m91960",
    "m94712",
    "m95085",
    "m92237_1",
    "m90886_1",
    "m94575",
    "m27970",
    "m22355_1",
    "m85061_1",
    "m15298_1",
    "m11589_1",
    "m34620",
    "m26375_1",
    "m82338_1",
    "m88415_1",
    "m30849",
    "m92771",
    "m93459",
    "m33441",
    "m37696",
    "m37644",
    "m33472",
    "m89606",
    "m91713",
    "m29025",
    "m40938",
    "m29084",
]


# ---------------------------
# Pydantic models
# ---------------------------


class MediaObject(BaseModel):
    objectid: str
    parentid: Optional[str] = None
    title: Optional[str] = "Kein Titel"
    description: Optional[str] = "Keine Beschreibung"
    relation: Optional[List[Any]] = None
    coverage: Optional[str] = None
    isPartOf: Optional[List[Any]] = None
    creator: Optional[List[str] | str] = "Kein Ersteller"
    publisher: Optional[List[str] | str] = "Kein Herausgeber"
    source: Optional[List[str] | str] = "Keine Quelle"
    date: Optional[str] = "Kein Datum"
    type: Optional[str] = None
    format: Optional[str] = None
    extent: Optional[str] = None
    language: Optional[str] = None
    rights: Optional[str] = None
    license: Optional[str] = None
    object_location: Optional[HttpUrl] = None
    image_alt_text: Optional[str] = ""
    object_thumb: Optional[HttpUrl] = None
    reference_url: Optional[HttpUrl] = None
    era: Optional[str] = Field(default=None, alias="Epoche")

    @field_validator("creator", "publisher", "source")
    @classmethod
    def _normalize_seq_or_str(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, Sequence):
            return ", ".join(str(x) for x in v)
        return str(v)


class ORMessageContent(BaseModel):
    role: str
    content: str
    refusal: Optional[str] = None
    reasoning: Optional[str] = None


class ORChoice(BaseModel):
    index: int
    finish_reason: Optional[str] = None
    native_finish_reason: Optional[str] = None
    message: ORMessageContent


class ORPromptTokensDetails(BaseModel):
    cached_tokens: Optional[int] = 0
    audio_tokens: Optional[int] = 0


class ORCompletionTokensDetails(BaseModel):
    reasoning_tokens: Optional[int] = 0
    image_tokens: Optional[int] = 0


class ORCostDetails(BaseModel):
    upstream_inference_cost: Optional[float] = None
    upstream_inference_prompt_cost: Optional[float] = None
    upstream_inference_completions_cost: Optional[float] = None


class ORUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: Optional[float] = None
    is_byok: Optional[bool] = None
    prompt_tokens_details: Optional[ORPromptTokensDetails] = None
    completion_tokens_details: Optional[ORCompletionTokensDetails] = None
    cost_details: Optional[ORCostDetails] = None


class ORCompletion(BaseModel):
    id: str
    provider: Optional[str] = None
    model: str
    object: Optional[str] = None
    created: Optional[int] = None
    choices: List[ORChoice]
    usage: Optional[ORUsage] = None


class RowWide(BaseModel):
    objectid: str
    prompt: str
    # dynamic model-specific fields are added at runtime


# ---------------------------
# Helpers
# ---------------------------


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def mk_run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(exist_ok=True)
    return run_dir


def fetch_and_save_metadata(url: str, run_dir: Path) -> dict[str, Any]:
    """Fetch metadata JSON from URL, save exact bytes for provenance, return parsed dict."""
    r = requests.get(url, timeout=TIMEOUT_SECONDS)
    if r.status_code != 200:
        raise FileNotFoundError(f"Failed to fetch metadata [{r.status_code}]: {url}")
    # Save pristine copy for this run
    meta_path = run_dir / "metadata.json"
    meta_path.write_bytes(r.content)
    try:
        payload = r.json()
    except Exception as e:
        raise ValueError(f"Metadata is not valid JSON: {e}") from e
    return payload


def load_db_from_payload(payload: dict[str, Any]) -> dict[str, MediaObject]:
    objects = payload.get("objects", [])
    out: dict[str, MediaObject] = {}
    for obj in objects:
        try:
            mo = MediaObject.model_validate(obj)
            out[mo.objectid] = mo
        except ValidationError as e:
            raise ValueError(f"Invalid media object: {e}") from e
    return out


def build_prompt(media: MediaObject) -> str:
    title = media.title or "Kein Titel"
    description = media.description or "Keine Beschreibung"
    date = media.date or "Kein Datum"
    era = media.era or media.coverage or "Keine Epoche"
    creator = media.creator or "Kein Ersteller"
    publisher = media.publisher or "Kein Herausgeber"
    source = media.source or "Keine Quelle"

    return f"""
**Prompt (für vLLM oder ähnliche Modelle):**

Schreibe barrierefreie Alternativtexte für Bilder nach WCAG 2.2 (SC 1.1.1) und W3C WAI-Standards.

**Ziel:** Funktionale, kontextabhängige, prägnante Alt-Texte.

**Regeln:**

* Beschreibe Bedeutung, nicht Aussehen.
* Max. 125 Zeichen, nur was für Inhalt/Zweck relevant ist.
* Für dekorative Bilder: `alt=""`.
* Für funktionale Bilder: Funktion oder Ziel nennen („Zum Warenkorb“).
* Keine Formulierungen wie „Bild von“ oder „Foto von“.
* Für komplexe Grafiken: Ausführliche Erklärung.

**Ausgabe:** Nur der Alternativtext (kein HTML, keine Erklärungen).

**Eingabe:**

* Nutzungskontext: Plattform mit historischen Quellen und Forschungsdaten für Forschende und Studierende aus historischen und archäologischen Disziplinen.
* Metadaten:
  * Titel: {title}
  * Beschreibung: {description}
  * Datum (Extended Date Time Format): {date}
  * Epoche: {era}
  * Ersteller: {creator}
  * Herausgeber: {publisher}
  * Quelle: {source}
""".strip()


def build_messages(prompt: str, image_url: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]


def call_openrouter(
    api_key: str, model: str, messages: list[dict[str, Any]]
) -> ORCompletion:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "usage": {"include": True}}
    resp = requests.post(
        OPENROUTER_URL, headers=headers, json=payload, timeout=TIMEOUT_SECONDS
    )
    if resp.status_code != 200:
        raise ValueError(f"Request failed [{resp.status_code}]: {resp.text[:1000]}")
    data = resp.json()
    try:
        return ORCompletion.model_validate(data)
    except ValidationError as e:
        raise ValueError(
            f"Response schema mismatch for model {model}: {e}\nRaw: {json.dumps(data)[:1200]}"
        ) from e


def ensure_env() -> str:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found. Set it in your environment or .env."
        )
    return api_key


def init_wide_row(objectid: str, prompt: str) -> dict[str, Any]:
    base = RowWide(objectid=objectid, prompt=prompt).model_dump()
    # Pre-create per-model columns to keep stable schema
    for m in MODELS:
        prefix = m.replace("/", "__")
        base[f"{prefix}__content"] = None
        base[f"{prefix}__finish_reason"] = None
        base[f"{prefix}__usage_prompt_tokens"] = None
        base[f"{prefix}__usage_completion_tokens"] = None
        base[f"{prefix}__usage_total_tokens"] = None
        base[f"{prefix}__usage_cost"] = None
        base[f"{prefix}__provider"] = None
        base[f"{prefix}__created"] = None
        base[f"{prefix}__id"] = None
    base["created_utc"] = utc_now_iso()
    return base


def persist_json(path: Path, data: dict | list | str | int | float | None) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------
# Main
# ---------------------------


def main() -> None:
    # Env
    api_key = ensure_env()

    # Run directory first, so we can store fetched metadata
    run_dir = mk_run_dir()
    raw_dir = run_dir / "raw"

    # Fetch metadata from URL and save exact copy for provenance
    payload = fetch_and_save_metadata(METADATA_URL, run_dir)
    db = load_db_from_payload(payload)

    # Validate requested IDs exist
    missing = [mid for mid in MEDIA_IDS if mid not in db]
    if missing:
        raise KeyError(f"Missing media ids in metadata: {missing}")

    # Prepare table with one row per media id
    rows: list[dict[str, Any]] = []

    for mid in MEDIA_IDS:
        media = db[mid]
        if not media.object_thumb:
            raise ValueError(f"No thumbnail URL found for media ID {mid}")

        prompt = build_prompt(media)
        row = init_wide_row(objectid=mid, prompt=prompt)

        messages = build_messages(prompt=prompt, image_url=str(media.object_thumb))

        for model in MODELS:
            try:
                orc = call_openrouter(api_key=api_key, model=model, messages=messages)
            except Exception as e:
                prefix = model.replace("/", "__")
                row[f"{prefix}__content"] = f"[ERROR] {type(e).__name__}: {e}"
                continue

            # Save raw response
            raw_path = raw_dir / f"{mid}__{model.replace('/', '__')}.json"
            persist_json(raw_path, orc.model_dump(mode="json"))

            # Extract first choice content
            content = ""
            finish_reason = None
            if orc.choices:
                content = orc.choices[0].message.content
                finish_reason = orc.choices[0].finish_reason

            prefix = model.replace("/", "__")
            row[f"{prefix}__content"] = content
            row[f"{prefix}__finish_reason"] = finish_reason
            row[f"{prefix}__provider"] = orc.provider
            row[f"{prefix}__created"] = orc.created
            row[f"{prefix}__id"] = orc.id

            if orc.usage:
                row[f"{prefix}__usage_prompt_tokens"] = orc.usage.prompt_tokens
                row[f"{prefix}__usage_completion_tokens"] = orc.usage.completion_tokens
                row[f"{prefix}__usage_total_tokens"] = orc.usage.total_tokens
                row[f"{prefix}__usage_cost"] = orc.usage.cost

            time.sleep(0.2)

        rows.append(row)

    # Build DataFrame (wide format)
    df = pd.DataFrame(rows)

    # Persist artifacts with timestamp
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_base = run_dir / f"alt_text_runs_{stamp}"
    df.to_csv(f"{table_base}.csv", index=False, encoding="utf-8")
    try:
        df.to_parquet(f"{table_base}.parquet", index=False)
    except Exception:
        pass

    with (Path(f"{table_base}.jsonl")).open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "created_utc": utc_now_iso(),
        "runs_dir": str(run_dir.resolve()),
        "metadata_url": METADATA_URL,
        "saved_metadata": str((run_dir / "metadata.json").resolve()),
        "models": MODELS,
        "media_ids": MEDIA_IDS,
        "table_csv": str(Path(f"{table_base}.csv").resolve()),
        "table_parquet": str(Path(f"{table_base}.parquet").resolve()),
        "table_jsonl": str(Path(f"{table_base}.jsonl").resolve()),
        "python_version": sys.version,
    }
    persist_json(run_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
