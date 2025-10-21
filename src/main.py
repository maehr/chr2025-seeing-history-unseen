#!/usr/bin/env python3
# ruff: noqa
from __future__ import annotations
import mimetypes, csv, json, os, sys, time, itertools, random, hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from time import perf_counter

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, ValidationError, field_validator

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 60
RUNS_DIR = Path("./runs")
METADATA_URL: str = (
    "https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json"
)
MODELS: list[str] = [
    "google/gemini-2.5-flash-lite",
    "mistralai/pixtral-12b",
    "openai/gpt-4o-mini",
    "meta-llama/llama-4-maverick",
]
MODE = "subsample"  # "testing" | "subsample" | "full"

MEDIA_IDS: list[str] = []
if MODE == "testing":
    MEDIA_IDS = ["m94775", "m27909_1", "m30203_1_1"]
if MODE == "subsample":
    MEDIA_IDS = [
        "m91000_1",
        "m94271",
        "m95804",
        "m94775",
        "m88415_1",
        "m15298_1",
        "m34620",
        "m92410",
        "m92357",
        "m82972",
        "m37716",
        "m22924",
        "m13176",
        "m12965",
        "m28635",
        "m29084",
        "m37030_1",
        "m39198_1",
    ]
if MODE == "full":
    MEDIA_IDS = [
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
RANDOM_SEED = 42


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


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def mk_run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(exist_ok=True)
    (run_dir / "images").mkdir(exist_ok=True)
    return run_dir


def fetch_and_save_metadata(url: str, run_dir: Path) -> dict[str, Any]:
    r = requests.get(url, timeout=TIMEOUT_SECONDS)
    if r.status_code != 200:
        raise FileNotFoundError(f"Failed to fetch metadata [{r.status_code}]: {url}")
    (run_dir / "metadata.json").write_bytes(r.content)
    try:
        return r.json()
    except Exception as e:
        raise ValueError(f"Metadata is not valid JSON: {e}") from e


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
    json_parts = {
        "Titel": media.title or "",
        "Beschreibung": media.description or "",
        "Ersteller": media.creator or "",
        "Herausgeber": media.publisher or "",
        "Quelle": media.source or "",
        "Datum": media.date or "",
        "Epoche": media.era or "",
    }
    return json.dumps(json_parts, ensure_ascii=False)


def build_messages(
    prompt: str, image_url: str
) -> Tuple[list[dict[str, Any]], str, str]:
    system = """ZIEL

Alt-Texte für historische und archäologische Sammlungsbilder.
Kurz, sachlich, zugänglich. Erfassung der visuellen Essenz für Screenreader.

REGELN

1. Essenz statt Detail. Keine Redundanz zum Seitentext, kein „Bild von“.
2. Zentralen Text im Bild wiedergeben oder kurz paraphrasieren.
3. Kontext (Epoche, Ort, Gattung, Material, Datierung) nur bei Relevanz für Verständnis.
4. Prägnante visuelle Merkmale nennen: Farbe, Haltung, Zustand, Attribute.
5. Karten/Diagramme: zentrale Aussage oder Variablen.
6. Sprache: neutral, präzise, faktenbasiert; keine Wertung, keine Spekulation.
7. Umfang:
   * Standard: 90–180 Zeichen
   * Komplexe Karten/Tabellen: max. 400 Zeichen

VERBOTE

* Kein alt=, Anführungszeichen, Preambeln oder Füllwörter („zeigt“, „darstellt“).
* Keine offensichtlichen Metadaten (z. B. Jahreszahlen aus Beschriftung).
* Keine Bewertungen, Hypothesen oder Stilkommentare.
* Keine Emojis oder emotionalen Begriffe.

HEURISTIKEN

Porträt: Person (Name, falls bekannt), Epoche, Pose oder Attribut, ggf. Funktion.
Objekt: Gattung, Material, Datierung, auffällige Besonderheit.
Dokument: Typ, Sprache/Schrift, Datierung, Kernaussage.
Karte: Gebiet, Zeitraum, Zweck, Hauptvariablen.
Ereignisfoto: Wer, was, wo, situativer Kontext.
Plakat/Cover: Titel, Zweck, zentrale Schlagzeile.

FALLBACK

Unklarer Inhalt: generische, aber sinnvolle Essenz aus Metadaten.

QUELLEN

Nur visuelle Analyse (Bildinhalt) und übergebene Metadaten. Keine externen Kontexte.""".strip()
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]
    return messages, system, prompt


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    session: Optional[requests.Session] = None,
) -> ORCompletion:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "usage": {"include": True}}
    resp = (session or requests).post(
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
    for m in MODELS:
        p = m.replace("/", "__")
        base[f"{p}__content"] = None
        base[f"{p}__finish_reason"] = None
        base[f"{p}__usage_prompt_tokens"] = None
        base[f"{p}__usage_completion_tokens"] = None
        base[f"{p}__usage_total_tokens"] = None
        base[f"{p}__usage_cost"] = None
        base[f"{p}__provider"] = None
        base[f"{p}__created"] = None
        base[f"{p}__id"] = None
        base[f"{p}__request_started_utc"] = None
        base[f"{p}__request_ended_utc"] = None
        base[f"{p}__latency_seconds"] = None
    base["created_utc"] = utc_now_iso()
    return base


def persist_json(path: Path, data: dict | list | str | int | float | None) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_image_to_folder(url: str, images_dir: Path, basename: str) -> str:
    r = requests.get(url, timeout=TIMEOUT_SECONDS)
    if r.status_code != 200:
        raise FileNotFoundError(f"Failed to fetch image [{r.status_code}]: {url}")
    mime = r.headers.get("Content-Type") or mimetypes.guess_type(url)[0] or "image/jpeg"
    ext = mimetypes.guess_extension(mime) or ".jpg"
    if ext in (".jpeg", ".jpe"):
        ext = ".jpg"
    fname = f"{basename}{ext}"
    (images_dir / fname).write_bytes(r.content)
    return f"images/{fname}"


def main() -> None:
    api_key = ensure_env()
    run_t0 = perf_counter()
    run_dir = mk_run_dir()
    raw_dir, images_dir = run_dir / "raw", run_dir / "images"
    session = requests.Session()

    payload = fetch_and_save_metadata(METADATA_URL, run_dir)
    db = load_db_from_payload(payload)
    missing = [mid for mid in MEDIA_IDS if mid not in db]
    if missing:
        raise KeyError(f"Missing media ids in metadata: {missing}")

    rows: list[dict[str, Any]] = []
    prompt_records: list[dict[str, str]] = []

    for mid in MEDIA_IDS:
        media = db[mid]
        if not media.object_thumb:
            raise ValueError(f"No thumbnail URL found for media ID {mid}")

        prompt = build_prompt(media)
        row = init_wide_row(objectid=mid, prompt=prompt)
        messages, system_prompt, user_prompt = build_messages(
            prompt=prompt, image_url=str(media.object_thumb)
        )
        row["system_prompt"] = system_prompt
        row["user_prompt"] = user_prompt
        prompt_records.append(
            {
                "objectid": mid,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "image_url": str(media.object_thumb),
            }
        )

        for model in MODELS:
            prefix = model.replace("/", "__")
            t0 = perf_counter()
            started_iso = utc_now_iso()
            try:
                orc = call_openrouter(
                    api_key=api_key, model=model, messages=messages, session=session
                )
                ended_iso = utc_now_iso()
                t1 = perf_counter()
                persist_json(
                    raw_dir / f"{mid}__{prefix}.json", orc.model_dump(mode="json")
                )
                content = ""
                finish_reason = None
                if orc.choices:
                    content = orc.choices[0].message.content
                    finish_reason = orc.choices[0].finish_reason
                row[f"{prefix}__content"] = content
                row[f"{prefix}__finish_reason"] = finish_reason
                row[f"{prefix}__provider"] = orc.provider
                row[f"{prefix}__created"] = orc.created
                row[f"{prefix}__id"] = orc.id
                if orc.usage:
                    row[f"{prefix}__usage_prompt_tokens"] = orc.usage.prompt_tokens
                    row[f"{prefix}__usage_completion_tokens"] = (
                        orc.usage.completion_tokens
                    )
                    row[f"{prefix}__usage_total_tokens"] = orc.usage.total_tokens
                    row[f"{prefix}__usage_cost"] = orc.usage.cost
                row[f"{prefix}__request_started_utc"] = started_iso
                row[f"{prefix}__request_ended_utc"] = ended_iso
                row[f"{prefix}__latency_seconds"] = round(t1 - t0, 6)
            except Exception as e:
                ended_iso = utc_now_iso()
                t1 = perf_counter()
                row[f"{prefix}__content"] = f"[ERROR] {type(e).__name__}: {e}"
                row[f"{prefix}__request_started_utc"] = started_iso
                row[f"{prefix}__request_ended_utc"] = ended_iso
                row[f"{prefix}__latency_seconds"] = round(t1 - t0, 6)
                continue
            time.sleep(0.2)

        rows.append(row)

        try:
            save_image_to_folder(str(media.object_thumb), images_dir, basename=mid)
        except Exception as e:
            _ = f"[ERROR fetching image: {type(e).__name__}: {e}]"

    df = pd.DataFrame(rows)
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

    prompts_json = run_dir / "prompts.json"
    persist_json(
        prompts_json,
        {
            "created_utc": utc_now_iso(),
            "system_prompt": prompt_records[0]["system_prompt"]
            if prompt_records
            else "",
            "per_media": prompt_records,
        },
    )

    run_t1 = perf_counter()
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
        "prompts_json": str(prompts_json.resolve()),
        "python_version": sys.version,
        "two_afc_pairs_per_media": len(list(itertools.combinations(MODELS, 2))),
        "random_seed": RANDOM_SEED,
        "images_dir": str((run_dir / "images").resolve()),
        "run_wall_time_seconds": round(run_t1 - run_t0, 6),
    }
    persist_json(run_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()


# TODO:

# - Add heuristic checks for obviously bad alt-texts (e.g., too short/too long, "image of", etc.)
# - Add heuristic for complex images (e.g., presence of "chart", "diagram", "map" in metadata)
