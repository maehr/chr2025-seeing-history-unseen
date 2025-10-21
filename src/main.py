#!/usr/bin/env python3
# ruff: noqa
from __future__ import annotations

import hashlib, json, logging, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, List, Optional, Sequence, Tuple

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
MODE: str = "testing"  # "testing" | "subsample" | "full"

MODE_TO_MEDIA_IDS: dict[str, list[str]] = {
    "testing": ["m94775", "m27909_1", "m30203_1_1"],
    "subsample": [
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
    ],
    "full": [
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
    ],
}
MEDIA_IDS = MODE_TO_MEDIA_IDS[MODE]

COLS = [
    "content",
    "finish_reason",
    "usage_prompt_tokens",
    "usage_completion_tokens",
    "usage_total_tokens",
    "usage_cost",
    "provider",
    "created",
    "id",
    "request_started_utc",
    "request_ended_utc",
    "latency_seconds",
    "error",
]


def col(prefix: str, name: str) -> str:
    return f"{prefix}__{name}"


logging.basicConfig(
    level=logging.INFO, format="%(asctime)sZ\t%(levelname)s\t%(message)s"
)


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
    prompt_sha256: str


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def sha256_utf8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def mk_run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = RUNS_DIR / ts
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw").mkdir(exist_ok=True)
    (d / "images").mkdir(exist_ok=True)
    return d


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def atomic_write_df_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)


def fetch_and_save_metadata(
    url: str, run_dir: Path, session: requests.Session
) -> dict[str, Any]:
    r = session.get(url, timeout=TIMEOUT_SECONDS)
    if r.status_code != 200:
        raise FileNotFoundError(f"Failed to fetch metadata [{r.status_code}]: {url}")
    atomic_write_bytes(run_dir / "metadata.json", r.content)
    try:
        return r.json()
    except Exception as e:
        raise ValueError(f"Metadata is not valid JSON: {e}") from e


def load_db_from_payload(payload: dict[str, Any]) -> dict[str, MediaObject]:
    out: dict[str, MediaObject] = {}
    for obj in payload.get("objects", []):
        try:
            out[MediaObject.model_validate(obj).objectid] = MediaObject.model_validate(
                obj
            )
        except ValidationError as e:
            raise ValueError(f"Invalid media object: {e}") from e
    return out


def build_prompt(media: MediaObject) -> str:
    return json.dumps(
        {
            "Titel": media.title or "",
            "Beschreibung": media.description or "",
            "Ersteller": media.creator or "",
            "Herausgeber": media.publisher or "",
            "Quelle": media.source or "",
            "Datum": media.date or "",
            "Epoche": media.era or "",
        },
        ensure_ascii=False,
    )


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
    return (
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        system,
        prompt,
    )


def ensure_env() -> str:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found. Set it in your environment or .env."
        )
    return api_key


def init_wide_row(objectid: str, prompt_sha: str) -> dict[str, Any]:
    base = RowWide(objectid=objectid, prompt_sha256=prompt_sha).model_dump()
    for m in MODELS:
        p = m.replace("/", "__")
        for k in COLS:
            base[col(p, k)] = None
    base["created_utc"] = utc_now_iso()
    return base


def persist_json(path: Path, data: dict | list | str | int | float | None) -> None:
    atomic_write_text(
        path, json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_image_to_folder(
    url: str, images_dir: Path, basename: str, session: requests.Session
) -> str:
    r = session.get(url, timeout=TIMEOUT_SECONDS)
    if r.status_code != 200:
        raise FileNotFoundError(f"Failed to fetch image [{r.status_code}]: {url}")
    ct = (r.headers.get("Content-Type") or "").lower()
    if ct not in ("image/jpeg", "image/jpg"):
        raise ValueError(
            f"Unexpected Content-Type '{ct}' for {url}. Only JPEG allowed."
        )
    dest = images_dir / f"{basename}.jpg"
    atomic_write_bytes(dest, r.content)
    return f"images/{dest.name}"


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    session: Optional[requests.Session] = None,
    max_attempts: int = 5,
) -> ORCompletion:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "usage": {"include": True}}
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = (session or requests).post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=TIMEOUT_SECONDS
            )
        except Exception as e:
            logging.error(
                "openrouter request error attempt=%d model=%s error=%s",
                attempt,
                model,
                repr(e),
            )
            if attempt >= max_attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
            continue
        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            try:
                wait_s = int(float(ra)) if ra else min(2 ** (attempt - 1), 16)
            except ValueError:
                wait_s = 2
            logging.warning(
                "rate limited model=%s attempt=%d retry_after=%ss",
                model,
                attempt,
                wait_s,
            )
            if attempt >= max_attempts:
                raise ValueError(
                    f"OpenRouter 429 after {attempt} attempts for {model}: {resp.text[:300]}"
                )
            time.sleep(wait_s)
            continue
        if resp.status_code != 200:
            logging.error(
                "openrouter http error model=%s status=%s body=%s",
                model,
                resp.status_code,
                resp.text[:500],
            )
            if attempt >= max_attempts or 400 <= resp.status_code < 500:
                raise ValueError(
                    f"Request failed [{resp.status_code}]: {resp.text[:1000]}"
                )
            time.sleep(min(2 ** (attempt - 1), 8))
            continue
        data = resp.json()
        try:
            return ORCompletion.model_validate(data)
        except ValidationError as e:
            logging.error(
                "schema mismatch model=%s error=%s raw_prefix=%s",
                model,
                repr(e),
                json.dumps(data)[:500],
            )
            raise ValueError(
                f"Response schema mismatch for model {model}: {e}\nRaw: {json.dumps(data)[:1200]}"
            ) from e


def main() -> None:
    api_key = ensure_env()
    run_t0 = perf_counter()
    run_dir = mk_run_dir()
    raw_dir, images_dir = run_dir / "raw", run_dir / "images"
    logfile = run_dir / "run.log"
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)sZ\t%(levelname)s\t%(message)s"))
    logging.getLogger().addHandler(fh)

    session = requests.Session()
    payload = fetch_and_save_metadata(METADATA_URL, run_dir, session)
    db = load_db_from_payload(payload)
    missing = [mid for mid in MEDIA_IDS if mid not in db]
    if missing:
        raise KeyError(f"Missing media ids in metadata: {missing}")

    rows_wide: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, str]] = []

    for mid in MEDIA_IDS:
        media = db[mid]
        if not media.object_thumb:
            raise ValueError(f"No thumbnail URL for media ID {mid}")
        prompt = build_prompt(media)
        prompt_sha = sha256_utf8(prompt)
        messages, system_prompt, _ = build_messages(
            prompt=prompt, image_url=str(media.object_thumb)
        )
        prompt_rows.append(
            {
                "objectid": mid,
                "prompt_sha256": prompt_sha,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "image_url": str(media.object_thumb),
            }
        )
        row = init_wide_row(objectid=mid, prompt_sha=prompt_sha)

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
                content = orc.choices[0].message.content if orc.choices else ""
                finish_reason = orc.choices[0].finish_reason if orc.choices else None
                row[col(prefix, "content")] = content
                row[col(prefix, "finish_reason")] = finish_reason
                row[col(prefix, "provider")] = orc.provider
                row[col(prefix, "created")] = orc.created
                row[col(prefix, "id")] = orc.id
                if orc.usage:
                    row[col(prefix, "usage_prompt_tokens")] = orc.usage.prompt_tokens
                    row[col(prefix, "usage_completion_tokens")] = (
                        orc.usage.completion_tokens
                    )
                    row[col(prefix, "usage_total_tokens")] = orc.usage.total_tokens
                    row[col(prefix, "usage_cost")] = orc.usage.cost
                row[col(prefix, "request_started_utc")] = started_iso
                row[col(prefix, "request_ended_utc")] = ended_iso
                row[col(prefix, "latency_seconds")] = round(t1 - t0, 6)
                long_rows.append(
                    {
                        "objectid": mid,
                        "prompt_sha256": prompt_sha,
                        "model": model,
                        "provider": orc.provider,
                        "created": orc.created,
                        "request_started_utc": started_iso,
                        "request_ended_utc": ended_iso,
                        "latency_seconds": round(t1 - t0, 6),
                        "finish_reason": finish_reason,
                        "content": content,
                        "usage_prompt_tokens": orc.usage.prompt_tokens
                        if orc.usage
                        else None,
                        "usage_completion_tokens": orc.usage.completion_tokens
                        if orc.usage
                        else None,
                        "usage_total_tokens": orc.usage.total_tokens
                        if orc.usage
                        else None,
                        "usage_cost": orc.usage.cost if orc.usage else None,
                        "error": None,
                    }
                )
            except Exception as e:
                ended_iso = utc_now_iso()
                t1 = perf_counter()
                err = f"{type(e).__name__}: {e}"
                logging.error(
                    "model call failed objectid=%s model=%s error=%s", mid, model, err
                )
                row[col(prefix, "error")] = err
                row[col(prefix, "request_started_utc")] = started_iso
                row[col(prefix, "request_ended_utc")] = ended_iso
                row[col(prefix, "latency_seconds")] = round(t1 - t0, 6)
                long_rows.append(
                    {
                        "objectid": mid,
                        "prompt_sha256": prompt_sha,
                        "model": model,
                        "provider": None,
                        "created": None,
                        "request_started_utc": started_iso,
                        "request_ended_utc": ended_iso,
                        "latency_seconds": round(t1 - t0, 6),
                        "finish_reason": None,
                        "content": None,
                        "usage_prompt_tokens": None,
                        "usage_completion_tokens": None,
                        "usage_total_tokens": None,
                        "usage_cost": None,
                        "error": err,
                    }
                )
                continue

        rows_wide.append(row)

        try:
            _ = save_image_to_folder(
                str(media.object_thumb), images_dir, basename=mid, session=session
            )
        except Exception as e:
            logging.error(
                "image fetch failed objectid=%s error=%s",
                mid,
                f"{type(e).__name__}: {e}",
            )
            raise

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = run_dir / f"alt_text_runs_{stamp}"
    df_wide = pd.DataFrame(rows_wide)
    df_long = pd.DataFrame(long_rows)

    atomic_write_df_csv(df_wide, Path(f"{base}_wide.csv"))
    atomic_write_df_csv(df_long, Path(f"{base}_long.csv"))
    df_wide.to_parquet(f"{base}_wide.parquet", index=False)  # fail loud
    df_long.to_parquet(f"{base}_long.parquet", index=False)  # fail loud

    jsonl_long = Path(f"{base}_long.jsonl")
    with (Path(str(jsonl_long) + ".tmp")).open("w", encoding="utf-8") as f:
        for r in df_long.to_dict(orient="records"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(str(jsonl_long) + ".tmp").replace(jsonl_long)

    run_t1 = perf_counter()
    manifest = {
        "created_utc": utc_now_iso(),
        "runs_dir": str(run_dir.resolve()),
        "metadata_url": METADATA_URL,
        "saved_metadata": str((run_dir / "metadata.json").resolve()),
        "models": MODELS,
        "mode": MODE,
        "media_ids": MEDIA_IDS,
        "tables": {
            "wide_csv": str(Path(f"{base}_wide.csv").resolve()),
            "wide_parquet": str(Path(f"{base}_wide.parquet").resolve()),
            "long_csv": str(Path(f"{base}_long.csv").resolve()),
            "long_parquet": str(Path(f"{base}_long.parquet").resolve()),
            "long_jsonl": str(Path(f"{base}_long.jsonl").resolve()),
        },
        "python_version": sys.version,
        "images_dir": str((run_dir / "images").resolve()),
        "run_wall_time_seconds": round(run_t1 - run_t0, 6),
        "logfile": str((run_dir / "run.log").resolve()),
        "openrouter_url": OPENROUTER_URL,
    }
    persist_json(run_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()


# TODO:

# - Add heuristic checks for obviously bad alt-texts (e.g., too short/too long, "image of", etc.)
# - Add heuristic for complex images (e.g., presence of "chart", "diagram", "map" in metadata)
