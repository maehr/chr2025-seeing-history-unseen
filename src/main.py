#!/usr/bin/env python3
# ruff: noqa
from __future__ import annotations

import hashlib, json, logging, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import pandas as pd
import requests
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    ValidationError,
    field_validator,
    SecretStr,
)
from pydantic_settings import BaseSettings

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
MODE: str = "subsample"  # "testing" | "subsample" | "full"

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

# Columns to be pivoted from long to wide format
PIVOT_COLS = [
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


logging.basicConfig(
    level=logging.INFO, format="%(asctime)sZ\t%(levelname)s\t%(message)s"
)


class Settings(BaseSettings):
    """Loads environment variables, e.g., from .env file"""

    OPENROUTER_API_KEY: SecretStr


class MediaObject(BaseModel):
    objectid: str
    parentid: str | None = None
    title: str | None = "Kein Titel"
    description: str | None = "Keine Beschreibung"
    relation: list[Any] | None = None
    coverage: str | None = None
    isPartOf: list[Any] | None = None
    creator: list[str] | str | None = "Kein Ersteller"
    publisher: list[str] | str | None = "Kein Herausgeber"
    source: list[str] | str | None = "Keine Quelle"
    date: str | None = "Kein Datum"
    type: str | None = None
    format: str | None = None
    extent: str | None = None
    language: str | None = None
    rights: str | None = None
    license: str | None = None
    object_location: HttpUrl | None = None
    image_alt_text: str | None = ""
    object_thumb: HttpUrl | None = None
    reference_url: HttpUrl | None = None
    era: str | None = Field(default=None, alias="Epoche")

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
    refusal: str | None = None
    reasoning: str | None = None


class ORChoice(BaseModel):
    index: int
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    message: ORMessageContent


class ORPromptTokensDetails(BaseModel):
    cached_tokens: int | None = 0
    audio_tokens: int | None = 0


class ORCompletionTokensDetails(BaseModel):
    reasoning_tokens: int | None = 0
    image_tokens: int | None = 0


class ORCostDetails(BaseModel):
    upstream_inference_cost: float | None = None
    upstream_inference_prompt_cost: float | None = None
    upstream_inference_completions_cost: float | None = None


class ORUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float | None = None
    is_byok: bool | None = None
    prompt_tokens_details: ORPromptTokensDetails | None = None
    completion_tokens_details: ORCompletionTokensDetails | None = None
    cost_details: ORCostDetails | None = None


class ORCompletion(BaseModel):
    id: str
    provider: str | None = None
    model: str
    object: str | None = None
    created: int | None = None
    choices: list[ORChoice]
    usage: ORUsage | None = None


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
    tmp = Path(f"{path}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    tmp = Path(f"{path}.tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def atomic_write_df_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = Path(f"{path}.tmp")
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
    return f"""Titel: {media.title or "Kein Titel"}
Beschreibung: {media.description or "Keine Beschreibung"}
Ersteller: {media.creator or "Kein Ersteller"}
Herausgeber: {media.publisher or "Kein Herausgeber"}
Quelle: {media.source or "Keine Quelle"}
Datum: {media.date or "Kein Datum"}
Epoche: {media.era or "Keine Epoche"}""".strip()


def build_messages(
    prompt: str, image_url: str
) -> tuple[list[dict[str, Any]], str, str]:
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
    """Loads API key from .env file or environment variables."""
    try:
        settings = Settings()
        return settings.OPENROUTER_API_KEY.get_secret_value()
    except ValidationError as e:
        raise ValueError(
            "OPENROUTER_API_KEY not found. Set it in your environment or .env.\n"
            f"Details: {e}"
        ) from e


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
    session: requests.Session | None = None,
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


def create_wide_df_from_long(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivots the long DataFrame to the wide format.
    """
    if df_long.empty:
        return pd.DataFrame()

    df_wide = df_long.pivot(
        index=["objectid", "title", "prompt_sha256"],
        columns="model",
        values=PIVOT_COLS,
    )

    # Flatten the multi-index columns (e.g., ('content', 'openai/gpt-4o-mini'))
    df_wide.columns = [
        f"{model.replace('/', '__')}__{col_name}"
        for col_name, model in df_wide.columns.values
    ]

    # Add a 'created_utc' column for the run
    df_wide["created_utc"] = utc_now_iso()
    return df_wide.reset_index()


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

    long_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, str]] = []

    for mid in MEDIA_IDS:
        media = db[mid]
        # --- ADDED: Get the title here ---
        title = media.title or "Kein Titel"  # Use default from model if None

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
                "title": title,
                "prompt_sha256": prompt_sha,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "image_url": str(media.object_thumb),
            }
        )

        for model in MODELS:
            t0 = perf_counter()
            started_iso = utc_now_iso()
            try:
                orc = call_openrouter(
                    api_key=api_key, model=model, messages=messages, session=session
                )
                ended_iso = utc_now_iso()
                t1 = perf_counter()
                persist_json(
                    raw_dir / f"{mid}__{model.replace('/', '__')}.json",
                    orc.model_dump(mode="json"),
                )
                content = orc.choices[0].message.content if orc.choices else ""
                finish_reason = orc.choices[0].finish_reason if orc.choices else None

                long_rows.append(
                    {
                        "objectid": mid,
                        "title": title,
                        "prompt_sha256": prompt_sha,
                        "model": model,
                        "provider": orc.provider,
                        "created": orc.created,
                        "id": orc.id,
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
                long_rows.append(
                    {
                        "objectid": mid,
                        "title": title,
                        "prompt_sha256": prompt_sha,
                        "model": model,
                        "provider": None,
                        "created": None,
                        "id": None,
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

    if prompt_rows:
        df_prompts = pd.DataFrame(prompt_rows)
        prompts_csv_path = Path(f"{base}_prompts.csv")
        atomic_write_df_csv(df_prompts, prompts_csv_path)
    else:
        logging.warning("No prompts were generated, skipping prompts.csv file.")
        prompts_csv_path = Path()

    # Create both DataFrames from the single 'long_rows' list
    df_long = pd.DataFrame(long_rows)
    df_wide = create_wide_df_from_long(df_long)

    atomic_write_df_csv(df_wide, Path(f"{base}_wide.csv"))
    atomic_write_df_csv(df_long, Path(f"{base}_long.csv"))
    df_wide.to_parquet(f"{base}_wide.parquet", index=False)
    df_long.to_parquet(f"{base}_long.parquet", index=False)

    jsonl_long = Path(f"{base}_long.jsonl")
    with (Path(f"{jsonl_long}.tmp")).open("w", encoding="utf-8") as f:
        for r in df_long.to_dict(orient="records"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(f"{jsonl_long}.tmp").replace(jsonl_long)

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
            "prompts_csv": str(prompts_csv_path.resolve())
            if prompts_csv_path.exists()
            else None,
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
