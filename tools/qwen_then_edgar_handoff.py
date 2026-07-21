"""Finish the frozen Qwen repair, then safely resume EDGAR embeddings.

This is an operational handoff controller. It never recomputes contract-valid
Qwen rows, never mutates the frozen EDGAR corpus, and never launches a duplicate
GPU worker.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from event_surprise_inference_5000 import ARTICLES, EVENT_TYPES, OUT, UNIT_FIELDS
from event_surprise_qwen_resume_5000 import FINAL, FINAL_SCHEMA

QWEN_REPO = Path(__file__).resolve().parents[1]
QWEN_MODEL = "qwen3.6:35b-a3b"
QWEN_RESUME = QWEN_REPO / "tools" / "event_surprise_qwen_resume_5000.py"
QWEN_STDOUT = OUT / "qwen_contract_repair_stdout.log"
QWEN_STDERR = OUT / "qwen_contract_repair_stderr.log"
HANDOFF_STATE = OUT / "qwen_edgar_handoff_state.json"
QUARANTINE = OUT / "quarantine"

EDGAR_REPO = Path(
    os.environ.get("PURE_NEWS_EDGAR_REPO", "pure-news-research-edgar-scale-v2")
).resolve()
EDGAR_RUNTIME = Path(
    os.environ.get("PURE_NEWS_EDGAR_RUNTIME", "data/edgar-scale-v2-runtime")
).resolve()
EDGAR_COMMAND_RECORD = EDGAR_RUNTIME / "logs" / "production_embeddings_command_20260714_155203.json"
EDGAR_PROGRESS = (
    EDGAR_RUNTIME
    / "embeddings"
    / "EDGAR_SCALE_V2_FULL_20260714_E5_MISTRAL_7B_FLOAT16"
    / "embedding_progress.json"
)
EDGAR_PID = EDGAR_RUNTIME / "logs" / "production_embeddings.pid"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_state(phase: str, **details: object) -> None:
    payload = {"updated_at_utc": utc_now(), "phase": phase, **details}
    HANDOFF_STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = HANDOFF_STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, HANDOFF_STATE)


def process_records() -> list[dict[str, object]]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'python' } | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if not result.stdout.strip():
        return []
    parsed = json.loads(result.stdout)
    return parsed if isinstance(parsed, list) else [parsed]


def matching_processes(*needles: str) -> list[dict[str, object]]:
    matches = []
    for record in process_records():
        command_line = str(record.get("CommandLine") or "").casefold()
        if all(needle.casefold() in command_line for needle in needles):
            matches.append(record)
    # A Windows venv launcher can appear as a parent python.exe plus its
    # python3.x child for one logical worker. Count only the outer process.
    matched_ids = {int(record["ProcessId"]) for record in matches}
    roots = [
        record for record in matches if int(record.get("ParentProcessId") or -1) not in matched_ids
    ]
    return roots or matches


def backup_checkpoint_once() -> dict[str, object] | None:
    checkpoint = OUT / "event_surprise_predictions.checkpoint.jsonl"
    if not checkpoint.exists():
        return None
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        QUARANTINE.glob("event_surprise_predictions.checkpoint.pre_contract_repair_*.jsonl")
    )
    if existing:
        path = existing[-1]
        return {"path": str(path), "sha256": sha256(path), "created": False}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = QUARANTINE / f"event_surprise_predictions.checkpoint.pre_contract_repair_{stamp}.jsonl"
    shutil.copy2(checkpoint, path)
    return {"path": str(path), "sha256": sha256(path), "created": True}


def wait_for_existing_qwen() -> None:
    while True:
        workers = matching_processes("event_surprise_qwen_resume_5000.py")
        if len(workers) > 1:
            raise RuntimeError(f"Refusing duplicate Qwen workers: {workers}")
        if not workers:
            return
        write_state("waiting_for_existing_qwen_worker", worker=workers[0])
        time.sleep(30)


def run_qwen_repair(backup: dict[str, object] | None) -> None:
    write_state("qwen_repair_starting", backup=backup)
    with (
        QWEN_STDOUT.open("a", encoding="utf-8") as stdout,
        QWEN_STDERR.open("a", encoding="utf-8") as stderr,
    ):
        stdout.write(f"\n=== repair start {utc_now()} ===\n")
        stdout.flush()
        result = subprocess.run(
            [sys.executable, "-u", str(QWEN_RESUME)],
            cwd=QWEN_REPO,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if result.returncode != 0:
        write_state("qwen_repair_failed", returncode=result.returncode, backup=backup)
        raise RuntimeError(f"Qwen repair exited with code {result.returncode}")


def validate_qwen_final() -> dict[str, object]:
    if not FINAL.exists():
        raise RuntimeError("Qwen final Parquet is absent")
    frame = pl.read_parquet(FINAL)
    expected_columns = set(FINAL_SCHEMA)
    if set(frame.columns) != expected_columns:
        raise RuntimeError("Qwen final Parquet columns do not match the frozen schema")
    if frame.height != 5000:
        raise RuntimeError(f"Qwen final Parquet has {frame.height} rows, expected 5000")
    if frame["article_id"].n_unique() != 5000 or frame["article_hash"].n_unique() != 5000:
        raise RuntimeError("Qwen final Parquet contains duplicate article identities")
    if not bool(frame["valid_json"].all()):
        raise RuntimeError("Qwen final Parquet contains invalid JSON rows")
    if frame.filter(~pl.col("event_type").is_in(EVENT_TYPES)).height:
        raise RuntimeError("Qwen final Parquet contains an event type outside the frozen taxonomy")
    for field in UNIT_FIELDS:
        if frame.filter(pl.col(field).is_null() | ~pl.col(field).is_between(0, 1)).height:
            raise RuntimeError(f"Qwen final Parquet contains invalid {field} values")
    for field in ("surprise_direction", "direction_score"):
        if frame.filter(pl.col(field).is_null() | ~pl.col(field).is_between(-1, 1)).height:
            raise RuntimeError(f"Qwen final Parquet contains invalid {field} values")

    source = pl.read_parquet(ARTICLES).select(["article_id", "title", "content"])
    expected_hashes = {
        str(row["article_id"]): hashlib.sha256(
            f"{row['article_id']}\n{row['title']}\n{row['content']}".encode()
        ).hexdigest()
        for row in source.to_dicts()
    }
    observed_hashes = dict(
        zip(frame["article_id"].to_list(), frame["article_hash"].to_list(), strict=True)
    )
    if observed_hashes != expected_hashes:
        raise RuntimeError("Qwen final Parquet does not match the frozen source hashes")
    return {
        "rows": frame.height,
        "unique_article_ids": frame["article_id"].n_unique(),
        "unique_article_hashes": frame["article_hash"].n_unique(),
        "sha256": sha256(FINAL),
    }


def unload_qwen() -> None:
    subprocess.run(["ollama", "stop", QWEN_MODEL], check=False, capture_output=True, text=True)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        result = subprocess.run(["ollama", "ps"], check=False, capture_output=True, text=True)
        if QWEN_MODEL.casefold() not in result.stdout.casefold():
            return
        time.sleep(5)
    raise RuntimeError("Qwen remained resident in Ollama after the unload deadline")


def launch_edgar(qwen_validation: dict[str, object]) -> dict[str, object]:
    active = matching_processes("edgar_scale_v2_full_pipeline.py", "embed")
    if len(active) > 1:
        raise RuntimeError(f"Refusing duplicate EDGAR embedding workers: {active}")
    if active:
        return {"already_running": True, "worker": active[0]}

    progress = json.loads(EDGAR_PROGRESS.read_text(encoding="utf-8-sig"))
    if int(progress["completed_rows"]) >= int(progress["total_rows"]):
        return {"already_complete": True, "progress": progress}

    disk = shutil.disk_usage(EDGAR_RUNTIME)
    free_gib = disk.free / (1024**3)
    if free_gib < 15:
        raise RuntimeError(f"Refusing EDGAR launch with only {free_gib:.2f} GiB free")

    command_record = json.loads(EDGAR_COMMAND_RECORD.read_text(encoding="utf-8-sig"))
    python_executable = shutil.which(str(command_record["executable"]))
    if not python_executable:
        raise RuntimeError("The original EDGAR Python interpreter is unavailable")
    arguments = [str(value) for value in command_record["arguments"]]
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in command_record["env"].items()})

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = EDGAR_RUNTIME / "logs" / f"production_embeddings_handoff_stdout_{stamp}.log"
    stderr_path = EDGAR_RUNTIME / "logs" / f"production_embeddings_handoff_stderr_{stamp}.log"
    stdout = stdout_path.open("ab", buffering=0)
    stderr = stderr_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            [python_executable, *arguments],
            cwd=EDGAR_REPO,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    finally:
        stdout.close()
        stderr.close()
    time.sleep(8)
    if process.poll() is not None:
        raise RuntimeError(f"EDGAR worker exited immediately with code {process.returncode}")

    temporary_pid = EDGAR_PID.with_suffix(".tmp")
    temporary_pid.write_text(str(process.pid), encoding="ascii")
    os.replace(temporary_pid, EDGAR_PID)
    launch_record = {
        "started_at_utc": utc_now(),
        "pid": process.pid,
        "python": python_executable,
        "arguments": arguments,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "resumed_from_completed_rows": int(progress["completed_rows"]),
        "total_rows": int(progress["total_rows"]),
        "free_gib_before_launch": free_gib,
        "qwen_validation": qwen_validation,
    }
    launch_path = EDGAR_RUNTIME / "logs" / f"production_embeddings_handoff_command_{stamp}.json"
    launch_path.write_text(json.dumps(launch_record, indent=2), encoding="utf-8")
    return launch_record


def main() -> None:
    try:
        backup = backup_checkpoint_once()
        wait_for_existing_qwen()
        validation: dict[str, object]
        try:
            validation = validate_qwen_final()
        except RuntimeError:
            run_qwen_repair(backup)
            validation = validate_qwen_final()
        write_state("qwen_validated", qwen_validation=validation, backup=backup)
        unload_qwen()
        write_state("qwen_unloaded", qwen_validation=validation, backup=backup)
        edgar = launch_edgar(validation)
        write_state(
            "edgar_running" if not edgar.get("already_complete") else "edgar_complete",
            qwen_validation=validation,
            edgar=edgar,
            backup=backup,
        )
    except Exception as error:
        write_state("failed", error=type(error).__name__, message=str(error))
        raise


if __name__ == "__main__":
    main()
