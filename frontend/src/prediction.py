from __future__ import annotations

import json
import re
import uuid
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


RUN_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_ALLOWED_HOSTS = {"127.0.0.1", "localhost"}


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    label: str


def list_runs(*, artifacts_dir: Path) -> list[RunSummary]:
    base = artifacts_dir.resolve()
    if not base.exists():
        return []

    runs: list[RunSummary] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        if not RUN_ID_RE.fullmatch(p.name):
            continue
        info_path = p / "run.json"
        if not info_path.exists():
            continue
        try:
            data = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        symbol = str(data.get("symbol", "")).strip()
        created_at = str(data.get("created_at", "")).strip()
        model_kind = str(data.get("model_kind", "")).strip()
        label = f"{p.name} · {symbol or '—'} · {model_kind or '—'} · {created_at or '—'}"
        runs.append(RunSummary(run_id=p.name, label=label))

    return sorted(runs, key=lambda r: r.run_id, reverse=True)


def post_predict(
    *,
    base_url: str,
    run_id: str,
    symbol: str,
    dataset_id: str | None,
    file_name: str | None,
    file_bytes: bytes | None,
    threshold: float = 0.5,
    include_threshold_metrics: bool = True,
    eval_window: int = 252,
    include_failure_analysis: bool = False,
) -> dict:
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or (parsed.hostname or "") not in _ALLOWED_HOSTS:
        raise ValueError("Backend URL must be http://localhost or http://127.0.0.1")

    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("Invalid run_id.")
    if not symbol or len(symbol) > 32:
        raise ValueError("Invalid symbol.")
    if not (0.0 < float(threshold) < 1.0):
        raise ValueError("threshold must be between 0 and 1.")

    boundary = uuid.uuid4().hex
    parts: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )

    add_field("run_id", run_id)
    add_field("symbol", symbol)
    add_field("include_features", "true")
    add_field("threshold", str(float(threshold)))
    add_field("include_threshold_metrics", "true" if include_threshold_metrics else "false")
    add_field("eval_window", str(int(eval_window)))
    add_field("include_failure_analysis", "true" if include_failure_analysis else "false")

    if bool(dataset_id) == bool(file_bytes):
        raise ValueError("Provide exactly one of dataset_id or file.")

    if dataset_id:
        add_field("dataset_id", dataset_id)
    else:
        assert file_bytes is not None
        if not file_name:
            file_name = "dataset.csv"
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
                f"Content-Type: text/csv\r\n\r\n"
            ).encode("utf-8")
            + file_bytes
            + b"\r\n"
        )

    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)

    url = base_url.rstrip("/") + "/ml/predict"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

