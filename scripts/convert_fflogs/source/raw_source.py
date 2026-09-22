"""raw FFLogs JSON 到训练 payload 的唯一入口。"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..config.constants import DEFAULT_DOWNTIME_GAP_SECONDS
from ..pipeline import convert_report_to_training_payload
from ..utils import build_backend, build_skill_book, load_job_project_config


def convert_raw_file(
    input_path: Path,
    *,
    job_tag: str,
    source: int | None = None,
    encounter: str | None = None,
    downtime_gap: float = DEFAULT_DOWNTIME_GAP_SECONDS,
) -> tuple[dict[str, object] | None, Counter[tuple[int, str]]]:
    """读取一份 raw JSON，并直接返回内存训练 payload。"""
    input_path = Path(input_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"raw FFLogs payload must be a mapping: {input_path}")

    source_id = int(source or payload.get("source_id") or 10)
    fights_meta = payload.get("fights", [])
    encounter_name = encounter
    if encounter_name is None and isinstance(fights_meta, list) and fights_meta:
        first_fight = fights_meta[0]
        if isinstance(first_fight, dict):
            encounter_name = first_fight.get("name")

    report_code = str(payload.get("report_code", input_path.stem))
    player_name = str(payload.get("player_name", "?"))
    project_config = load_job_project_config(job_tag)
    backend = build_backend(job_tag=job_tag)
    try:
        return convert_report_to_training_payload(
            payload,
            backend=backend,
            project_config=project_config,
            skill_book=build_skill_book(project_config),
            source_id=source_id,
            encounter_name=encounter_name,
            report_code=report_code,
            player_name=player_name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            downtime_gap_seconds=float(downtime_gap),
        )
    finally:
        backend.close()
