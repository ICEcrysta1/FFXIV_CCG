"""调用独立 Node 解析进程，验证结果并原子保存完整标注 JSON。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from common.dataset_layout import map_dataset_output_path
from scripts.common.json_io import atomic_write_json

from .config import BridgeConfig, severity_weights

RUNTIME = Path(__file__).parent / "runtime" / "run.cjs"


def _selected_job(raw: dict, source_id: int) -> str:
    actors = [actor for actor in raw.get("friendlies", []) if actor.get("id") == source_id]
    if len(actors) != 1 or not isinstance(actors[0].get("type"), str):
        raise ValueError(f"expected one friendly player with source ID {source_id}")
    job_tag = re.sub(r"(?<!^)(?=[A-Z])", "_", actors[0]["type"]).lower()
    if re.fullmatch(r"[a-z]+(?:_[a-z]+)*", job_tag) is None:
        raise ValueError("invalid player job type")
    return job_tag


def output_path_for_source(source: Path, output_root: Path | None, source_root: Path | None) -> Path:
    """仅决定评估阶段根目录，目录层级交由公共映射方法。"""
    if output_root is None:
        stage = next((parent for parent in source.resolve().parents if parent.name == "raw"), None)
        if stage is None:
            raise ValueError("input outside raw requires --output-root")
        output_root = stage.parent / "annotated"
    output = map_dataset_output_path(source, output_root=output_root, source_root=source_root)
    if output == source.resolve():
        raise ValueError("analysis output must not replace its raw input")
    return output


def annotate_file(
    source: Path, *, config: BridgeConfig, output_root: Path | None = None,
    source_root: Path | None = None,
) -> Path:
    """成功后新增 analysis，原始字段与 events 保持不变；失败不覆盖旧结果。"""
    source = source.resolve()
    output = output_path_for_source(source, output_root, source_root)
    content = source.read_bytes()
    raw = json.loads(content)
    if not isinstance(raw, dict) or not raw.get("events") or raw.get("events_complete") is not True:
        raise ValueError("analysis requires a complete raw report with events")
    if "analysis" in raw:
        raise ValueError("input already contains analysis; select the original raw JSON")
    selected = raw.get("source_id")
    if isinstance(selected, bool) or not isinstance(selected, int) or selected <= 0:
        raise ValueError("source_id must be a positive integer")
    job_tag = _selected_job(raw, selected)
    weights = severity_weights(job_tag)
    checksum = hashlib.sha256(content).hexdigest()
    with tempfile.TemporaryDirectory(prefix="action-quality-") as temporary:
        request_path = Path(temporary) / "request.json"
        result_path = Path(temporary) / "analysis.json"
        atomic_write_json(request_path, {
            "source": str(source), "analyzer_root": str(config.analyzer_root),
            "node_modules": str(config.node_modules), "source_id": selected,
        })
        process_env = dict(os.environ)
        for key in ("FFLOGS_V2_CLIENT_ID", "FFLOGS_V2_CLIENT_SECRET"):
            process_env.pop(key, None)
        result = subprocess.run(
            [config.node, str(RUNTIME), str(request_path), str(result_path)],
            cwd=RUNTIME.parent, env=process_env, capture_output=True,
            text=True, encoding="utf-8", timeout=config.timeout, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"analysis process failed:\n{result.stderr.strip()}")
        analysis = json.loads(result_path.read_text(encoding="utf-8"))
    if analysis.get("schema_version") != 1 or analysis.get("source", {}).get("sha256") != checksum:
        raise ValueError("analysis source or schema mismatch")
    if analysis.get("actor", {}).get("id") != str(selected) or analysis.get("module_errors"):
        raise ValueError("analysis actor mismatch or failed modules")
    if hashlib.sha256(source.read_bytes()).hexdigest() != checksum:
        raise ValueError("raw input changed during analysis")
    for suggestion in analysis["fight_labels"]:
        suggestion["severity_weight"] = weights.get(suggestion["severity"])
    analysis["severity_weights"] = weights
    analysis["job_tag"] = job_tag
    atomic_write_json(output, {**raw, "analysis": analysis})
    return output
