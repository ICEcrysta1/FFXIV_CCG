"""调用独立 Node 解析进程，验证结果并原子保存完整标注 JSON。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from common.dataset_layout import map_dataset_output_path
from scripts.common.json_io import atomic_write_json

from .config import BridgeConfig

RUNTIME = Path(__file__).parent / "runtime" / "run.cjs"


def analyzer_commit(config: BridgeConfig) -> str | None:
    """仅复用可确认且工作树干净的分析器版本。"""
    try:
        result = subprocess.run(
            ["git", "-C", str(config.analyzer_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    if not commit:
        return None
    try:
        status = subprocess.run(
            ["git", "-C", str(config.analyzer_root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return commit if status.returncode == 0 and not status.stdout.strip() else None


def annotation_is_current(source: Path, output: Path, *, commit: str | None) -> bool:
    """仅当原始内容与分析器版本一致时复用已有产物。"""
    if commit is None or not output.is_file():
        return False
    try:
        content = source.read_bytes()
        raw = json.loads(content)
        annotated = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(annotated, dict):
            return False
        analysis = annotated.pop("analysis", None)
        if not isinstance(analysis, dict) or annotated != raw:
            return False
        if any(not isinstance(analysis.get(key), dict) for key in ("source", "actor", "engine", "time_basis")):
            return False
        selected = raw.get("source_id")
        if isinstance(selected, bool) or not isinstance(selected, int) or selected <= 0:
            return False
        job_tag = _selected_job(raw, selected)
        labels = analysis.get("fight_labels")
        if not isinstance(labels, list) or any(
            not isinstance(label, dict)
            or "severity" not in label
            or "severity_weight" in label
            for label in labels
        ):
            return False
        return (
            raw.get("events_complete") is True
            and bool(raw.get("events"))
            and analysis.get("schema_version") == 2
            and analysis.get("bridge_version") == 3
            and analysis.get("status") == "annotated"
            and analysis.get("training_ready") is False
            and analysis.get("source", {}).get("sha256") == hashlib.sha256(content).hexdigest()
            and analysis.get("source", {}).get("fight_id") == raw.get("fight_id")
            and analysis.get("actor", {}).get("id") == str(selected)
            and analysis.get("job_tag") == job_tag
            and analysis.get("engine", {}).get("commit") == commit
            and analysis.get("time_basis", {}).get("unit") == "ms"
            and analysis.get("time_basis", {}).get("origin") == "pull_start"
            and analysis.get("module_errors") == []
            and "severity_weights" not in analysis
        )
    except (OSError, TypeError, ValueError, UnicodeError, KeyError):
        return False


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
    checksum = hashlib.sha256(content).hexdigest()
    request = {
        "source": str(source), "analyzer_root": str(config.analyzer_root),
        "node_modules": str(config.node_modules), "source_id": selected,
    }
    process_env = dict(os.environ)
    for key in ("FFLOGS_V2_CLIENT_ID", "FFLOGS_V2_CLIENT_SECRET"):
        process_env.pop(key, None)
    result = subprocess.run(
        [config.node, str(RUNTIME)],
        input=json.dumps(request, ensure_ascii=False),
        cwd=RUNTIME.parent, env=process_env, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=config.timeout, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"analysis process failed:\n{result.stderr.strip()}")
    analysis = json.loads(result.stdout)
    if not isinstance(analysis, dict):
        raise TypeError("analysis result must be a JSON object")
    for field in ("source", "time_basis", "actor"):
        if not isinstance(analysis.get(field), dict):
            raise TypeError(f"analysis {field} must be a JSON object")
    suggestions = analysis.get("fight_labels")
    if not isinstance(suggestions, list):
        raise TypeError("analysis fight_labels must be a list")
    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            raise TypeError("analysis fight label must be a JSON object")
        if "severity" not in suggestion:
            raise ValueError("analysis fight label lacks severity")
    if (
        analysis.get("schema_version") != 2
        or analysis.get("bridge_version") != 3
        or analysis.get("source", {}).get("sha256") != checksum
    ):
        raise ValueError("analysis source or schema mismatch")
    if "severity_weights" in analysis or any("severity_weight" in label for label in suggestions):
        raise ValueError("analysis contains obsolete model quality weights")
    basis = analysis.get("time_basis", {})
    if basis.get("unit") != "ms" or basis.get("origin") != "pull_start":
        raise ValueError("analysis time basis mismatch")
    if analysis.get("actor", {}).get("id") != str(selected) or analysis.get("module_errors"):
        raise ValueError("analysis actor mismatch or failed modules")
    if hashlib.sha256(source.read_bytes()).hexdigest() != checksum:
        raise ValueError("raw input changed during analysis")
    analysis["job_tag"] = job_tag
    atomic_write_json(output, {**raw, "analysis": analysis})
    return output
