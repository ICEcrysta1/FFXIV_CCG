"""ONNX 导出公开 API 与原子目录生命周期。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ..config.config import OnnxExportConfig
from .builder import build_package as _build_package
from .publish import rename_with_retry


def export_from_config(config: OnnxExportConfig) -> Path:
    """执行已完成 `.env` / CLI 合并和校验的导出配置。"""
    return export_package(
        checkpoint_path=config.checkpoint_path,
        output_dir=config.output_dir,
        deployment_profile_path=config.deployment_profile_path,
        opset=config.opset,
        precision=config.precision,
        ort_provider=config.ort_provider,
        validation_devices=config.validation_devices,
        overwrite=config.overwrite,
    )


def export_package(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    opset: int,
    precision: str,
    deployment_profile_path: Path | None = None,
    ort_provider: str,
    validation_devices: tuple[str, ...],
    overwrite: bool,
) -> Path:
    """完整生成并验证部署包；验证失败时保持既有目录不变。"""
    checkpoint_path = checkpoint_path.resolve()
    output_dir = output_dir.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.export-", dir=output_dir.parent)
    )
    try:
        _build_package(
            checkpoint_path=checkpoint_path,
            package_dir=temp_dir,
            deployment_profile_path=deployment_profile_path,
            opset=opset,
            precision=precision,
            ort_provider=ort_provider,
            validation_devices=validation_devices,
        )
        _publish_directory(temp_dir, output_dir, overwrite=overwrite)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return output_dir


def _publish_directory(temp_dir: Path, output_dir: Path, *, overwrite: bool) -> None:
    from .publish import publish_directory

    publish_directory(
        temp_dir,
        output_dir,
        overwrite=overwrite,
        rename_with_retry_fn=rename_with_retry,
    )
