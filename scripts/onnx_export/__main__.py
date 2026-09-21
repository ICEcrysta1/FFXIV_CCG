"""``python -m scripts.onnx_export`` 命令入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .export import export_from_config
from .config.config import load_export_config
from .runtime.precision import SUPPORTED_PRECISIONS


def main() -> int:
    parser = argparse.ArgumentParser(description="导出并验证固定容量 ONNX policy")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--deployment-profile",
        type=Path,
        help="一次性固化的部署画像；默认按 checkpoint 职业读取内置 profile",
    )
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument(
        "--precision",
        choices=SUPPORTED_PRECISIONS,
        default=None,
    )
    parser.add_argument(
        "--ort-provider",
        default=None,
        help="临时覆盖根目录 .env 中的共用 ORT Provider",
    )
    parser.add_argument(
        "--validation-devices",
        default=None,
        help="临时覆盖根目录 .env，例如 cpu,cuda",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()
    config = load_export_config(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        deployment_profile=args.deployment_profile,
        opset=args.opset,
        precision=args.precision,
        ort_provider=args.ort_provider,
        validation_devices=args.validation_devices,
        overwrite=args.overwrite,
    )
    output = export_from_config(config)
    print(
        "ONNX graph package validated; rollout parity is still required: "
        f"{output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
