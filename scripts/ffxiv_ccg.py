"""项目操作菜单：配置与业务规则由所属模块提供，入口只负责分发和生命周期。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MENU_ENTRIES = (
    ("fflogs", "FFLogs 数据下载"),
    ("annotate", "XIVanalysis 数据标记"),
    ("convert", "训练文件转换"),
    ("train", "训练（BC 预训练）"),
    ("resume", "恢复训练（选择 BC checkpoint）"),
    ("grpo", "GRPO 后训练"),
    ("export", "ONNX 导出（导出、PT/ORT parity 门禁、发布校验）"),
    ("analysis", "模型分析图生成（不含损失地形图）"),
    ("replay", "模型自回归回放"),
)
CHECKPOINT_ACTIONS = {"resume", "grpo", "export", "analysis", "replay"}


def resolve_action(value: str, *, allow_number: bool = False) -> str:
    value = value.strip().lower()
    if value == "menu":
        return value
    if allow_number and value == "10":
        return "exit"
    for number, (action, _label) in enumerate(MENU_ENTRIES, 1):
        if value == action or (allow_number and value == str(number)):
            return action
    raise ValueError(f"未知操作：{value}，可用值：{', '.join(action for action, _ in MENU_ENTRIES)}，或 menu")


def show_menu() -> str:
    print("\nFFXIV_CCG 常用工具菜单\n所有命令使用项目 .venv，参数读取 config/ 与根目录 .env。\n")
    for number, (_action, label) in enumerate(MENU_ENTRIES, 1):
        print(f"  {number}. {label}")
    print("  10. 退出\n")
    return resolve_action(input("请输入选项编号："), allow_number=True)


def checkpoint_target() -> dict:
    # 延迟导入，使帮助和菜单展示不依赖 GPU 或 checkpoint 加载。
    from common.policy.config import (
        resolve_policy_model_config_path,
        resolve_policy_model_job_tag,
        resolve_policy_model_variant,
    )
    from training.config import load_run_config
    from training.loop.checkpoint import (
        UNKNOWN_MAX_FILES,
        collect_checkpoint_candidates,
    )

    config_path = resolve_policy_model_config_path()
    config = load_run_config(config_path)

    def serialize(items, source):
        return [{"path": str(item.path), "name": item.path.name, "epoch": item.epoch,
                 "source": source, "output_dir": str(item.path.parent),
                 "max_files": "unknown" if item.max_files is UNKNOWN_MAX_FILES else item.max_files,
                 "validation_files": item.validation_files} for item in items]

    resumable, rejected, skipped = collect_checkpoint_candidates(config.output_dir, max_epochs=config.max_epochs)
    grpo = []
    directories = sorted(path for path in config.output_dir.parent.glob(f"{config.output_dir.name}_grpo*") if path.is_dir())
    skipped = list(skipped)
    for directory in directories:
        allowed, completed, invalid = collect_checkpoint_candidates(directory, max_epochs=config.max_epochs)
        grpo.extend(serialize((*allowed, *completed), "grpo"))
        skipped.extend(invalid)
    return {"job_tag": resolve_policy_model_job_tag(config_path),
            "model_variant": resolve_policy_model_variant(config_path), "config_path": str(config_path),
            "output_dir": str(config.output_dir), "grpo_output_dir": [str(path) for path in directories],
            "max_files": config.max_files, "validation_files": config.validation_files, "max_epochs": config.max_epochs,
            "resumable": serialize(resumable, "bc"), "rejected": serialize(rejected, "bc"), "grpo": grpo,
            "skipped": [{"name": item.path.name, "error": item.error} for item in skipped]}


def show_checkpoint_target(target: dict) -> None:
    print(f"\n当前配置：FFXIV_JOB_TAG={target['job_tag']}，FFXIV_MODEL_VARIANT={target['model_variant']}")
    print(f"模型配置：{target['config_path']}\nBC checkpoint 目录：{target['output_dir']}")
    for directory in target["grpo_output_dir"]:
        print(f"GRPO checkpoint 目录：{directory}")
    print(f"训练数据上限：{target['max_files']}；验证数据份数：{target['validation_files']}（来自模型 YAML）")
    for item in target["skipped"]:
        print(f"跳过无法读取的 checkpoint：{item['name']}：{item['error']}", file=sys.stderr)


def select_checkpoint(candidates: list[dict]) -> dict | None:
    if not candidates:
        raise ValueError("checkpoint 目录中没有可选择的 .pt 文件")
    entries = {}
    print("\n可用 checkpoint：")
    for source, prefix, progress in (("bc", "", "epoch"), ("grpo", "a", "iteration")):
        ordered = sorted((item for item in candidates if item["source"] == source),
                         key=lambda item: (item["name"] == "best.pt", item["name"]))
        for index, item in enumerate(ordered, 1):
            key = f"{prefix}{index}"
            entries[key] = item
            print(f"  {key:>3}. {item['name']}（{source.upper()}，{progress} {item['epoch']}，目录 {Path(item['output_dir']).name}）")
    choice = input("请输入 checkpoint 编号（如 1 或 a1，直接回车取消）：").strip().lower()
    if not choice:
        return None
    if choice not in entries:
        raise ValueError(f"无效的 checkpoint 编号：{choice}（BC 使用 1、2、3；GRPO 使用 a1、a2、a3）")
    return entries[choice]


def interactive_checkpoint(target: dict, *, resumable_only: bool) -> dict | None:
    if resumable_only:
        if not target["resumable"]:
            raise ValueError(f"没有可恢复的 BC checkpoint；需检查文件或提高 training.max_epochs={target['max_epochs']}。")
        for item in target["rejected"]:
            print(f"已跳过不可续训的 checkpoint：{item['name']}（epoch {item['epoch']} 已达 training.max_epochs）")
        candidates = target["resumable"]
    else:
        candidates = target["resumable"] + target["rejected"] + target["grpo"]
    return select_checkpoint(candidates)


def configured_checkpoint(requested: Path | None) -> dict:
    from common.policy.config import (
        resolve_policy_checkpoint_path,
        resolve_policy_model_config_path,
    )
    from common.pytorch.serialization import safe_torch_load

    path = requested or resolve_policy_checkpoint_path(resolve_policy_model_config_path())
    path = Path(path)
    path = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise ValueError(f"checkpoint 不存在：{path}")
    payload = safe_torch_load(path, mmap=True, map_location="meta")
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint 必须为 mapping：{path}")
    return {"path": str(path), "name": path.name, "source": "grpo" if payload.get("grpo_checkpoint") else "bc"}


def confirm_resume(selected: dict, target: dict) -> bool:
    mismatches = [key for key in ("max_files", "validation_files")
                  if key not in selected or selected[key] == "unknown" or selected[key] != target[key]]
    if not mismatches:
        return False
    for key in mismatches:
        print(f"checkpoint {selected['name']} 保存时的 training.{key}={selected.get(key, '未知')}，当前 YAML 为 {target[key]}；数据选择可能变化。")
    if input("仍要强制继续吗？输入 Y 确认，直接回车或输入 N 取消：").strip().lower() != "y":
        raise OperationCancelled("已取消恢复训练。")
    return True


class OperationCancelled(Exception):
    """用户取消菜单操作。"""


def run_module(module: str, *arguments: str) -> int:
    return subprocess.run([sys.executable, "-m", module, *arguments], cwd=PROJECT_ROOT, check=False).returncode


def invoke_tool(action: str, *, checkpoint: Path | None = None, interactive: bool = False,
                force_resume_data_mismatch: bool = False) -> int:
    selected = None
    selection = None
    if action in CHECKPOINT_ACTIONS:
        if interactive:
            selection = checkpoint_target()
            show_checkpoint_target(selection)
            selected = interactive_checkpoint(selection, resumable_only=action == "resume")
            if selected is None:
                print("已取消操作。")
                return 0
        else:
            selected = configured_checkpoint(checkpoint)
    if action == "resume":
        if selected["source"] == "grpo":
            raise ValueError("恢复训练仅支持 BC checkpoint；GRPO checkpoint 可作为新的 GRPO 权重起点。")
        if interactive and not force_resume_data_mismatch:
            force_resume_data_mismatch = confirm_resume(selected, selection)
    download_arguments = []
    if action == "fflogs":
        target = input("请输入副本代号（如 FRU、M1s）或 Zone ID（直接回车取消）：").strip()
        if not target:
            return 0
        value = input("请输入训练集目标份数（验证集按下载器配置计算，直接回车取消）：").strip()
        if not value:
            return 0
        count = int(value)
        if count < 1:
            raise ValueError("训练集目标份数必须是正整数。")
        download_arguments = ["batch", target, "--count", str(count)]
    modules = {
        "fflogs": ("scripts.fflogs_scraper", download_arguments),
        "annotate": ("scripts.action_quality", []),
        "convert": ("scripts.convert_fflogs.cli", ["--training-selection"]),
        "train": ("training.train", []),
        "resume": ("training.train", ["--resume", selected["path"]] if selected else []),
        "grpo": ("grpo", ["--checkpoint", selected["path"]] if selected else []),
        "export": ("scripts.onnx_export.workflow", ["all", "--checkpoint", selected["path"]] if selected else []),
        "analysis": ("scripts.model_analysis", ["--checkpoint", selected["path"]] if selected else []),
        "replay": ("scripts.autoregressive_replay", ["--checkpoint", selected["path"]] if selected else []),
    }
    module, arguments = modules[action]
    if action == "resume" and force_resume_data_mismatch:
        arguments.append("--force-resume-data-mismatch")
    monitor = action in {"train", "resume", "grpo"}
    owner = str(os.getpid())
    try:
        if monitor:
            scope = "grpo" if action == "grpo" else "bc"
            if run_module("scripts.tensorboard_server", "--background", "--open-browser", "--if-enabled", "--scope", scope, "--owner-pid", owner):
                print("TensorBoard Web 监控启动失败；训练仍会继续。", file=sys.stderr)
        status = run_module(module, *arguments)
        if action == "export" and status == 2:
            raise RuntimeError("ONNX 图已导出，但 PT/ORT parity 门禁未通过；请检查 parity JSON 和 release_report.json。")
        if status != 0:
            raise RuntimeError(f"工具执行失败，Python 退出码：{status}")
        return 0
    finally:
        if monitor:
            try:
                if run_module("scripts.tensorboard_server", "--stop-owned", "--owner-pid", owner):
                    print("TensorBoard Web 监控关闭失败，请检查占用端口的进程。", file=sys.stderr)
            except OSError as error:
                print(f"TensorBoard Web 监控关闭失败：{error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FFXIV_CCG 项目操作入口")
    parser.add_argument("--action", default="menu", help="menu 或 " + ", ".join(action for action, _ in MENU_ENTRIES))
    parser.add_argument("--checkpoint", type=Path, help="非交互操作的 checkpoint；省略时读取模型配置默认值")
    parser.add_argument("--force-resume-data-mismatch", action="store_true", help="明确允许续训数据份数变化")
    args = parser.parse_args(argv)
    try:
        action = resolve_action(args.action)
        interactive = action == "menu"
        if interactive:
            action = show_menu()
        if action == "exit":
            return 0
        return invoke_tool(action, checkpoint=args.checkpoint, interactive=interactive,
                           force_resume_data_mismatch=args.force_resume_data_mismatch)
    except OperationCancelled as error:
        print(str(error))
        return 0
    except Exception as error:
        print(f"执行失败：{error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n操作已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
