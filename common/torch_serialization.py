"""跨模块复用的安全 PyTorch 序列化读取工具。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PosixPath, PurePath, PureWindowsPath
from typing import Any

from .torch_dependencies import import_torch


# WindowsPath 在 Linux 上不能实例化；显式映射成只解析路径的纯路径类，
# 兼容旧 checkpoint，且不修改 pathlib 的进程级行为。
_SAFE_PATH_GLOBALS = (
    Path, PosixPath,
    (PureWindowsPath, "pathlib.WindowsPath"),
)


def stringify_paths(value: Any) -> Any:
    """保存元数据前递归将路径转成字符串，保留张量与容器的原有语义。"""
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dict):
        return {stringify_paths(key): stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stringify_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(stringify_paths(item) for item in value)
    return value


def safe_torch_load(
    path: str | Path,
    *,
    mmap: bool = False,
    map_location: str | Path = "cpu",
    safe_globals: Iterable[Any] = (),
) -> Any:
    """以 `weights_only=True` 读取仅包含张量和显式允许类型的文件。

    `torch.load` 的默认 pickle 路径可以在反序列化阶段执行任意代码，因此
    checkpoint 和 compiled cache 都必须经过这里读取。路径类型被允许是为了
    兼容旧 checkpoint 中展开后的 `RunConfig`；调用方只应额外传入没有自定义
    反序列化行为的项目数据类型。`map_location` 默认为 CPU；只检查非张量元数据
    （例如 checkpoint 的 `epoch`）时传 `"meta"`，可避免读取全部权重。
    """
    torch = import_torch()
    allowed_globals = (*_SAFE_PATH_GLOBALS, *tuple(safe_globals))
    load_kwargs: dict[str, object] = {
        "map_location": map_location,
        "weights_only": True,
    }
    if mmap:
        load_kwargs["mmap"] = True
    with torch.serialization.safe_globals(list(allowed_globals)):
        return torch.load(path, **load_kwargs)
