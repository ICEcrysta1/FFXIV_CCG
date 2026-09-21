"""跨模块复用的安全 PyTorch 序列化读取工具。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PosixPath, WindowsPath
from typing import Any

from .torch_dependencies import import_torch


_SAFE_PATH_GLOBALS = (Path, PosixPath, WindowsPath)


def safe_torch_load(
    path: str | Path,
    *,
    mmap: bool = False,
    safe_globals: Iterable[Any] = (),
) -> Any:
    """以 `weights_only=True` 读取仅包含张量和显式允许类型的文件。

    `torch.load` 的默认 pickle 路径可以在反序列化阶段执行任意代码，因此
    checkpoint 和 compiled cache 都必须经过这里读取。路径类型被允许是为了
    兼容旧 checkpoint 中展开后的 `RunConfig`；调用方只应额外传入没有自定义
    反序列化行为的项目数据类型。
    """
    torch = import_torch()
    allowed_globals = (*_SAFE_PATH_GLOBALS, *tuple(safe_globals))
    load_kwargs: dict[str, object] = {
        "map_location": "cpu",
        "weights_only": True,
    }
    if mmap:
        load_kwargs["mmap"] = True
    with torch.serialization.safe_globals(list(allowed_globals)):
        return torch.load(path, **load_kwargs)
