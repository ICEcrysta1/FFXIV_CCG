"""ONNX Runtime Execution Provider 的统一自动选择。"""

from __future__ import annotations


ORT_PROVIDER_AUTO = "auto"
ORT_PROVIDER_CUDA = "CUDAExecutionProvider"
ORT_DISABLE_CPU_FALLBACK_KEY = "session.disable_cpu_ep_fallback"


def is_cpu_fallback_disabled(requested: str) -> bool:
    """显式非 CPU EP 属于严格验收，禁止把未支持节点分配给 CPU。"""
    value = str(requested).strip()
    return value.lower() != ORT_PROVIDER_AUTO and value != "CPUExecutionProvider"


def resolve_ort_providers(ort, requested: str) -> tuple[str, ...]:
    """本地优先 CUDA，CPU/CI 环境自动回落，显式 provider 则严格要求。"""
    value = str(requested).strip()
    available = set(ort.get_available_providers())
    if value.lower() == ORT_PROVIDER_AUTO:
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider"]
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            return tuple(providers)
        if "CPUExecutionProvider" in available:
            return ("CPUExecutionProvider",)
        raise RuntimeError(
            "ORT auto provider requires CUDAExecutionProvider or CPUExecutionProvider; "
            f"available={sorted(available)}"
        )
    if not value:
        raise ValueError("ORT provider must not be empty")
    if value not in available:
        raise RuntimeError(
            f"ORT provider {value!r} unavailable; available={sorted(available)}"
        )
    # 显式 provider 用于严格验收，不允许 ORT 静默回落到另一个 EP。
    return (value,)


def ort_session_providers(provider_names: tuple[str, ...]) -> list[object]:
    """生成 session provider 配置；FP32 parity 下禁止 CUDA EP 隐式使用 TF32。"""
    return [
        (name, {"use_tf32": "0"})
        if name == "CUDAExecutionProvider"
        else name
        for name in provider_names
    ]


def ort_session_options(ort, requested: str):
    """显式非 CPU EP 使用严格 session，禁止图分配失败时悄悄落到 CPU。"""
    options = ort.SessionOptions()
    if is_cpu_fallback_disabled(requested):
        options.add_session_config_entry(ORT_DISABLE_CPU_FALLBACK_KEY, "1")
    return options


def create_ort_session(ort, model_path, requested: str):
    """创建并核验 ORT session；GPU 运行库漂移时提供可执行的项目级错误。"""
    resolved = resolve_ort_providers(ort, requested)
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=ort_session_options(ort, requested),
            providers=ort_session_providers(resolved),
        )
    except Exception as exc:
        version = getattr(ort, "__version__", "unknown")
        raise RuntimeError(
            "ORT failed to initialize the requested Execution Provider: "
            f"requested={requested!r}, onnxruntime={version}. "
            "For project GPU inference, reinstall the active virtual environment "
            "from requirements.txt and requirements-onnx-gpu.txt; CPU fallback is disabled."
        ) from exc
    active = tuple(session.get_providers())
    if not active:
        raise RuntimeError("ORT session has no active Execution Provider")
    if str(requested).strip().lower() != ORT_PROVIDER_AUTO and active[0] != resolved[0]:
        raise RuntimeError(
            "ORT explicit provider was not activated: "
            f"requested={resolved[0]!r}, active={active!r}"
        )
    return session, resolved, active
