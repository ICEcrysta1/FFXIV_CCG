"""GraphQL 标量与报告编号的输入校验。"""

from .constants import INTEGER_PATTERN, RANKING_METRICS, REPORT_CODE_PATTERN


def _validate_alphanumeric(value: str, field_name: str) -> str:
    """校验只能安全嵌入 GraphQL 字符串字面量的字母数字参数。"""
    if not isinstance(value, str) or not REPORT_CODE_PATTERN.fullmatch(value):
        raise ValueError(f"invalid {field_name}: {value!r}")
    return value


def _validate_integer(value: int | str, field_name: str, *, minimum: int | None = None) -> int:
    """把公开 API 的整数参数规范化为安全的 GraphQL Int 字面量。"""
    if isinstance(value, bool):
        raise ValueError(f"invalid {field_name}: {value!r}")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str) and INTEGER_PATTERN.fullmatch(value):
        normalized = int(value)
    else:
        raise ValueError(f"invalid {field_name}: {value!r}")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"invalid {field_name}: {value!r}")
    return normalized


def _validate_integer_list(
    values: list[int],
    field_name: str,
    *,
    minimum: int | None = None,
) -> list[int]:
    """逐项校验整数列表，避免将列表原样拼入 GraphQL 查询。"""
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"invalid {field_name}: {values!r}")
    return [
        _validate_integer(value, f"{field_name}[{index}]", minimum=minimum)
        for index, value in enumerate(values)
    ]


def _validate_ranking_metric(metric: str) -> str:
    """校验排行指标，避免把公开 API 参数拼入任意 GraphQL 片段。"""
    if not isinstance(metric, str) or metric not in RANKING_METRICS:
        raise ValueError(f"invalid metric: {metric!r}")
    return metric


def _validate_report_code(report_code: str) -> str:
    """校验报告编号，确保它只能作为文件名的一部分使用。"""
    return _validate_alphanumeric(report_code, "report code")
