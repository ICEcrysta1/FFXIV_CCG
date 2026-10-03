"""真实 Python.NET 批量桥接：基础类型、数值位模式、容器隔离与并发。"""

from concurrent.futures import ThreadPoolExecutor
import math
import pickle
import pickletools

import pytest

from scripts.common.inprocess_backend import InProcessBackend, _load_dotnet_types
from tests.scripts.conftest import _require_inprocess_backend


@pytest.fixture(autouse=True)
def require_bridge():
    _require_inprocess_backend()


def _encode(value):
    *_, encoder = _load_dotnet_types()
    return encoder.Encode(value)


def _decode(packet):
    with memoryview(packet) as buffer:
        return pickle.loads(buffer)


def test_packet_preserves_primitive_types_and_float_bits():
    _load_dotnet_types()
    from System import Array, Boolean, Byte, Char, Decimal, Double, Int16, Int32, Int64, Object
    from System import SByte, Single, UInt16, UInt32, UInt64
    from System.Collections import Hashtable

    values = [None, Boolean(True), Boolean(False), Byte(255), SByte(-128), Int16(-32768),
              UInt16(65535), Int32(-(2**31)), UInt32(2**32 - 1), Int64(-(2**63)),
              UInt64(2**64 - 1), Single(1.25), Double(-0.0), Double(float("inf")),
              Double(float("-inf")), Double(float("nan")), Decimal(123), Char("冰"), "火\0冰😀"]
    value = Hashtable()
    value.Add("values", Array[Object](values))
    packet = _encode(value)
    actual = _decode(packet)["values"]
    assert actual[:12] == [None, True, False, 255, -128, -32768, 65535, -(2**31),
                          2**32 - 1, -(2**63), 2**64 - 1, 1.25]
    assert type(actual[1]) is bool and type(actual[3]) is int and type(actual[11]) is float
    assert math.copysign(1, actual[12]) == -1
    assert actual[13:15] == [float("inf"), float("-inf")]
    assert math.isnan(actual[15])
    assert actual[16:] == [123.0, "冰", "火\0冰😀"]
    with memoryview(packet) as buffer:
        instructions = {op.name for op, _, _ in pickletools.genops(buffer.tobytes())}
    assert instructions <= {"PROTO", "EMPTY_DICT", "MARK", "BINUNICODE", "LONG_BINPUT",
                            "LONG_BINGET", "EMPTY_LIST", "NONE", "NEWTRUE", "NEWFALSE",
                            "BININT", "LONG1", "BINFLOAT", "APPENDS", "SETITEMS", "STOP"}


def test_packet_does_not_alias_mutable_containers_and_decodes_after_gc():
    _load_dotnet_types()
    from System import GC
    from System.Collections import ArrayList, Hashtable

    nested = ArrayList()
    nested.Add("重复字符串")
    value = Hashtable()
    value.Add("first", nested)
    value.Add("second", nested)
    with ThreadPoolExecutor(max_workers=8) as workers:
        packets = list(workers.map(lambda _: _encode(value), range(32)))
    GC.Collect()
    contexts = [_decode(packet) for packet in packets]
    contexts[0]["first"].append("只改当前容器")
    assert contexts[0]["second"] == ["重复字符串"]
    assert all(item == {"first": ["重复字符串"], "second": ["重复字符串"]} for item in contexts[1:])


def test_packet_rejects_unsupported_keys_values_and_cycles():
    _load_dotnet_types()
    from System import Object
    from System.Collections import Hashtable

    for key, item, message in [(1, None, "键必须是字符串"), ("bad", Object(), "不支持的 CLR")]:
        value = Hashtable()
        value.Add(key, item)
        with pytest.raises(Exception, match=message):
            _encode(value)
    cyclic = Hashtable()
    cyclic.Add("cycle", cyclic)
    with pytest.raises(Exception, match="循环引用"):
        _encode(cyclic)


@pytest.mark.parametrize("format", ["seconds", "gcd", "vector"])
def test_observations_are_independent_after_caller_mutation_and_reset(format):
    from copy import deepcopy

    with InProcessBackend("black_mage", max_history=2) as backend:
        assert backend.submit_action(0, "blizzard_iii").accepted
        backend.advance_to(4)
        backend.record_policy_action(4, "ogcd_wait", 6)
        first = backend.observe_at(4, format=format, next_observation_timestamp=6).context
        expected = deepcopy(first)
        if format == "vector":
            first["state_history_context"]["tokens"][0]["player_state"][0] = 123456
            first["skill_history_context"][0]["skill_key"] = "caller_mutation"
        else:
            first.clear()
        assert backend.observe_at(4, format=format, next_observation_timestamp=6).context == expected
        backend.init(initial_timestamp=0)
        fresh = backend.observe_at(0, format="vector", next_observation_timestamp=0).context
        assert fresh["skill_history_context"] == []
