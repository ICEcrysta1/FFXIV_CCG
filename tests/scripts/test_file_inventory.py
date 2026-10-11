"""文件校验索引的持久复用、依赖失效和损坏回退。"""

import json

import pytest

from scripts.common.file_inventory import FileInventory


def test_inventory_reuses_validation_and_invalidates_changed_dependencies(tmp_path):
    source, output, shard = [tmp_path / name for name in ("source", "output", "shard")]
    for path in (source, output, shard):
        path.write_bytes(b"original")
    index = tmp_path / "inventory.json"
    calls = []

    def check(context=1):
        with FileInventory(index) as inventory:
            return inventory.check(
                output, context=context, paths=[source, output],
                validate=lambda: (calls.append(context) or True, [shard]),
            )

    assert check() is True
    assert check() is True
    assert calls == [1]
    for path in (source, output, shard):
        path.write_bytes(path.read_bytes() + b"changed")
        assert check() is True
    assert calls == [1] * 4
    assert check(2) is True
    shard.unlink()
    assert check(2) is True
    assert calls == [1] * 4 + [2, 2]


@pytest.mark.parametrize("content", ["broken", "[]", '{"version":0,"entries":{}}', '{"version":1,"entries":[]}'])
def test_inventory_damage_falls_back_to_validator(tmp_path, content):
    index = tmp_path / "inventory.json"
    index.write_text(content, encoding="utf-8")
    with FileInventory(index) as inventory:
        assert inventory.check(index, context=1, paths=[], validate=lambda: (True, ())) is True
    assert json.loads(index.read_text(encoding="utf-8"))["version"] == 1


def test_inventory_does_not_save_result_when_file_changes_during_validation(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"before")
    index = tmp_path / "inventory.json"

    def validate():
        source.write_bytes(b"after")
        return True, ()

    with FileInventory(index) as inventory:
        assert inventory.check(source, context=1, paths=[source], validate=validate) is None
    assert not index.exists()
    with FileInventory(index) as inventory:
        assert inventory.check(source, context=1, paths=[source], validate=lambda: (False, ())) is False
    assert not index.exists()


def test_inventory_write_failure_does_not_break_processing(tmp_path, monkeypatch):
    from scripts.common import file_inventory

    def fail(*_args):
        raise PermissionError("read-only inventory")

    monkeypatch.setattr(file_inventory, "atomic_write_json", fail)
    with FileInventory(tmp_path / "index.json") as inventory:
        assert inventory.check(tmp_path, context=1, paths=[], validate=lambda: (True, ())) is True


def test_inventory_rejects_async_result_if_source_changed_after_submission(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"before")
    index = tmp_path / "inventory.json"
    before = FileInventory.snapshot([source])
    source.write_bytes(b"changed while worker was validating")
    with FileInventory(index) as inventory:
        assert inventory.check(
            source, context=1, paths=[source], before=before,
            validate=lambda: (True, ()),
        ) is None
        assert inventory.lookup(source, context=1, paths=[source]) is None
    assert not index.exists()


def test_2000_existing_with_200_stale_and_200_new_only_revalidates_400(tmp_path):
    sources = [tmp_path / f"report-{index}" for index in range(2000)]
    for source in sources:
        source.write_bytes(b"valid")
    calls = []
    index = tmp_path / "inventory.json"

    def scan():
        with FileInventory(index) as inventory:
            for source in sources:
                assert inventory.check(
                    source, context=1, paths=[source],
                    validate=lambda: (calls.append(source) or True, ()),
                ) is True

    scan()
    assert len(calls) == 2000
    calls.clear()
    for source in sources[:200]:
        source.write_bytes(b"outdated")
    new = [tmp_path / f"new-{index}" for index in range(200)]
    for source in new:
        source.write_bytes(b"new")
    sources.extend(new)
    scan()
    assert set(calls) == set(sources[:200] + new)
    assert len(calls) == 400
