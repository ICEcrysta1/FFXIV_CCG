from __future__ import annotations

import json
from types import SimpleNamespace

import main


class _FakeBackend:
    instances: list[_FakeBackend] = []

    def __init__(self, *, job_tag: str):
        self.job_tag = job_tag
        self.timestamp = 0.0
        self.calls: list[tuple[str, object]] = []
        self.__class__.instances.append(self)

    def __enter__(self) -> _FakeBackend:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def observe_at(
        self,
        timestamp: float,
        *,
        format: str = "seconds",
        next_observation_timestamp: float | None = None,
    ) -> SimpleNamespace:
        self.calls.append(("observe_at", (timestamp, format, next_observation_timestamp)))
        if format == "vector":
            return SimpleNamespace(
                context={
                    "candidate_skill_context": [
                        {"skill_key": "fire_iii", "is_legal": True},
                        {"skill_key": "fire_iv", "is_legal": False},
                    ]
                }
            )
        return SimpleNamespace(
            context={
                "time_seconds": self.timestamp,
                "cast_remaining_seconds": 2.5,
                "gcd_remaining_seconds": 2.5,
                "mp": 0,
                "astral_fire": 3,
                "umbral_ice": 0,
                "umbral_hearts": 0,
                "astral_soul": 0,
                "polyglot": 0,
            }
        )

    def submit_action(self, timestamp: float, action: str) -> SimpleNamespace:
        self.calls.append(("submit_action", (timestamp, action)))
        return SimpleNamespace(
            accepted=True,
            accepted_timestamp=timestamp,
            reason="",
        )

    def advance_to(self, timestamp: float) -> SimpleNamespace:
        self.calls.append(("advance_to", timestamp))
        self.timestamp = timestamp
        return SimpleNamespace(timestamp=timestamp)


class _FakeEngine:
    def __init__(self, job_tag, *, capacity):
        self.job_tag = job_tag
        self.capacity = capacity

    def create_backend(self, **_kwargs):
        return _FakeBackend(job_tag=self.job_tag)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_list_actions_uses_vector_observation(monkeypatch, capsys):
    _FakeBackend.instances.clear()
    monkeypatch.setattr(main, "InProcessEngine", _FakeEngine)

    monkeypatch.setattr("sys.argv", ["main", "list-actions", "--job-tag", "black_mage"])
    main.main()

    assert capsys.readouterr().out.splitlines() == ["fire_iii"]
    calls = _FakeBackend.instances[0].calls
    assert calls == [("observe_at", (0.0, "vector", 0.0))]


def test_smoke_uses_absolute_time_protocol(monkeypatch, capsys):
    _FakeBackend.instances.clear()
    monkeypatch.setattr(main, "InProcessEngine", _FakeEngine)

    monkeypatch.setattr("sys.argv", ["main", "smoke", "--job-tag", "black_mage"])
    main.main()

    output = json.loads(capsys.readouterr().out)
    assert output["time"] == 15.0
    calls = _FakeBackend.instances[0].calls
    assert sum(call[0] == "submit_action" for call in calls) == 6
    assert not any(call[0] in {"step", "advance", "format_vector_state"} for call in calls)
