"""验证真实事件、Projector 文件及诊断采样的边界与副作用。"""

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from common.policy.config import ModelConfig
from common.policy.data import SkillVocab
from common.policy.model import CausalPolicyModel
from common.training.tensorboard import TensorBoardConfig, TensorBoardMonitor
from tests.training._causal_fixtures import make_batch, make_data_spec


def _model():
    spec = make_data_spec(action_to_vocab_id=(2, 1), action_is_gcd=(True, False))
    return CausalPolicyModel(
        spec, ModelConfig(d_model=8, n_layers=1, n_heads=2, num_kv_heads=1,
                          ff_dim=16, dropout=0.1), vocab_size=4,
    )


@pytest.mark.parametrize("name", ["graph", "pr_curves"])
def test_config_requires_real_booleans(name):
    with pytest.raises(TypeError, match=f"{name} must be a boolean"):
        TensorBoardConfig.from_mapping({name: 1})


@pytest.mark.parametrize("name", ["histogram_every_steps", "embedding_every_epochs", "max_histogram_values", "pr_curve_max_samples"])
def test_diagnostic_config_requires_positive_integers(name):
    for value, exception in ((0, ValueError), (True, TypeError), (1.5, TypeError)):
        with pytest.raises(exception, match=name):
            TensorBoardConfig.from_mapping({name: value})


def test_real_event_files_activate_supported_panels_and_projector(tmp_path):
    model = _model()
    vocab = SkillVocab.from_entries([(101, 1), (102, 2), (103, 3)])
    writer = SummaryWriter(str(tmp_path))
    monitor = TensorBoardMonitor(
        writer, TensorBoardConfig(enabled=True, histogram_every_steps=2), model=model,
        vocab=vocab, metadata={"training": {"seed": 42}},
    )
    batch = make_batch(model.data_spec, batch_size=2)
    rng = torch.get_rng_state().clone()
    monitor.log_graph(batch)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert model.training
    logits = model(batch)["logits"]
    logits.sum().backward()
    monitor.log_gradients(1, grad_norm=torch.tensor(0.3))
    monitor.log_step(logits, global_step=1, labels=batch["label_index"])
    monitor.begin_validation()
    monitor.update_validation(torch.tensor([[3., 1.], [2., 1.], [0., 4.]]), torch.tensor([0, 1, 1]))
    monitor.end_validation(global_step=2)
    monitor.log_epoch(global_step=2)
    monitor.finish({"validation/top1_accuracy": 2 / 3}, global_step=2)
    monitor.finish({"validation/top1_accuracy": 0}, global_step=3)
    writer.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    tags = events.Tags()
    assert tags["graph"] is True
    assert "parameters/output_head.weight" in tags["histograms"]
    assert "gradients/output_head.weight" in tags["distributions"]
    assert "validation/confusion_matrix" in tags["images"]
    assert "run/configuration/text_summary" in tags["tensors"]
    assert "validation/actions/first" in tags["tensors"]
    assert "custom_scalars__config__" in tags["tensors"]
    assert events.Scalars("validation/actions/second/recall")[0].value == pytest.approx(0.5)
    assert events.Scalars("validation/actions/first/precision")[0].value == pytest.approx(0.5)
    assert len(events.Scalars("validation/top1_accuracy")) == 1
    assert events.Scalars("validation/top1_accuracy")[0].step == 2
    plugin_tags = {tag for tag, metadata in events.summary_metadata.items() if metadata.plugin_data.plugin_name == "hparams"}
    assert "_hparams_/experiment" in plugin_tags
    assert "_hparams_/session_start_info" in plugin_tags
    assert "_hparams_/session_end_info" in plugin_tags
    projector = (tmp_path / "projector_config.pbtxt").read_text(encoding="utf-8")
    assert "embeddings/input_skills:00000" in projector
    assert "embeddings/output_actions:00002" in projector
    metadata_files = list(tmp_path.rglob("metadata.tsv"))
    assert len(metadata_files) == 4
    input_metadata = next(path for path in metadata_files if "input_skills" in str(path) and "00000" in str(path)).read_text(encoding="utf-8").splitlines()
    assert input_metadata[1:] == ["second\t101\t1\togcd", "first\t102\t2\tgcd", "raw_skill_103\t103\t3\thistory_only"]
    output_metadata = next(path for path in metadata_files if "output_actions" in str(path) and "00000" in str(path)).read_text(encoding="utf-8").splitlines()
    assert output_metadata[1:] == ["first\t102\t2\tgcd", "second\t101\t1\togcd"]
    assert all(path.name.isdigit() for path in tmp_path.iterdir() if path.is_dir())


def test_histograms_are_bounded_finite_and_deterministic():
    class Writer:
        def __init__(self):
            self.values = []

        def add_histogram(self, tag, values, step, **kwargs):
            self.values.append((tag, values.clone(), step))

    writer = Writer()
    model = _model()
    monitor = TensorBoardMonitor(writer, TensorBoardConfig(enabled=True, max_histogram_values=4), model=model)
    rng = torch.get_rng_state().clone()
    monitor.log_distributions({"values": torch.tensor([0., 1., float("inf"), 3., float("nan"), 5., 6., 7., 8.])}, global_step=4)
    assert writer.values[0][0] == "rollout/values"
    assert writer.values[0][1].numel() <= 4
    assert torch.isfinite(writer.values[0][1]).all()
    torch.testing.assert_close(writer.values[0][1], torch.tensor([0., 3., 6.]))
    torch.testing.assert_close(torch.get_rng_state(), rng)


def test_validation_reservoir_is_bounded_batch_invariant_and_covers_all_confusion_samples():
    config = TensorBoardConfig(enabled=True, pr_curve_max_samples=7)
    model = _model()
    one = TensorBoardMonitor(object(), config, model=model)
    many = TensorBoardMonitor(object(), config, model=model)
    logits = torch.arange(120).reshape(60, 2).float()
    labels = torch.arange(60) % 2
    rng = torch.get_rng_state().clone()
    one.update_validation(logits, labels)
    for start in range(0, 60, 3):
        many.update_validation(logits[start:start + 3], labels[start:start + 3])
        assert many._pr_labels.numel() <= 7
    assert int(one._confusion.sum()) == int(many._confusion.sum()) == 60
    torch.testing.assert_close(one._pr_priorities, many._pr_priorities)
    torch.testing.assert_close(one._pr_labels, many._pr_labels)
    torch.testing.assert_close(one._pr_probs, many._pr_probs)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert one._pr_priorities.numel() == 7


def test_disabled_monitor_does_not_forward_or_touch_writer():
    class Writer:
        def __getattr__(self, name):
            pytest.fail(f"disabled monitor touched writer.{name}")

    model = _model()
    monitor = TensorBoardMonitor(Writer(), TensorBoardConfig(), model=model)
    model.forward = lambda *_: pytest.fail("disabled monitor performed forward")
    monitor.log_graph({})
    monitor.log_step(torch.zeros(1, 2), global_step=1)
    monitor.log_gradients(1)
    monitor.log_epoch(global_step=1)
    monitor.update_validation(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long))
    monitor.end_validation(global_step=1)
    monitor.finish({"loss": 1}, global_step=1)


def test_validation_rejects_invalid_rows_explicitly_and_empty_round_is_visible():
    class Writer:
        def __init__(self):
            self.scalars = {}

        def add_scalar(self, tag, value, step):
            self.scalars[tag] = value

    writer = Writer()
    monitor = TensorBoardMonitor(writer, TensorBoardConfig(enabled=True), model=_model())
    monitor.update_validation(torch.tensor([[float("nan"), 0.], [0., 1.]]), torch.tensor([0, 5]))
    monitor.end_validation(global_step=2)
    assert writer.scalars["validation/diagnostic_samples"] == 0
    assert writer.scalars["validation/diagnostic_rejected_samples"] == 2
    assert not any(tag.endswith("/recall") for tag in writer.scalars)
    monitor.begin_validation()
    monitor.update_validation(torch.tensor([[1., 0.]]), torch.tensor([0]))
    monitor.end_validation(global_step=3)
    assert writer.scalars["validation/diagnostic_samples"] == 1
    assert writer.scalars["validation/diagnostic_rejected_samples"] == 0


def test_resumed_embedding_uses_actual_step_and_skips_unchanged_step(tmp_path):
    model = _model()
    with SummaryWriter(str(tmp_path)) as writer:
        monitor = TensorBoardMonitor(writer, TensorBoardConfig(enabled=True), model=model, initial_step=137)
        monitor.log_epoch(global_step=137)
    projector = (tmp_path / "projector_config.pbtxt").read_text(encoding="utf-8")
    assert "embeddings/input_skills:00137" in projector
    assert "00000" not in projector
    assert len(list(tmp_path.rglob("metadata.tsv"))) == 2
