"""验证真实事件、Projector 文件及诊断采样的边界与副作用。"""

import pytest
import torch
from google.protobuf import text_format
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.custom_scalar.layout_pb2 import Layout
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from torch.utils.tensorboard import SummaryWriter

from common.policy.config import ModelConfig
from common.policy.data import SkillVocab
from common.policy.model import CausalPolicyModel
from common.training.tensorboard import TensorBoardConfig, write_scalar_metrics
from common.training.tensorboard_labels import tensorboard_tag
from common.training.tensorboard_monitor import TensorBoardMonitor
from tests.training._causal_fixtures import make_batch, make_data_spec


def _model(n_layers=1):
    spec = make_data_spec(action_to_vocab_id=(2, 1), action_is_gcd=(True, False))
    return CausalPolicyModel(
        spec, ModelConfig(d_model=8, n_layers=n_layers, n_heads=2, num_kv_heads=1,
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
    assert "参数/输出动作头/权重（output_head.weight）" in tags["histograms"]
    assert "梯度/输出动作头/权重（output_head.weight）" in tags["distributions"]
    assert "验证/动作混淆矩阵（按真实动作归一化）" in tags["images"]
    assert "运行信息/配置/text_summary" in tags["tensors"]
    assert "验证/动作/first/精确率与召回率曲线" in tags["tensors"]
    assert "custom_scalars__config__" in tags["tensors"]
    assert events.Scalars("验证/动作/second/召回率")[0].value == pytest.approx(0.5)
    assert events.Scalars("验证/动作/first/精确率")[0].value == pytest.approx(0.5)
    assert len(events.Scalars("验证/Top-1准确率")) == 1
    assert events.Scalars("验证/Top-1准确率")[0].step == 2
    plugin_tags = {tag for tag, metadata in events.summary_metadata.items() if metadata.plugin_data.plugin_name == "hparams"}
    assert "_hparams_/experiment" in plugin_tags
    assert "_hparams_/session_start_info" in plugin_tags
    assert "_hparams_/session_end_info" in plugin_tags
    projector = text_format.Parse(
        (tmp_path / "projector_config.pbtxt").read_text(encoding="utf-8"), ProjectorConfig(),
    )
    names = {embedding.tensor_name for embedding in projector.embeddings}
    assert "嵌入向量/输入技能表:00000" in names
    assert "嵌入向量/输出动作头:00002" in names
    metadata_files = list(tmp_path.rglob("metadata.tsv"))
    assert len(metadata_files) == 4
    input_metadata = next(path for path in metadata_files if "输入技能表" in str(path) and "00000" in str(path)).read_text(encoding="utf-8").splitlines()
    assert input_metadata[0] == "技能标识\t原始技能ID\t词表ID\t技能类型"
    assert input_metadata[1:] == ["second\t101\t1\togcd", "first\t102\t2\tgcd", "raw_skill_103\t103\t3\thistory_only"]
    output_metadata = next(path for path in metadata_files if "输出动作头" in str(path) and "00000" in str(path)).read_text(encoding="utf-8").splitlines()
    assert output_metadata[1:] == ["first\t102\t2\tgcd", "second\t101\t1\togcd"]
    assert all(path.name.isdigit() for path in tmp_path.iterdir() if path.is_dir())


def test_validation_figure_falls_back_without_cjk_fonts(tmp_path, monkeypatch):
    from matplotlib import font_manager, rcParams

    # 仅保留 Matplotlib 自带字体，稳定复现 Linux 未安装中文字体的环境。
    monkeypatch.setattr(font_manager.fontManager, "ttflist", [
        font for font in font_manager.fontManager.ttflist if font.name == "DejaVu Sans"
    ])
    font_family = list(rcParams["font.family"])
    with SummaryWriter(str(tmp_path)) as writer:
        monitor = TensorBoardMonitor(writer, TensorBoardConfig(enabled=True), model=_model())
        monitor.update_validation(torch.tensor([[3., 1.], [0., 4.]]), torch.tensor([0, 1]))
        monitor.end_validation(global_step=2)

    events = EventAccumulator(str(tmp_path)).Reload()
    image = events.Images(tensorboard_tag("validation/confusion_matrix"))[0]
    assert image.step == 2
    assert image.width > 0 and image.height > 0
    assert image.encoded_image_string.startswith(b"\x89PNG\r\n\x1a\n")
    assert tensorboard_tag("validation/actions/first") in events.Tags()["tensors"]
    assert rcParams["font.family"] == font_family


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
    assert writer.values[0][0] == "回放/values"
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
    assert writer.scalars["验证/有效诊断样本数"] == 0
    assert writer.scalars["验证/拒绝诊断样本数"] == 2
    assert not any(tag.endswith("/召回率") for tag in writer.scalars)
    monitor.begin_validation()
    monitor.update_validation(torch.tensor([[1., 0.]]), torch.tensor([0]))
    monitor.end_validation(global_step=3)
    assert writer.scalars["验证/有效诊断样本数"] == 1
    assert writer.scalars["验证/拒绝诊断样本数"] == 0


def test_resumed_embedding_uses_actual_step_and_skips_unchanged_step(tmp_path):
    model = _model()
    with SummaryWriter(str(tmp_path)) as writer:
        monitor = TensorBoardMonitor(writer, TensorBoardConfig(enabled=True), model=model, initial_step=137)
        monitor.log_epoch(global_step=137)
    projector = text_format.Parse(
        (tmp_path / "projector_config.pbtxt").read_text(encoding="utf-8"), ProjectorConfig(),
    )
    names = {embedding.tensor_name for embedding in projector.embeddings}
    assert "嵌入向量/输入技能表:00137" in names
    assert all("00000" not in name for name in names)
    assert len(list(tmp_path.rglob("metadata.tsv"))) == 2


@pytest.mark.parametrize("n_layers", [1, 3])
def test_layer_labels_follow_actual_model_and_keep_unique_parameter_paths(tmp_path, n_layers):
    model = _model(n_layers=n_layers)
    with SummaryWriter(str(tmp_path)) as writer:
        monitor = TensorBoardMonitor(writer, TensorBoardConfig(enabled=True), model=model)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        monitor.log_gradients(1)
        monitor.log_epoch(global_step=1)
    tags = EventAccumulator(str(tmp_path)).Reload().Tags()
    gradients = tags["histograms"]
    assert len(gradients) == 2 * len(list(model.named_parameters()))
    for index in range(n_layers):
        for module, label in (
            ("q_a", "查询低秩投影"), ("q_b", "查询展开投影"),
            ("kv_proj", "共享键值投影"), ("q_norm", "查询低秩归一化"),
            ("kv_norm", "共享键值归一化"),
        ):
            name = f"encoder.layers.{index}.self_attn.{module}.weight"
            suffix = f"编码器第{index + 1}层/自注意力/{label}/权重（{name}）"
            assert f"梯度/{suffix}" in gradients
            assert f"参数/{suffix}" in gradients
    assert not any(f"编码器第{n_layers + 1}层/" in tag for tag in gradients)


def test_unknown_fields_and_action_ids_are_preserved_without_fixed_layer_limit():
    assert tensorboard_tag("gradients/encoder.layers.37.gate_proj.bias") == (
        "梯度/编码器第38层/前馈门控投影/偏置（encoder.layers.37.gate_proj.bias）"
    )
    assert tensorboard_tag("parameters/input_encoder.scene_proj.7.weight") == (
        "参数/输入编码器/场景投影/第8项/权重（input_encoder.scene_proj.7.weight）"
    )
    assert tensorboard_tag("parameters/custom_module.weight") == "参数/custom_module/权重（custom_module.weight）"
    assert tensorboard_tag("validation/actions/loss/recall") == "验证/动作/loss/召回率"
    assert tensorboard_tag("validation/actions/loss") == "验证/动作/loss/精确率与召回率曲线"
    assert tensorboard_tag("train/step/custom_metric") == "训练/记录区间/custom_metric"
    assert tensorboard_tag("diagnostics/nonfinite_samples/gradients/encoder.layers.37.gate_proj.bias") == (
        "诊断/非有限值采样数/梯度/编码器第38层/前馈门控投影/偏置（encoder.layers.37.gate_proj.bias）"
    )


@pytest.mark.parametrize("stage", ["bc", "grpo"])
def test_custom_scalar_layout_references_real_chinese_events(tmp_path, stage):
    groups = ({
        "grpo/update": {"loss": 1., "policy_loss": 0.5, "kl": 0.1, "learning_rate": 0.01, "next_learning_rate": 0.005},
        "grpo/iteration": {"greedy_ppg_before": 100., "greedy_ppg_after": 110.},
    } if stage == "grpo" else {
        "train/step": {"loss": 1., "top1_accuracy": 0.5, "learning_rate": 0.01},
        "train/epoch": {"loss": 1.},
        "validation": {"loss": 1., "top1_accuracy": 0.5, "top3_accuracy": 1., "val_ppg": 100., "none_ppg": 110.},
    })
    with SummaryWriter(str(tmp_path)) as writer:
        TensorBoardMonitor(writer, TensorBoardConfig(enabled=True), model=_model(), metadata={"stage": stage})
        for prefix, metrics in groups.items():
            write_scalar_metrics(writer, metrics, prefix=prefix, global_step=17)
    events = EventAccumulator(str(tmp_path)).Reload()
    layout = Layout.FromString(events.Tensors("custom_scalars__config__")[0].tensor_proto.string_val[0])
    assert layout.category[0].title == ("GRPO训练" if stage == "grpo" else "行为克隆训练")
    assert {chart.title for chart in layout.category[0].chart} >= {"损失", "回放PPG", "学习率"}
    for category in layout.category:
        for chart in category.chart:
            for tag in chart.multiline.tag:
                assert tag in events.Tags()["scalars"]
                assert events.Scalars(tag)[0].step == 17


def test_nonfinite_histogram_diagnostics_use_the_same_chinese_source_label():
    class Writer:
        def __init__(self):
            self.scalars = {}
            self.histograms = {}

        def add_scalar(self, tag, value, step):
            self.scalars[tag] = value

        def add_histogram(self, tag, values, step, **kwargs):
            self.histograms[tag] = values

    writer = Writer()
    monitor = TensorBoardMonitor(writer, TensorBoardConfig(enabled=True), model=_model())
    monitor.log_distributions({"policy/ratio": torch.tensor([1., float("inf")])}, global_step=1, prefix="grpo/update")
    assert writer.scalars == {"诊断/非有限值采样数/GRPO/优化更新/策略/新旧策略概率比": 1}
    torch.testing.assert_close(writer.histograms["GRPO/优化更新/策略/新旧策略概率比"], torch.tensor([1.]))
