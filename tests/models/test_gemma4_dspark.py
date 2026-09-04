# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4DSparkForCausalLM must not register its attention layers under the
target model's names.

Every Attention layer registers itself in
``compilation_config.static_forward_context`` under its prefix and raises
``ValueError("Duplicate layer name: ...")`` on a repeat
(vllm/model_executor/layers/attention/attention.py). The target Gemma-4 model
owns ``model.layers.{i}.self_attn.attn``; the draft used to be built under the
same ``model`` root, so loading ANY Gemma4DSparkModel checkpoint on a Gemma-4
target failed at ``model.layers.0.self_attn.attn``. The Qwen3 DSpark path
offsets its layer names past the target's (``start_layer_id``).

These tests build the draft on CPU with the sub-modules that need a TP group
or an attention backend replaced by recorders, so what is exercised is exactly
the naming: the prefix each decoder layer is registered under, the module path
each layer's parameters live at (which the checkpoint's ``layers.{i}.*``
tensors must keep resolving to), and the per-layer config index. They hold for
both fixes of the defect (a distinct draft root prefix, or a start_layer_id
offset with an explicit draft-local layer index).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config import (
    DeviceConfig,
    VllmConfig,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.config.compilation import CompilationMode
from vllm.model_executor.models import gemma4_dspark
from vllm.model_executor.models.utils import extract_layer_index

TARGET_NUM_LAYERS = 30  # Gemma-4-26B-A4B
DRAFT_NUM_LAYERS = 5  # the published Gemma-4 DSpark drafts are 5-layer

TARGET_ATTN_NAMES = frozenset(
    f"model.layers.{i}.self_attn.attn" for i in range(TARGET_NUM_LAYERS)
)


class _Dummy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _RecordingDecoderLayer(nn.Module):
    """Stands in for Gemma4DSparkDecoderLayer: registers the name its
    attention layer registers, with the duplicate check Attention.__init__
    applies, and records what it was handed."""

    def __init__(
        self,
        config,
        cache_config,
        quant_config,
        prefix: str,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.layer_idx = layer_idx
        self.attn_name = f"{prefix}.self_attn.attn"
        ctx = get_current_vllm_config().compilation_config.static_forward_context
        if self.attn_name in ctx:
            raise ValueError(f"Duplicate layer name: {self.attn_name}")
        ctx[self.attn_name] = self
        # One parameter per layer, so named_parameters() shows the module path.
        self.weight = nn.Parameter(torch.zeros(1))


def _draft_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=16,
        num_hidden_layers=DRAFT_NUM_LAYERS,
        vocab_size=64,
        rms_norm_eps=1e-6,
        target_layer_ids=[3, 12, 21],
        markov_rank=4,
        layer_types=["sliding_attention"] * (DRAFT_NUM_LAYERS - 1) + ["full_attention"],
    )


@pytest.fixture
def vllm_config(monkeypatch: pytest.MonkeyPatch) -> VllmConfig:
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    cfg.compilation_config.mode = CompilationMode.NONE
    cfg.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=_draft_hf_config())
    )
    cfg.model_config = SimpleNamespace(
        dtype=torch.float32,
        get_num_layers=lambda parallel_config: TARGET_NUM_LAYERS,
    )
    for name in (
        "VocabParallelEmbedding",
        "ReplicatedLinear",
        "RMSNorm",
        "DSparkMarkovHead",
        "ParallelLMHead",
        "LogitsProcessor",
    ):
        monkeypatch.setattr(gemma4_dspark, name, _Dummy)
    monkeypatch.setattr(
        gemma4_dspark, "Gemma4DSparkDecoderLayer", _RecordingDecoderLayer
    )
    # The target's attention layers, as Gemma4ForCausalLM registers them
    # before the drafter is built.
    ctx = cfg.compilation_config.static_forward_context
    for name in TARGET_ATTN_NAMES:
        ctx[name] = object()
    return cfg


def _build(cfg: VllmConfig) -> gemma4_dspark.Gemma4DSparkForCausalLM:
    with set_current_vllm_config(cfg):
        return gemma4_dspark.Gemma4DSparkForCausalLM(vllm_config=cfg)


@pytest.mark.cpu_test
def test_draft_attention_names_do_not_collide_with_the_target(vllm_config):
    model = _build(vllm_config)  # raised "Duplicate layer name" before the fix
    layers = list(model.model.layers)
    assert len(layers) == DRAFT_NUM_LAYERS
    draft_names = {layer.attn_name for layer in layers}
    assert len(draft_names) == DRAFT_NUM_LAYERS
    assert draft_names.isdisjoint(TARGET_ATTN_NAMES)
    ctx = vllm_config.compilation_config.static_forward_context
    assert set(ctx) == TARGET_ATTN_NAMES | draft_names


@pytest.mark.cpu_test
def test_checkpoint_parameter_paths_are_unchanged(vllm_config):
    """load_weights prepends "model." to every non-lm_head checkpoint name,
    so the draft's layers must still live at model.layers.{i} whatever
    prefix they are registered under."""
    model = _build(vllm_config)
    names = dict(model.named_parameters())
    for i in range(DRAFT_NUM_LAYERS):
        assert f"model.layers.{i}.weight" in names


@pytest.mark.cpu_test
def test_per_layer_config_index_is_the_draft_local_index(vllm_config):
    """layer_types / gemma4_layer_config are indexed by the draft's own
    layer number, whatever prefix the layer is registered under."""
    model = _build(vllm_config)
    layer_types = (
        vllm_config.speculative_config.draft_model_config.hf_config.layer_types
    )
    for i, layer in enumerate(model.model.layers):
        local = (
            layer.layer_idx
            if layer.layer_idx is not None
            else extract_layer_index(layer.prefix)
        )
        assert local == i
        assert 0 <= local < len(layer_types)


@pytest.mark.cpu_test
def test_registering_the_draft_under_the_target_root_collides(vllm_config):
    """The defect, stated positively: a draft backbone built under the
    target's own root with no offset collides on its first layer."""
    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(
            ValueError, match=r"Duplicate layer name: model\.layers\.0\.self_attn\.attn"
        ),
    ):
        gemma4_dspark.Gemma4DSparkModel(vllm_config=vllm_config, prefix="model")


@pytest.mark.cpu_test
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Gemma4DSparkForCausalLM.load_weights skips every confidence_head "
        "tensor and Gemma4DSparkModel builds no confidence head, unlike "
        "Qwen3DSparkModel under enable_confidence_head; adaptive verification "
        "is unavailable on the Gemma-4 DSpark path. Documented, not fixed."
    ),
)
def test_confidence_head_weights_are_loaded(vllm_config, monkeypatch):
    model = _build(vllm_config)
    monkeypatch.setattr(model.model, "_build_fused_kv_buffers", lambda: None)
    loaded = model.load_weights(
        [
            ("confidence_head.proj.weight", torch.zeros(1, 16)),
            ("confidence_head.proj.bias", torch.zeros(1)),
        ]
    )
    assert {
        "model.confidence_head.proj.weight",
        "model.confidence_head.proj.bias",
    } <= loaded
