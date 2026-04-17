import copy
from dataclasses import fields

import torch.nn as nn

from auto_round.export.export_to_autoround.export import _build_extra_config_for_autoround_export
from auto_round.schemes import QuantizationScheme


class DummySharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 4)


class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Identity()
        self.shared_expert = DummySharedExpert()


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = DummyMLP()


class DummyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer()])


class DummyHFRoot(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = DummyLanguageModel()


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyHFRoot()


def make_quantization_config():
    return {
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "data_type": "int",
        "act_bits": 16,
        "act_group_size": None,
        "act_sym": None,
        "act_data_type": None,
        "act_dynamic": True,
        "super_bits": None,
        "super_group_size": None,
        "hadamard_config": None,
        "block_name_to_quantize": None,
        "regex_config": {},
    }


def test_build_extra_config_expands_regex_to_literal_keys_with_vllm_alias(monkeypatch):
    monkeypatch.delenv("AR_EMIT_VLLM_COMPAT_PREFIXES", raising=False)
    model = DummyModel()
    layer_config = {
        "model.language_model.layers.0.mlp.shared_expert.gate_proj": {
            "bits": 16,
            "group_size": 128,
            "sym": True,
            "data_type": "fp",
            "act_bits": 16,
            "act_group_size": None,
            "act_sym": None,
            "act_data_type": None,
            "act_dynamic": True,
            "super_bits": None,
            "super_group_size": None,
            "hadamard_config": None,
            "in_blocks": True,
        }
    }
    quantization_config = make_quantization_config()
    quantization_config["regex_config"] = {
        "mlp.gate": {
            "bits": 16,
            "group_size": 128,
            "sym": True,
            "data_type": "fp",
        }
    }

    extra_config = _build_extra_config_for_autoround_export(
        model=model,
        layer_config=layer_config,
        quantization_config=copy.deepcopy(quantization_config),
        scheme_keys=[f.name for f in fields(QuantizationScheme)],
    )

    assert extra_config["model.language_model.layers.0.mlp.shared_expert.gate_proj"] == {
        "bits": 16,
        "data_type": "fp",
    }
    assert extra_config["model.language_model.layers.0.mlp.gate"] == {
        "bits": 16,
        "data_type": "fp",
    }
    assert extra_config["language_model.model.layers.0.mlp.gate"] == {
        "bits": 16,
        "data_type": "fp",
    }


def test_build_extra_config_can_disable_vllm_alias_emission(monkeypatch):
    monkeypatch.setenv("AR_EMIT_VLLM_COMPAT_PREFIXES", "0")
    model = DummyModel()
    quantization_config = make_quantization_config()
    quantization_config["regex_config"] = {
        "mlp.gate": {
            "bits": 16,
            "group_size": 128,
            "sym": True,
            "data_type": "fp",
        }
    }

    extra_config = _build_extra_config_for_autoround_export(
        model=model,
        layer_config={},
        quantization_config=copy.deepcopy(quantization_config),
        scheme_keys=[f.name for f in fields(QuantizationScheme)],
    )

    assert extra_config["model.language_model.layers.0.mlp.gate"] == {
        "bits": 16,
        "data_type": "fp",
    }
    assert "language_model.model.layers.0.mlp.gate" not in extra_config
