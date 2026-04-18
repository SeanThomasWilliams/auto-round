import os
from types import SimpleNamespace

import pytest
import torch

from auto_round.utils import device as device_utils


def test_build_max_memory_dict_force_disk_offload_uses_home_default(monkeypatch, tmp_path):
    monkeypatch.delenv("AR_WORK_SPACE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AR_FORCE_DISK_OFFLOAD", "1")
    monkeypatch.delenv("AR_MAX_GPU_MEMORY_GB", raising=False)
    monkeypatch.delenv("AR_MAX_CPU_MEMORY_GB", raising=False)

    monkeypatch.setattr(device_utils, "get_max_memory_with_uma_correction", lambda: {0: 123, "cpu": 456})
    monkeypatch.setattr(device_utils.shutil, "disk_usage", lambda path: SimpleNamespace(total=0, used=0, free=987))

    max_memory, offload_dir = device_utils.build_max_memory_dict(device_map="auto")

    assert max_memory[0] == 16 * 1024**3
    assert max_memory["cpu"] == 100 * 1024**3
    assert max_memory["disk"] == 987
    assert offload_dir == os.path.join(str(tmp_path), ".cache", "auto-round", "offload")


def test_build_max_memory_dict_explicit_caps_override_force_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("AR_WORK_SPACE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AR_FORCE_DISK_OFFLOAD", "1")
    monkeypatch.setenv("AR_MAX_GPU_MEMORY_GB", "8")
    monkeypatch.setenv("AR_MAX_CPU_MEMORY_GB", "12")

    monkeypatch.setattr(device_utils, "get_max_memory_with_uma_correction", lambda: {0: 123, "cpu": 456})
    monkeypatch.setattr(device_utils.shutil, "disk_usage", lambda path: SimpleNamespace(total=0, used=0, free=654))

    max_memory, offload_dir = device_utils.build_max_memory_dict(device_map="auto")

    assert max_memory[0] == 8 * 1024**3
    assert max_memory["cpu"] == 12 * 1024**3
    assert max_memory["disk"] == 654
    assert offload_dir == os.path.join(str(tmp_path), ".cache", "auto-round", "offload")


def test_dispatch_model_no_offload_aware_uses_disk_offload_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("AR_WORK_SPACE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AR_FORCE_DISK_OFFLOAD", "1")

    monkeypatch.setattr(device_utils, "is_single_device_no_offload", lambda requested_device_map: False)
    captured = {}

    def fake_dispatch_model(model, device_map, **kwargs):
        captured.update(kwargs)
        captured["device_map"] = device_map
        return model

    monkeypatch.setattr(device_utils, "dispatch_model", fake_dispatch_model)

    model = torch.nn.Linear(2, 2)
    result = device_utils.dispatch_model_no_offload_aware(model, {"layer": "disk"})

    assert result is model
    assert captured["device_map"] == {"layer": "disk"}
    assert captured["offload_dir"] == os.path.join(str(tmp_path), ".cache", "auto-round", "offload")
