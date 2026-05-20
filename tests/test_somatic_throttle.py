"""tests/test_somatic_throttle.py — Unit tests for SomaticComputeSentinel

Verifies parameter cuts occur under elevated arousal or hardware stress.
"""
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from core.brain.llm.somatic_throttle import SomaticComputeSentinel


def test_somatic_throttle_normal():
    # Test normal/unstressed parameters remain unchanged
    with (
        patch("core.brain.llm.somatic_throttle.resolve_affect_engine") as mock_resolve,
        patch("psutil.cpu_percent", return_value=15.0),
        patch("psutil.virtual_memory") as mock_vm,
    ):
        mock_affect = MagicMock()
        mock_affect.current = SimpleNamespace(arousal=0.2)
        mock_resolve.return_value = mock_affect
        
        mock_vm_obj = MagicMock()
        mock_vm_obj.percent = 40.0
        mock_vm.return_value = mock_vm_obj
        
        sentinel = SomaticComputeSentinel()
        opts = {"max_tokens": 512, "temperature": 0.7, "recurrent_depth": 0.8}
        adjusted = sentinel.adjust_generation_options(opts.copy())
        
        assert adjusted["max_tokens"] == 512
        assert adjusted["temperature"] == 0.7
        assert adjusted["recurrent_depth"] == 0.8


def test_somatic_throttle_stressed():
    # Test stressed parameters cap max_tokens and adjust temp/lane depth
    with (
        patch("core.brain.llm.somatic_throttle.resolve_affect_engine") as mock_resolve,
        patch("psutil.cpu_percent", return_value=50.0),
        patch("psutil.virtual_memory") as mock_vm,
    ):
        mock_affect = MagicMock()
        mock_affect.current = SimpleNamespace(arousal=0.85)  # Elevated arousal!
        mock_resolve.return_value = mock_affect
        
        mock_vm_obj = MagicMock()
        mock_vm_obj.percent = 50.0
        mock_vm.return_value = mock_vm_obj
        
        sentinel = SomaticComputeSentinel()
        opts = {"max_tokens": 512, "temperature": 0.7, "recurrent_depth": 0.8}
        adjusted = sentinel.adjust_generation_options(opts.copy())
        
        assert adjusted["max_tokens"] == 256
        assert adjusted["temperature"] == 0.3
        assert adjusted["recurrent_depth"] == 0.4


def test_somatic_throttle_critical():
    # Test critical parameters restrict max_tokens to 128
    with (
        patch("core.brain.llm.somatic_throttle.resolve_affect_engine") as mock_resolve,
        patch("psutil.cpu_percent", return_value=95.0),  # High CPU
        patch("psutil.virtual_memory") as mock_vm,
    ):
        mock_affect = MagicMock()
        mock_affect.current = SimpleNamespace(arousal=0.95)  # Critical arousal!
        mock_resolve.return_value = mock_affect
        
        mock_vm_obj = MagicMock()
        mock_vm_obj.percent = 95.0  # Critical RAM
        mock_vm.return_value = mock_vm_obj
        
        sentinel = SomaticComputeSentinel()
        opts = {"max_tokens": 512, "temperature": 0.7, "recurrent_lane_depth": 0.8}
        adjusted = sentinel.adjust_generation_options(opts.copy())
        
        assert adjusted["max_tokens"] == 128
        assert adjusted["temperature"] == 0.15
        assert adjusted["recurrent_lane_depth"] == 0.2
