"""A driver for a device that is not here is run against the device as its datasheet gives it.

A correct driver passes every scenario and keeps every rule; drivers with the
usual mistakes (starting while the device is busy, leaving its interrupt
pending, reading more than it holds, overrunning their own buffer) are caught,
by the device's rules or by the sanitizers.
"""
from __future__ import annotations

import shutil

import pytest

from core.stand_ins.a_device_on_paper import DeviceSpec, the_model_in_c, tried_on_paper

pytestmark = pytest.mark.skipif(shutil.which("cc") is None and shutil.which("clang") is None, reason="no C compiler")

# A small DMA engine: words are written into TX, LEN says how many, START sends
# them; BUSY while it does, then DONE and an interrupt; RX holds what came back.
_DMA = DeviceSpec("a small DMA engine", registers=[
    {"name": "CTRL", "offset": 0x00, "access": "rw", "fields": {"START": [0, 1], "IRQ_EN": [1, 1]}},
    {"name": "STATUS", "offset": 0x04, "access": "ro", "fields": {"BUSY": [0, 1], "DONE": [1, 1]}},
    {"name": "IRQ", "offset": 0x08, "access": "w1c", "fields": {"DONE": [0, 1]}},
    {"name": "LEN", "offset": 0x0C, "access": "rw"},
    {"name": "TX", "offset": 0x10, "access": "wo", "fifo": {"depth": 8}},
], behaviour=[
    {"on_write": "CTRL.START", "equals": 1, "requires": [["STATUS.BUSY", 0]], "requires_nonzero": ["LEN"],
     "now": [["STATUS.BUSY", 1], ["STATUS.DONE", 0]], "fifo_in": {"from": "TX", "count": "LEN"},
     "after_cycles": 20, "later": [["STATUS.BUSY", 0], ["STATUS.DONE", 1], ["IRQ.DONE", 1]], "interrupt_if": "CTRL.IRQ_EN"},
])

_CAREFUL = r"""
static int send(const uint32_t *words, int n) {
  if (n <= 0 || n > 8) return -1;
  while (readl(0x04) & 1u) { }                  /* wait while BUSY */
  for (int i = 0; i < n; i++) writel(words[i], 0x10);
  writel((uint32_t)n, 0x0C);
  writel(0x3u, 0x00);                           /* START, with the interrupt on */
  while (!dev_irq()) dev_tick();
  writel(0x1u, 0x08);                           /* clear DONE */
  return (readl(0x04) & 2u) ? 0 : -2;
}
"""

_SCENARIOS = {
    "sends one word": "uint32_t w[1] = {7}; CHECK(send(w, 1) == 0, \"one word was not sent\");",
    "sends a full FIFO": "uint32_t w[8] = {1,2,3,4,5,6,7,8}; CHECK(send(w, 8) == 0, \"eight words were not sent\");",
    "sends twice in a row": "uint32_t w[2] = {1,2}; CHECK(send(w, 2) == 0, \"first\"); CHECK(send(w, 2) == 0, \"second\");",
}


def test_the_model_is_c_with_the_bus_a_driver_uses():
    code = the_model_in_c(_DMA)
    assert "uint32_t readl(uint32_t offset)" in code and "void writel(uint32_t value, uint32_t offset)" in code


def test_a_careful_driver_passes_and_keeps_every_rule():
    tried = tried_on_paper(_DMA, _CAREFUL, _SCENARIOS)
    assert tried.ok, tried.says()
    assert tried.passed == list(_SCENARIOS)


def test_a_driver_that_does_not_wait_starts_the_device_while_busy():
    hasty = _CAREFUL.replace("while (readl(0x04) & 1u) { }", "").replace("while (!dev_irq()) dev_tick();", "")
    # Its own answers are not looked at: the device's rules alone catch it.
    tried = tried_on_paper(_DMA, hasty, {"sends twice in a row": "uint32_t w[2] = {1,2}; send(w, 2); send(w, 2); CHECK(1, \"ran\");"})
    assert not tried.ok and any("while STATUS.BUSY was not 0" in b for b in tried.broke), tried.says()


def test_a_driver_that_leaves_the_interrupt_pending_is_caught():
    forgetful = _CAREFUL.replace("writel(0x1u, 0x08);", "")
    tried = tried_on_paper(_DMA, forgetful, {"sends one word": _SCENARIOS["sends one word"]})
    assert not tried.ok and any("interrupt was left pending" in b for b in tried.broke), tried.says()


def test_a_driver_that_overruns_its_own_buffer_is_caught_by_the_sanitizer():
    overrun = _CAREFUL.replace("if (n <= 0 || n > 8) return -1;", "")
    tried = tried_on_paper(_DMA, overrun, {"sends from a short buffer": "uint32_t w[2] = {1,2}; CHECK(send(w, 3) == 0, \"sent\");"})
    assert not tried.ok and (tried.sanitizer or tried.broke), tried.says()


def test_a_driver_that_does_not_compile_says_why():
    tried = tried_on_paper(_DMA, "static int send(void) { return undefined_thing; }", {"x": "CHECK(1, \"x\");"})
    assert not tried.ok and "undefined_thing" in tried.compile_error
