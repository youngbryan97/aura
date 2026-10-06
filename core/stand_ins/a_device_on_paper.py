"""A device known from its datasheet, standing in for the one that is not here, so a driver for it can be run.

A driver for a board that is not on the desk cannot be run against the board.
What its makers do before the silicon exists is run it against a model of the
device built from the same register map the driver is written from: so does
this. The datasheet's facts become a model in C: each register with its
offset, its reset value and its access (read-write, read-only, write-only,
write-1-to-clear), FIFOs with their depth, and what the device does when a
register is written ("START set: BUSY now; DONE and an interrupt twenty cycles
later"), with what it requires first ("not while BUSY"). The driver is
compiled with it, against a two-call bus (``readl``, ``writel``), with
address and undefined-behaviour sanitizers, and run through the scenarios a
person would try. Every read is a cycle of the device's clock, so a driver
that polls lets time pass, and one that does not wait is caught.

What it reports: each scenario that failed, every rule of the device the
driver broke (a write while BUSY, a read of an empty FIFO, a write to a
read-only register, an interrupt left pending), and anything the sanitizers
found. Nothing here knows any device: the device is the spec it is given.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["DeviceSpec", "Tried", "the_model_in_c", "tried_on_paper"]


@dataclass
class DeviceSpec:
    """A device as its datasheet gives it.

    ``registers``: [{"name", "offset", "access": "rw" | "ro" | "wo" | "w1c", "reset": 0,
                     "fields": {"NAME": [lowest bit, width]}, "fifo": {"depth": n}}]
    ``behaviour``: [{"on_write": "CTRL.START", "equals": 1,
                     "requires": [["STATUS.BUSY", 0], ...],      (else the driver broke a rule)
                     "now": [["STATUS.BUSY", 1], ...],
                     "after_cycles": 20,
                     "later": [["STATUS.BUSY", 0], ["STATUS.DONE", 1], ["IRQ.DONE", 1]],
                     "interrupt_if": "CTRL.IRQ_EN",                (raised at "later", when that bit is set)
                     "fifo_in": {"from": "TX", "count": "LEN"}}]   (the device takes LEN words from a FIFO)
    """

    name: str
    registers: list[dict[str, Any]]
    behaviour: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Tried:
    ok: bool
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    broke: list[str] = field(default_factory=list)
    sanitizer: str = ""
    compile_error: str = ""

    def says(self) -> str:
        if self.compile_error:
            return f"the driver does not compile against the device: {self.compile_error[:400]}"
        if self.ok:
            return f"every scenario passed ({', '.join(self.passed)}) and the driver kept every rule of the device"
        parts = []
        if self.failed:
            parts.append("failed: " + "; ".join(self.failed[:5]))
        if self.broke:
            parts.append("broke the device's rules: " + "; ".join(sorted(set(self.broke))[:5]))
        if self.sanitizer:
            parts.append("the sanitizer found: " + self.sanitizer[:300])
        return ". ".join(parts)


_C_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _place(spec: DeviceSpec, said: str) -> tuple[int, int, int]:
    """A register's index, and a field's lowest bit and mask, from "REG" or "REG.FIELD"."""
    reg_name, _, field_name = said.partition(".")
    for index, reg in enumerate(spec.registers):
        if reg["name"] == reg_name:
            if not field_name:
                return index, 0, 0xFFFFFFFF
            low, width = reg.get("fields", {})[field_name]
            return index, low, ((1 << width) - 1) << low
    raise KeyError(f"no register {said!r} on {spec.name}")


def _set(spec: DeviceSpec, said: str, value: int) -> str:
    index, low, mask = _place(spec, said)
    return f"regs[{index}] = (regs[{index}] & ~0x{mask:08X}u) | ((({value}u) << {low}) & 0x{mask:08X}u);"


def _get(spec: DeviceSpec, said: str) -> str:
    index, low, mask = _place(spec, said)
    return f"((regs[{index}] & 0x{mask:08X}u) >> {low})"


def the_model_in_c(spec: DeviceSpec) -> str:
    """The device as C: its registers, FIFOs and behaviour, the bus a driver uses, and the record of the rules broken."""
    for reg in spec.registers:
        if not _C_NAME.match(reg["name"]):
            raise ValueError(f"register name {reg['name']!r} is not a name")
    regs = spec.registers
    lines = [
        "#include <stdint.h>", "#include <stdio.h>", "#include <string.h>",
        f"/* A model of {spec.name}, from its datasheet. */",
        f"static uint32_t regs[{len(regs)}];",
        "static int broke_count; static uint64_t cycle;",
        "static void broke(const char *what) { broke_count++; fprintf(stdout, \"BROKE %llu %s\\n\", (unsigned long long)cycle, what); }",
        "static int irq_line;",
        "struct pending { int live; uint64_t at; int which; };",
        "static struct pending pending[16];",
    ]
    for reg in regs:
        if "fifo" in reg:
            depth = int(reg["fifo"].get("depth", 16))
            lines += [f"static uint32_t fifo_{reg['name']}[{depth}]; static int fifo_{reg['name']}_n;"]
    reset = " ".join(f"regs[{i}] = {int(reg.get('reset', 0))}u;" for i, reg in enumerate(regs))
    fifo_reset = " ".join(f"fifo_{reg['name']}_n = 0;" for reg in regs if "fifo" in reg)
    lines.append(f"void dev_reset(void) {{ {reset} {fifo_reset} memset(pending, 0, sizeof pending); irq_line = 0; broke_count = 0; cycle = 0; }}")
    # What happens later, by behaviour number.
    later_cases = []
    for n, rule in enumerate(spec.behaviour):
        body = " ".join(_set(spec, place, value) for place, value in rule.get("later", []))
        if rule.get("interrupt_if"):
            body += f" if ({_get(spec, rule['interrupt_if'])}) irq_line = 1;"
        later_cases.append(f"case {n}: {body} break;")
    lines.append("static void happen(int which) { switch (which) { " + " ".join(later_cases) + " default: break; } }")
    lines.append("void dev_tick(void) { cycle++; for (int i = 0; i < 16; i++) if (pending[i].live && pending[i].at <= cycle) { pending[i].live = 0; happen(pending[i].which); } }")
    lines.append("static void later(int which, uint64_t after) { for (int i = 0; i < 16; i++) if (!pending[i].live) { pending[i].live = 1; pending[i].at = cycle + after; pending[i].which = which; return; } broke(\"the device was asked to do more at once than it can\"); }")
    lines.append("int dev_irq(void) { return irq_line; }")
    # Reads.
    reads = []
    for index, reg in enumerate(regs):
        offset, name = int(reg["offset"]), reg["name"]
        if "fifo" in reg:
            reads.append(f"case 0x{offset:X}: if (fifo_{name}_n == 0) {{ broke(\"read {name} while its FIFO was empty\"); return 0; }} "
                         f"{{ uint32_t v = fifo_{name}[0]; memmove(fifo_{name}, fifo_{name} + 1, (size_t)(--fifo_{name}_n) * sizeof(uint32_t)); return v; }}")
        elif reg.get("access") == "wo":
            reads.append(f"case 0x{offset:X}: broke(\"read {name}, which can only be written\"); return 0;")
        else:
            reads.append(f"case 0x{offset:X}: return regs[{index}];")
    lines.append("uint32_t readl(uint32_t offset) { dev_tick(); switch (offset) { " + " ".join(reads)
                 + " default: { char m[64]; snprintf(m, sizeof m, \"read 0x%X, where there is no register\", offset); broke(m); return 0; } } }")
    # Writes, and what they set off.
    writes = []
    for index, reg in enumerate(regs):
        offset, name, access = int(reg["offset"]), reg["name"], reg.get("access", "rw")
        if "fifo" in reg:
            depth = int(reg["fifo"].get("depth", 16))
            body = (f"if (fifo_{name}_n >= {depth}) {{ broke(\"wrote {name} while its FIFO was full\"); break; }} "
                    f"fifo_{name}[fifo_{name}_n++] = value;")
        elif access == "ro":
            body = f"broke(\"wrote {name}, which can only be read\");"
        elif access == "w1c":
            body = f"regs[{index}] &= ~value; if (regs[{index}] == 0) irq_line = 0;"
        else:
            body = f"regs[{index}] = value;"
        for n, rule in enumerate(spec.behaviour):
            if rule["on_write"].partition(".")[0] != name:
                continue
            trigger = f"{_get(spec, rule['on_write'])} == {int(rule.get('equals', 1))}u"
            checks = " ".join(
                f"if (!({_get(spec, place)} == {int(value)}u)) broke(\"wrote {rule['on_write']} while {place} was not {int(value)}\");"
                for place, value in rule.get("requires", []))
            if rule.get("requires_nonzero"):
                checks += " " + " ".join(f"if ({_get(spec, place)} == 0u) broke(\"wrote {rule['on_write']} while {place} was 0\");"
                                         for place in rule["requires_nonzero"])
            now = " ".join(_set(spec, place, value) for place, value in rule.get("now", []))
            taken = ""
            if rule.get("fifo_in"):
                source, count = rule["fifo_in"]["from"], rule["fifo_in"]["count"]
                taken = (f"{{ uint32_t want = {_get(spec, count)}; if (want > (uint32_t)fifo_{source}_n) broke(\"started with fewer words in {source} than {count} says\"); "
                         f"else {{ memmove(fifo_{source}, fifo_{source} + want, (size_t)(fifo_{source}_n - (int)want) * sizeof(uint32_t)); fifo_{source}_n -= (int)want; }} }}")
            body += f" if ({trigger}) {{ {checks} {now} {taken} later({n}, {int(rule.get('after_cycles', 1))}); }}"
        writes.append(f"case 0x{offset:X}: {body} break;")
    lines.append("void writel(uint32_t value, uint32_t offset) { dev_tick(); switch (offset) { " + " ".join(writes)
                 + " default: { char m[64]; snprintf(m, sizeof m, \"wrote 0x%X, where there is no register\", offset); broke(m); } } }")
    lines.append("int dev_broke_count(void) { return broke_count; }")
    lines.append("int dev_interrupt_pending(void) { return irq_line; }")
    return "\n".join(lines) + "\n"


_HARNESS = r"""
#include <stdio.h>
#include <stdint.h>
void dev_reset(void); int dev_broke_count(void); int dev_interrupt_pending(void); int dev_irq(void); void dev_tick(void);
uint32_t readl(uint32_t offset); void writel(uint32_t value, uint32_t offset);
static int failures;
#define CHECK(cond, what) do { if (!(cond)) { printf("FAILED %%s: %%s\n", current, what); failures++; return; } } while (0)
static const char *current = "";
%(driver)s
%(scenarios)s
int main(void) {
%(calls)s
  return failures ? 1 : 0;
}
"""


def tried_on_paper(spec: DeviceSpec, driver: str, scenarios: dict[str, str], *, timeout_s: float = 60.0) -> Tried:
    """``driver`` (C, using readl/writel/dev_irq/dev_tick) run through ``scenarios`` (name: C body using CHECK) on the device ``spec`` describes."""
    compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if compiler is None:
        return Tried(False, compile_error="no C compiler on this machine")
    functions, calls = [], []
    for n, (name, body) in enumerate(scenarios.items()):
        functions.append(f"static void scenario_{n}(void) {{\n{body}\n}}")
        quoted = name.replace("\\", "\\\\").replace('"', '\\"')
        calls.append(f'  current = "{quoted}"; dev_reset(); {{ int before = failures; scenario_{n}(); '
                     f'if (dev_interrupt_pending()) {{ printf("BROKE end an interrupt was left pending\\n"); }} '
                     f'if (failures == before) printf("PASSED %s\\n", current); }}')
    with tempfile.TemporaryDirectory(prefix="aura-device-") as scratch:
        folder = Path(scratch)
        (folder / "device.c").write_text(the_model_in_c(spec), "utf-8")
        (folder / "test.c").write_text(_HARNESS % {"driver": driver, "scenarios": "\n".join(functions), "calls": "\n".join(calls)}, "utf-8")
        built = subprocess.run([compiler, "-std=c11", "-g", "-O1", "-Wall", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                                "-o", str(folder / "tried"), str(folder / "device.c"), str(folder / "test.c")],
                               capture_output=True, text=True, timeout=timeout_s)
        if built.returncode != 0:
            return Tried(False, compile_error=(built.stderr or built.stdout).strip())
        try:
            ran = subprocess.run([str(folder / "tried")], capture_output=True, text=True, timeout=timeout_s, cwd=scratch,
                                 env={"ASAN_OPTIONS": "detect_leaks=0:abort_on_error=0", "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1"})
        except subprocess.TimeoutExpired:
            return Tried(False, failed=[f"the driver did not finish within {timeout_s:g}s: it waits for something the device never does"])
    out = ran.stdout.splitlines()
    passed = [line[7:] for line in out if line.startswith("PASSED ")]
    failed = [line[7:] for line in out if line.startswith("FAILED ")]
    broke = [re.sub(r"^BROKE \S+ ", "", line) for line in out if line.startswith("BROKE ")]
    found = ""
    if "AddressSanitizer" in ran.stderr or "runtime error:" in ran.stderr:
        found = next((line.strip() for line in ran.stderr.splitlines() if "ERROR: AddressSanitizer" in line or "runtime error:" in line), ran.stderr[:300])
    ok = not failed and not broke and not found and ran.returncode == 0 and len(passed) == len(scenarios)
    return Tried(ok, passed, failed, broke, found)
