"""Contracts for the release train — boring update, boring rollback.

Proven need (July 8): a dirty tree made `git pull --ff-only` fail silently
inside a script and the instance rebooted on the OLD tip while the operator
believed it updated. These contracts pin: dirty WIP is autostashed with a
label, a failed pull restores the stash and reports loudly, rollback only
targets ancestors of origin/main (so the NEXT update still fast-forwards),
and every action lands in the history ledger.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.release_train import ReleaseTrain

pytestmark = pytest.mark.unit


class FakeResult:
    def __init__(self, ok=True, stdout="", stderr="", returncode=None):
        self.ok = ok
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = 0 if ok else 1 if returncode is None else returncode
        if returncode is not None:
            self.returncode = returncode


class FakeGit:
    """Scriptable git: state dict drives responses; every call is recorded."""

    def __init__(self, *, head="aaa111", origin="bbb222", dirty=False,
                 pull_fails=False, ancestors=None):
        self.state = {
            "head": head, "origin": origin, "dirty": dirty,
            "pull_fails": pull_fails,
            "ancestors": set(ancestors or []),   # commits that are ancestors of origin/main
        }
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float) -> FakeResult:
        self.calls.append(tuple(command))
        argv = list(command)
        if argv[0] != "git":
            # compile / pytest sanity steps
            return FakeResult(stdout="ok")
        args = argv[3:]  # skip git -C <root>
        if args[:2] == ["rev-parse", "HEAD"]:
            return FakeResult(stdout=self.state["head"] + "\n")
        if args[:2] == ["rev-parse", "origin/main"]:
            return FakeResult(stdout=self.state["origin"] + "\n")
        if args[0] == "fetch":
            return FakeResult()
        if args[:2] == ["status", "--porcelain"]:
            return FakeResult(stdout=" M core/x.py\n" if self.state["dirty"] else "")
        if args[:2] == ["stash", "push"]:
            self.state["dirty"] = False
            self.state["stashed"] = args[3]
            return FakeResult()
        if args[:2] == ["stash", "pop"]:
            # Never issued any more: pop takes whatever is on top of a stack
            # other sessions push to. A test that sees it should fail.
            raise AssertionError("release_train must not `git stash pop` in a shared checkout")
        if args[:2] == ["stash", "list"]:
            if not self.state.get("stashed"):
                return FakeResult(stdout="")
            if args[2] == "--format=%H %gs":
                return FakeResult(stdout=f"deadbeef {self.state['stashed']}\n")
            return FakeResult(stdout=f"stash@{{0}} {self.state['stashed']}\n")
        if args[:2] == ["stash", "apply"]:
            assert args[2] == "deadbeef", "apply by the entry's own SHA, not by position"
            self.state["dirty"] = True
            return FakeResult()
        if args[:2] == ["stash", "drop"]:
            self.state["stashed"] = ""
            return FakeResult()
            return FakeResult()
        if args[0] == "pull":
            if self.state["pull_fails"]:
                return FakeResult(ok=False, stderr="error: Your local changes would be overwritten")
            self.state["head"] = self.state["origin"]
            return FakeResult(stdout="Fast-forward")
        if args[:2] == ["merge-base", "--is-ancestor"]:
            return FakeResult(returncode=0 if args[2] in self.state["ancestors"] else 1)
        if args[:2] == ["reset", "--hard"]:
            self.state["head"] = args[2]
            return FakeResult()
        return FakeResult(ok=False, stderr=f"unexpected git args: {args}")


def make_train(tmp_path: Path, git: FakeGit) -> ReleaseTrain:
    return ReleaseTrain(tmp_path, runner=git, history_path=tmp_path / "history.jsonl")


class TestUpdate:
    def test_clean_update_moves_head_and_records(self, tmp_path):
        git = FakeGit()
        train = make_train(tmp_path, git)
        outcome = train.update()
        assert outcome["ok"] is True and outcome["step"] == "complete"
        assert git.state["head"] == "bbb222"
        history = train.history()
        assert history[-1]["action"] == "update" and history[-1]["from"] == "aaa111"

    def test_already_current_is_a_noop(self, tmp_path):
        git = FakeGit(head="same", origin="same")
        outcome = make_train(tmp_path, git).update()
        assert outcome["step"] == "noop" and outcome["ok"] is True

    def test_dirty_tree_is_autostashed_and_given_back(self, tmp_path):
        """A successful update restores the WIP it set aside.

        It used to leave it in the stash list, labelled, and report the label:
        "autostashed" that stays stashed is uncommitted work made to disappear
        on a checkout where other sessions push and pop the same stack.
        """
        git = FakeGit(dirty=True)
        outcome = make_train(tmp_path, git).update()
        assert outcome["ok"] is True
        assert outcome["stash_restored"] is True
        assert outcome["stash_label"] == ""
        assert git.state["dirty"] is True, "the WIP is back in the working tree"
        assert git.state["stashed"] == "", "and its stash entry was dropped"

    def test_failed_pull_restores_stash_and_reports_loudly(self, tmp_path):
        git = FakeGit(dirty=True, pull_fails=True)
        train = make_train(tmp_path, git)
        outcome = train.update()
        assert outcome["ok"] is False and outcome["step"] == "pull"
        assert "overwritten" in outcome["error"]
        assert outcome["stash_restored"] is True
        assert git.state["dirty"] is True          # WIP back in the tree
        assert train.history()[-1]["ok"] is False  # failure is in the ledger too


class TestRollback:
    def _train_after_update(self, tmp_path):
        git = FakeGit(ancestors={"aaa111"})
        train = make_train(tmp_path, git)
        assert train.update()["ok"]
        return train, git

    def test_rolls_back_to_last_update_point(self, tmp_path):
        train, git = self._train_after_update(tmp_path)
        outcome = train.rollback()
        assert outcome["ok"] is True
        assert git.state["head"] == "aaa111"
        assert train.history()[-1]["action"] == "rollback"

    def test_refuses_non_ancestor_target(self, tmp_path):
        train, git = self._train_after_update(tmp_path)
        outcome = train.rollback(to="fff999")     # not in ancestors set
        assert outcome["ok"] is False and outcome["step"] == "guard"
        assert "fast-forward" in outcome["error"]
        assert git.state["head"] == "bbb222"      # untouched

    def test_no_recorded_point_and_no_target_fails_clearly(self, tmp_path):
        git = FakeGit()
        outcome = make_train(tmp_path, git).rollback()
        assert outcome["ok"] is False
        assert "no rollback point" in outcome["error"]

    def test_rollback_then_update_fast_forwards_again(self, tmp_path):
        """The coherence invariant: rollback never breaks the next update."""
        train, git = self._train_after_update(tmp_path)
        assert train.rollback()["ok"]
        outcome = train.update()
        assert outcome["ok"] is True and git.state["head"] == "bbb222"


class TestStatus:
    def test_status_shape(self, tmp_path):
        git = FakeGit()
        train = make_train(tmp_path, git)
        train.update()
        status = train.status()
        assert status["up_to_date"] is True
        assert status["rollback_point"] == "aaa111"
        assert len(status["history_tail"]) == 1

    def test_history_survives_corrupt_lines(self, tmp_path):
        git = FakeGit()
        train = make_train(tmp_path, git)
        train.update()
        with train.history_path.open("a", encoding="utf-8") as fh:
            fh.write("{broken\n")
        assert len(train.history()) == 1
        assert json.loads(train.history_path.read_text().splitlines()[0])["action"] == "update"
