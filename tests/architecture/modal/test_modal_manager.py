from core.environment.modal import ModalManager, ModalState


def test_modal_state_blocks_normal_policy_and_resolves_safe_default():
    modal = ModalState(
        kind="confirmation",
        text="Overwrite file?",
        legal_responses={"y", "n"},
        safe_default="n",
        dangerous_responses={"y"},
    )
    manager = ModalManager()
    assert manager.should_block_normal_policy(modal)
    assert manager.resolve(modal) == "n"


def test_unknown_modal_degrades_closed():
    modal = ModalState(kind="unknown", text="???", legal_responses=set())
    assert ModalManager().resolve(modal) is None
