"""Legacy compatibility wrapper for the canonical core train-self skill."""

from pathlib import Path

from core.skills.train_self import TrainSelfSkill as _CoreTrainSelfSkill


class TrainSelfSkill(_CoreTrainSelfSkill):
    """Preserve the legacy ``workspace_root`` constructor without forking logic."""

    def __init__(self, workspace_root: str = "."):
        super().__init__()
        if workspace_root and workspace_root != ".":
            self.dataset_path = Path(workspace_root) / "data" / "training" / "dataset.jsonl"
            get_task_tracker().create_task(get_storage_gateway().create_dir(self.dataset_path.parent, cause='TrainSelfSkill.__init__'))


__all__ = ["TrainSelfSkill"]
