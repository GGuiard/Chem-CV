"""
Load and give convenient access to config.yaml.
"""

from pathlib import Path
import yaml


class Config:
    def __init__(self, project_root: str | Path):
        self.launcher_root = Path(__file__).parent.parent.resolve()

        self.project_root = Path(project_root).resolve()
        self.config_path = self.project_root / "config.yaml"
        with open(self.config_path) as f:
            data = yaml.safe_load(f)
        self._data = data

        self._data["slurm"]["labeling"]["n_array"] = self._data["selection"]["target"]

    def __getitem__(self, key: str):
        return self._data[key]