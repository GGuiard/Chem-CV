"""
Command-line entrypoint: submit N active-learning cycles as one chained
sequence of SLURM jobs.
"""

from .config import Config
from .steps import write_cycle


def main(project_root: str) -> None:
    cfg = Config(project_root)
    n_cycles = cfg["protocol"]["n_cycles"]
    for cycle in range(n_cycles):
        write_cycle(cycle, cfg)