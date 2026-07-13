"""
Builds the full chain of SLURM jobs for one active-learning cycle:
renders each step's job script from its template, writes it to jobs,
and registers it with job_submitter.
"""

from . import templating, slurm


STEPS = [
    "exploration",
    "selection",
    "plot_exploration",
    "labeling",
    "gather_xyz",
    "split_xyz",
    "mlip_fine-tuning",
]


DEPENDENCES = {
    "exploration":      ["previous_cycle"],
    "selection":        ["exploration"],
    "plot_exploration": ["selection"],
    "labeling":         ["selection"],
    "gather_xyz":       ["labeling"],
    "split_xyz":        ["gather_xyz"],
    "mlip_fine-tuning": ["split_xyz"],
}


INDEX = {
    "exploration":      "1",
    "selection":        "2",
    "plot_exploration": "2.a",
    "labeling":         "3",
    "gather_xyz":       "4",
    "split_xyz":        "5",
    "mlip_fine-tuning": "6",
}

ARRAY_STEPS = ["labeling"]


class CycleWriter():

    def __init__(self, cycle: int, cfg):
        self.cycle = cycle
        self.cfg = cfg
        self.jobs_dir = self.cfg.project_root / "jobs"
        self.workdir = f"../cycle-{self.cycle}/"
        self.config_path = self.jobs_dir / "js_config.txt"

    def _get_job_name(self, step_name):
        return f"al-c{self.cycle}-s{INDEX[step_name]}-{step_name}"

    def _get_script_path(self, step_name):
        return self.jobs_dir / f"{self._get_job_name(step_name)}.sh"

    def _get_deps(self, step_name):
        deps = []
        for dep in DEPENDENCES[step_name]:
            if dep == "previous_cycle":
                if self.cycle > 0:
                    deps.append(CycleWriter(self.cycle-1, self.cfg)._get_job_name("mlip_fine-tuning"))
            else:
                deps.append(self._get_job_name(dep))
        return deps

    def write_step(self, step_name: str, add_params: dict = {}):
        job_name = self._get_job_name(step_name)
        script_path = self._get_script_path(step_name)
        script_text = templating.fill(self.cfg, step_name, job_name, self.workdir, add_params)
        script_text += f"\ncd {self.jobs_dir}\npython {self.cfg.launcher_root}/al_launcher/job_submitter.py $SLURM_JOB_NAME\n"
        slurm.write_script(script_path, script_text)

        deps = self._get_deps(step_name)
        if deps:
            config = self.config_path.read_text().split('\n') if self.config_path.exists() else []
            config += [f"{','.join(deps)} -> {job_name}"]
            self.config_path.write_text('\n'.join(config))

    def write_cycle(self):
        (self.cfg.project_root / f"cycle-{self.cycle}/").mkdir(parents=True, exist_ok=True)

        for step_name in STEPS:
            if step_name == "exploration":
                if self.cycle == 0:
                    add_params = {"plumed": "../plumed-mace.dat"}
                else:
                    add_params = {"plumed":  "../plumed-franken.dat",
                                  "franken": f"../cycle-{self.cycle-1}/franken.pt"}
                self.write_step(step_name, add_params)
            else:
                self.write_step(step_name)


def write_cycle(cycle: int, cfg):
    CycleWriter(cycle, cfg).write_cycle()