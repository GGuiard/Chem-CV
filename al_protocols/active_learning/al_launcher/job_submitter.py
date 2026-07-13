from pathlib import Path
import re

from slurm import submit


DEPENDENCY = "DEPENDENCY"
IN_PROGRESS = "IN_PROGRESS"
COMPLETE = "COMPLETE"


class JobSubmitter():

    def __init__(self):
        self.config_path = Path("js_config.txt")
        self.states_path = Path("js_states.txt")

        self.config = self._load_config()
        self.states = self._init_states() | self._load_states()

    def _load_config(self):
        content = self.config_path.read_text()
        lines = content.split('\n')
        config = {}
        for line in lines:
            deps, job = line.split(" -> ")
            deps = deps.split(',')
            config[job] = deps
        return config

    def _init_states(self):
        states = {}
        for job, deps in self.config.items():
            for dep in deps:
                if dep not in states:
                    states[dep] = ""
            states[job] = DEPENDENCY
        return states

    def _load_states(self):
        states = {}
        if self.states_path.exists():
            content = self.states_path.read_text()
            lines = content.split('\n')
            for line in lines:
                job, state = line.split(" : ")
                states[job] = state
        return states

    def _update_state(self, job: str):
        if '/' in self.states[job]:
            index, total = map(int, self.states[job].split(' ')[1].split('/'))
            index += 1
            if index == total:
                self.states[job] = COMPLETE
            else:
                self.states[job] = IN_PROGRESS + f" {index}/{total}"
        else:
            self.states[job] = COMPLETE

    def _submit_job(self, job: str):
        _ = submit(f"{job}.sh")
        content = Path(f"{job}.sh").read_text()
        match = re.search(r"#SBATCH --array=0-(\d*)", content)
        if match:
            total = int(match.group(1)) + 1
            self.states[job] = IN_PROGRESS + f" 0/{total}"
        else:
            self.states[job] = IN_PROGRESS

    def _update_states(self):
        content = '\n'.join([f"{job} : {state}" for job, state in self.states.items()])
        self.states_path.write_text(content)

    def update(self, job: str):
        self._update_state(job)
        for job, state in self.states.items():
            if state == DEPENDENCY and all(self.states[dep] == COMPLETE for dep in self.config[job]):
                self._submit_job(job)            
        self._update_states()
                

if __name__ == "__main__":
    import sys
    job = sys.argv[1]
    JobSubmitter().update(job)