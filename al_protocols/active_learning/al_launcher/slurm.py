import subprocess
from pathlib import Path


def write_script(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


def submit(script_path: Path, project_path: Path = Path("."), dry_run: bool = False) -> str:
    cmd = ["sbatch", str(script_path)]
    if dry_run:
        print(' '.join(cmd))
        return 
    else:
        result = subprocess.run(cmd, cwd=str(project_path), capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        return job_id
