DEFAULTS = {
    "account":   "IscrB_ProAmmo",
    "partition": "boost_usr_prod",
    "time":      "1-00:00:00",
    "nodes":     1,
    "ntasks":    1,
    "cpus":      32,
    "gpus":      4,
    "mem":       128,
    "cap_array": 8,
}


def fill(
    cfg,
    step_name: str,
    job_name: str,
    workdir: str,
    add_params: dict = {},
) -> str:
    mapping = {
        "job_name": job_name,
        "workdir": workdir,
        "scripts_dir": str(cfg.launcher_root / "scripts"),
        **DEFAULTS,
    }

    mapping["cap_array-1"] = mapping["cap_array"] - 1

    if step_name in cfg["slurm"]:
        mapping = mapping | cfg["slurm"][step_name]

    params = cfg[step_name] | add_params if step_name in cfg._data else add_params
    params = "\\\n" + " \\\n".join([f"\t--{key} {value}" for key, value in params.items()]) if params else ""
    mapping["params"] = params
    
    out = (cfg.launcher_root / f"slurm_templates/{step_name}.sh").read_text()
    for key, value in mapping.items():
        out = out.replace(f"__{key.upper()}__", str(value))

    return out