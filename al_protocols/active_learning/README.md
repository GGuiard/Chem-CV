# Active Learning for MLIPs

Automated active-learning loop for Machine-Learning Interatomic Potentials using
OPES-biased MD, DEAL configuration selection, ORCA DFT labelling, and Franken
fine-tuning.

## Project layout

```
.
├── config.yaml              ← all parameters (edit this)
├── config_loader.py         ← shared config utility (don't edit)
├── submit_cycle.py          ← master orchestrator: submit one cycle
│
├── 1-md.py                  ← step 1: biased MD (Bussi + OPES)
├── 2-deal.py                ← step 2: DEAL configuration selection
├── 3-dft-single.py          ← step 3a: ORCA on one structure (called by array)
├── 3-dft-gather.py          ← step 3b: collect per-structure results → dft.xyz
├── 4-franken.py             ← step 4: Franken fine-tuning + parity plot
├── plot-md_deal.py          ← phase-space plot (run inside step 2 job)
│
├── slurm/
│   ├── job_md.sh            ← SLURM template for step 1
│   ├── job_deal.sh          ← SLURM template for step 2
│   ├── job_dft_array.sh     ← SLURM array template for step 3
│   ├── job_dft_gather.sh    ← SLURM template for step 3b gather
│   └── job_franken.sh       ← SLURM template for step 4
│
├── logs/                    ← SLURM stdout/stderr (auto-created)
├── init.xyz                 ← initial structure (you provide this)
├── plumed-mace.dat          ← PLUMED input for mace_mp (you provide this)
└── plumed-franken.dat       ← PLUMED input for Franken model (you provide this)
```

Output per cycle (inside `cycle-N/`):

```
cycle-0/
├── 1-md/
│   ├── traj_comp.traj
│   ├── COLVAR               ← written by PLUMED
│   └── franken.pt           ← model copied from previous cycle (cycle > 0)
├── 2-deal/
│   └── deal_selected.xyz
├── 3-dft/
│   ├── dft_single/          ← per-structure ORCA outputs
│   └── dft.xyz              ← gathered results
├── 4-franken/
│   ├── train.xyz
│   ├── test.xyz
│   ├── franken.pt           ← new fine-tuned model
│   └── franken_parity.png
└── md_deal.png              ← phase-space + DEAL selection plot
```

---

## Quick start

### 1. Edit `config.yaml`

Set your SLURM account, partition, time limits, MD parameters, DFT charge/mult, etc.

### 2. Submit a cycle

```bash
# Cycle 0 (no prior model — uses mace_mp)
python submit_cycle.py --cycle 0

# Cycle 1 (uses franken.pt from cycle 0)
python submit_cycle.py --cycle 1

# Dry run — see what would be submitted without actually doing it
python submit_cycle.py --cycle 0 --dry-run
```

### 3. Restart after a failure

If a step fails, fix the issue and restart from that step:

```bash
python submit_cycle.py --cycle 2 --start-from dft
```

Available values for `--start-from`: `md`, `deal`, `dft`, `franken`.

### 4. Chain cycles automatically

The `--dep` flag makes step 1 of cycle N depend on a specific job ID,
which lets you queue several cycles at once:

```bash
python submit_cycle.py --cycle 0
# → final job ID printed, e.g. 123456

python submit_cycle.py --cycle 1 --dep 123456
```

---

## Running individual steps manually

Each script has a `--help` flag:

```bash
python 1-md.py --help
python 2-deal.py --help
python 3-dft-single.py --help
python 3-dft-gather.py --help
python 4-franken.py --help
python plot-md_deal.py --help
```

Example — rerun DFT for one structure:

```bash
python 3-dft-single.py \
    --index 42 \
    --input cycle-2/2-deal/deal_selected.xyz \
    --outdir cycle-2/3-dft/dft_single \
    --nprocs 32
```

---

## Environment notes

| Step         | Conda env  | Notes                                      |
|--------------|------------|--------------------------------------------|
| 1-md         | `md_env`   | ASE, MACE, Franken, PLUMED                 |
| 2-deal       | `deal`     | separate env required by DEAL              |
| plot         | `md_env`   | plumed Python, mlcolvar                    |
| 3-dft        | `md_env`   | ASE, ORCA, orca_parser                     |
| 4-franken    | `md_env`   | Franken, PyTorch                           |

The `deal` environment is activated only for step 2, automatically handled
by `slurm/job_deal.sh`.
