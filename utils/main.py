import os
os.chdir("SN2/OPES")

from postprocessing import postprocessing

cv = {
    "dd": {
        "label": r"$d_{C-Cl_1} - d_{C-Cl_2}$  [Å]",
        "bounds": (-2.5, 2.5),
    },
    "d1": {
        "label": r"$d_{C-Cl_1}$  [Å]",
        "bounds": (1.5, 4.0),
    },
    "d2": {
        "label": r"$d_{C-Cl_2}$  [Å]",
        "bounds": (1.5, 4.0),
    },
}

postprocessing(
    # --- CV selection ---
    cv=cv,
    cv_1d=["dd", "d1", "d2"],
    cv_2d=[("d1", "d2")],

    # --- Trajectory options ---
    time_unit="ps",
    color_with_time=True,

    # --- KDE parameters ---
    num_samples=200,
    bandwidth=0.01,
    nb_levels=11,
    smooth_2d=True,

    # --- FES options ---
    temperature=300.0,
    transient=0.0,
    blocks=3,
    fes_units="eV",

    # --- Display limits ---
    fes_max_1d=1.2,
    density_min_2d=0.1,
    fes_max_2d=0.8,
    error_min_2d=0.005,
    error_max_2d=0.05,

    # --- Layout ---
    symmetric=True,
    save=True,
    show=True,
)
