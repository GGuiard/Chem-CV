from utils import postprocessing

cv = {
    "dd": {
        "label": "Geometric CV [Å]",
        "bounds": (-3.5, 3.5),
    },
    "d1": {
        "label": r"$d_1$  [Å]",
        "bounds": (1.5, 6.0),
    },
    "d2": {
        "label": r"$d_2$  [Å]",
        "bounds": (1.5, 6.0),
    },
    "chemcv.node-0": {
        "label": "Mayer bond order",
        "bounds": (-1, 1),
    }
}

postprocessing.postprocessing(
    # --- CV selection ---
    cv=cv,
    cv_1d=["dd", "chemcv.node-0"],
    cv_2d=[("d1", "d2")],

    # --- FES ---
    transient=200,
    fes_max=1.,

    # --- Chemiscope ---
    remove_com=True,

    # --- General options ---
    directory=f"MD_RUN_METAD_Franken/b_Mayer",
    symmetric=True,
    show=False,
)