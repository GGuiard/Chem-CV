import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from chemiscope import write_input, all_atomic_environments
# from ase.data import covalent_radii, chemical_symbols

def trj_E(Emec, av, std):
    fig, ax = plt.subplots(layout='tight')
    ax.axhspan(av-std, av+std, color='grey', alpha=0.3)
    ax.plot(Emec)
    ax.axhline(av, color='k', linestyle='--')

    ax.set_xlabel("number of frame")
    ax.set_ylabel("E [eV]")

    fig.savefig("trj_E.svg")
    plt.close()

def trj_T(T, av, std):
    fig, ax = plt.subplots(layout='tight')
    ax.axhspan(av-std, av+std, color='grey', alpha=0.3)
    ax.plot(T)
    ax.axhline(av, color='k', linestyle='--')

    ax.set_xlabel("number of frame")
    ax.set_ylabel("T [K]")

    fig.savefig("trj_T.svg")
    plt.close()

def trj_dd(time, dd):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, dd)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$d_{C-Cl_1} - d_{C-Cl_2}\ [A]$")

    fig.savefig("trj_dd.svg")
    plt.close()

def trj_d1(time, d1):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, d1)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$d_{C-Cl_1}\ [A]$")

    fig.savefig("trj_d1.svg")
    plt.close()

def trj_d2(time, d2):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, d2)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$d_{C-Cl_2}\ [A]$")

    fig.savefig("trj_d2.svg")
    plt.close()

def trj_2D(time, d1, d2):
    fig, ax = plt.subplots(layout='tight')
    scatter = ax.scatter(d1, d2, s=1/100, c=time, cmap="magma")

    ax.set_xlabel(r"$d_{C-Cl_1}\ [A]$")
    ax.set_ylabel(r"$d_{C-Cl_2}\ [A]$")
    ax.set_xlim(1.5, 4)
    ax.set_ylim(1.5, 4)
    ax.set_aspect('equal', 'box')

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(label="t [ps]")

    fig.savefig("trj_2D.png", dpi=300)
    plt.close()

def trj_rct(time, rct):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, rct)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel("OPES rct")

    fig.savefig("trj_rct.svg")
    plt.close()

def trj_zed(time, zed):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, zed)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel("OPES zed")

    fig.savefig("trj_zed.svg")
    plt.close()

def trj_n(time, neff, nker):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, neff, label=r"$n_{eff}$")
    ax.plot(time, nker, label=r"$n_{ker}$")

    ax.set_xlabel("t [ps]")
    ax.set_ylabel("N")
    ax.legend()

    fig.savefig("trj_n.svg")
    plt.close()

def density_dd(grid, density):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(grid, density)

    ax.set_xlabel(r"$d_{C-Cl_1} - d_{C-Cl_2}\ [A]$")
    ax.set_ylabel("Density")
    ax.set_xlim(-2.5, 2.5)
    
    fig.savefig("density_dd.svg")
    plt.close()

def density_d1(grid, density):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(grid, density)

    ax.set_xlabel(r"$d_{C-Cl_1}\ [A]$")
    ax.set_ylabel("Density")
    ax.set_xlim(1.5, 4)
    
    fig.savefig("density_d1.svg")
    plt.close()

def density_d2(grid, density):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(grid, density)

    ax.set_xlabel(r"$d_{C-Cl_2}\ [A]$")
    ax.set_ylabel("Density")
    ax.set_xlim(1.5, 4)
    
    fig.savefig("density_d2.svg")
    plt.close()

def density_2D(grid_d, density):
    fig, ax = plt.subplots(layout='tight')
    im = ax.contourf(grid_d, grid_d, density.T, 10, cmap=colormaps['Blues'])
    ax.contour(grid_d, grid_d, density.T, 10, linestyles='-', colors='darkgray', linewidths=1.2)

    ax.set_xlabel(r"$d_{C-Cl_1}\ [A]$")
    ax.set_ylabel(r"$d_{C-Cl_2}\ [A]$")
    ax.set_xlim(1.5, 4)
    ax.set_ylim(1.5, 4)
    ax.set_aspect('equal', 'box')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label="Density")

    fig.savefig("density_2D.svg")
    plt.close()

def fes_dd(grid, fes, err):
    fig, ax = plt.subplots(layout='tight')
    ax.fill_between(grid, fes-err, fes+err, alpha=0.3)
    ax.plot(grid, fes)

    ax.set_xlabel(r"$d_{C-Cl_1} - d_{C-Cl_2}\ [A]$")
    ax.set_ylabel("FES [eV]")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 1)
    
    fig.savefig("fes_dd.svg")
    plt.close()

def fes_d1(grid, fes, err):
    fig, ax = plt.subplots(layout='tight')
    ax.fill_between(grid, fes-err, fes+err, alpha=0.3)
    ax.plot(grid, fes)

    ax.set_xlabel(r"$d_{C-Cl_1}\ [A]$")
    ax.set_ylabel("FES [eV]")
    ax.set_xlim(1.5, 4)
    ax.set_ylim(0, 1)
    
    fig.savefig("fes_d1.svg")
    plt.close()

def fes_d2(grid, fes, err):
    fig, ax = plt.subplots(layout='tight')
    ax.fill_between(grid, fes-err, fes+err, alpha=0.3)
    ax.plot(grid, fes)

    ax.set_xlabel(r"$d_{C-Cl_2}\ [A]$")
    ax.set_ylabel("FES [eV]")
    ax.set_xlim(1.5, 4)
    ax.set_ylim(0, 1)
    
    fig.savefig("fes_d2.svg")
    plt.close()

def fes_2D(grid_d, fes):
    fig, ax = plt.subplots(layout='tight')
    im = ax.contourf(grid_d, grid_d, fes.T, np.linspace(0, 1, 11), cmap=colormaps['Blues_r'])
    ax.contour(grid_d, grid_d, fes.T, np.linspace(0, 1, 11), linestyles='-', colors='darkgray', linewidths=1.2)

    ax.set_xlabel(r"$d_{C-Cl_1}\ [A]$")
    ax.set_ylabel(r"$d_{C-Cl_2}\ [A]$")
    ax.set_xlim(1.5, 4)
    ax.set_ylim(1.5, 4)
    ax.set_aspect('equal', 'box')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label="FES [eV]")

    fig.savefig("fes_2D.svg")
    plt.close()

def err_fes_2D(grid_d, err):
    fig, ax = plt.subplots(layout='tight')
    im = ax.contourf(grid_d, grid_d, err.T, np.linspace(0, 0.02, 11), cmap=colormaps['Blues_r'])
    ax.contour(grid_d, grid_d, err.T, np.linspace(0, 0.02, 11), linestyles='-', colors='darkgray', linewidths=1.2)

    ax.set_xlabel(r"$d_{C-Cl_1}\ [A]$")
    ax.set_ylabel(r"$d_{C-Cl_2}\ [A]$")
    ax.set_xlim(1.5, 4)
    ax.set_ylim(1.5, 4)
    ax.set_aspect('equal', 'box')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label="Err(FES) [eV]")

    fig.savefig("err_fer_2D.svg")
    plt.close()

def chemiscope(structures, time, d1, d2):
    properties = {"d1": {"target": "structure",
                        "values": d1,
                        "description": "Distance between the carbon atom and the first chlorin atom"},
                  "d2": {"target": "structure",
                        "values": d2,
                        "description": "Coordination between the carbon atom and the second chlorin atom"},
                  "time": {"target": "structure",
                           "values": time,
                           "description": "time [ps]"}}
    
    # atom_radius = []
    # for atoms in structures:
    #     for atom in atoms:
    #         atom_radius.append({"radius": covalent_radii[chemical_symbols.index(atom.symbol)]})
    # shapes = {"selection": {"kind": "sphere", "parameters": {"atom": atom_radius}}}
    # "shape": "selection"
    # shapes=shapes

    settings = {"target": "structure",
                "map": {"x": {"property": "d1"},
                        "y": {"property": "d2"},
                        "color": {"property": "time"}},
                "structure": [{"bonds": True,
                               "spaceFilling": False,
                               "keepOrientation": True,
                               "playbackDelay": 200}]}
    
    write_input("chemiscope.json.gz", structures=structures, properties=properties, settings=settings)

def chemiscope_chemcv(structures, time, d1, d2, chemcv, color):
    properties = {"d1": {"target": "structure",
                        "values": d1,
                        "description": "Distance between the carbon atom and the first chlorin atom"},
                  "d2": {"target": "structure",
                        "values": d2,
                        "description": "Coordination between the carbon atom and the second chlorin atom"},
                  "time": {"target": "structure",
                           "values": time,
                           "description": "time [ps]"},
                  "color": {"target": "atom",
                        "values": color.ravel(),
                        "description": "charge [e]"}}
    
    for name, value in chemcv.items():
        properties[name] = {"target": "structure", "values": value}

    settings = {"target": "structure",
                "map": {"x": {"property": "d1"},
                        "y": {"property": "d2"},
                        "color": {"property": "time"}},
                "structure": [{"bonds": True,
                               "spaceFilling": False,
                               "keepOrientation": True,
                               "playbackDelay": 200,
                               "atomLabels": True,
                               "labelsProperty": "color",
                               "color": {"property": "color", "palette": "bwr", "min":-1, "max":1}}]}

    environments = all_atomic_environments(structures)

    write_input("chemiscope_chemcv.json.gz", structures=structures, properties=properties, environments=environments, settings=settings)

def pred(ref, pred):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color='k', linestyle='--')
    ax.plot(ref, pred, 'o', ms=1)
    
    ax.set_xlabel("Reference")
    ax.set_ylabel("Prediction")

    fig.savefig("pred.svg")
    plt.close()