import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from chemiscope import write_input, all_atomic_environments
from ase.data import covalent_radii, chemical_symbols

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

def chemiscope(structures, time, d1, d2, adapt_radius: bool = True):
    properties = {"d1": {"target": "structure",
                        "values": d1,
                        "description": "Distance between the carbon atom and the first chlorin atom"},
                  "d2": {"target": "structure",
                        "values": d2,
                        "description": "Coordination between the carbon atom and the second chlorin atom"},
                  "time": {"target": "structure",
                           "values": time,
                           "description": "time [ps]"}}
    
    settings = {"target": "structure",
                "map": {"x": {"property": "d1"},
                        "y": {"property": "d2"},
                        "color": {"property": "time"}},
                "structure": [{"bonds": True,
                               "spaceFilling": False,
                               "keepOrientation": True,
                               "playbackDelay": 200}]}
    
    if adapt_radius:
        atom_radius = []
        for atoms in structures:
            for atom in atoms:
                atom_radius.append({"radius": covalent_radii[chemical_symbols.index(atom.symbol)]})
        shapes = {"selection": {"kind": "sphere", "parameters": {"atom": atom_radius}}}
        settings["shape"] = "selection"
    else:
        shapes = {}
    
    write_input("chemiscope.json.gz", structures=structures, properties=properties, shapes=shapes, settings=settings)

def chemiscope_chemcv(structures, time, d1, d2, chemcv, color = None, adapt_radius: bool = False):
    properties = {"d1": {"target": "structure",
                        "values": d1,
                        "description": "Distance between the carbon atom and the first chlorin atom"},
                  "d2": {"target": "structure",
                        "values": d2,
                        "description": "Coordination between the carbon atom and the second chlorin atom"},
                  "time": {"target": "structure",
                           "values": time,
                           "description": "time [ps]"}}
    
    for name, value in chemcv.items():
        properties[name] = {"target": "structure", "values": value}

    settings = {"target": "structure",
                "map": {"x": {"property": "d1"},
                        "y": {"property": "d2"},
                        "color": {"property": "time"}},
                "structure": [{"bonds": True,
                               "spaceFilling": False,
                               "keepOrientation": True,
                               "playbackDelay": 200}]}

    if color:
        properties["color"] = {"target": "atom",
                               "values": color.ravel(),
                               "description": "charge [e]"}
        settings["structure"][0] | {"atomLabels": True,
                                    "labelsProperty": "color",
                                    "color": {"property": "color", "palette": "bwr", "min":-1, "max":1}}
        
    if adapt_radius:
        atom_radius = []
        for atoms in structures:
            for atom in atoms:
                atom_radius.append({"radius": covalent_radii[chemical_symbols.index(atom.symbol)]})
        shapes = {"selection": {"kind": "sphere", "parameters": {"atom": atom_radius}}}
        settings["shapes"] = "selection"
    else:
        shapes = {}

    environments = all_atomic_environments(structures)

    write_input("chemiscope_chemcv.json.gz", structures=structures, properties=properties, environments=environments, shapes=shapes, settings=settings)

def trj_chemcv(chemcv: dict, ylabel: str = "ChemCV", fixmin: bool = False, fixmax: bool = False, threshold: str = 0.0, legend: bool = True, fname: str = '') -> None:
    fig, ax = plt.subplots(layout="constrained")
    for key, value in chemcv.items():

        if max(value) - min(value) < threshold:
            continue

        if fixmin:
            value -= min(value)
            if fixmax:
                value /= max(value)
        elif fixmax:
            value -= max(value)

        if isinstance(key, tuple):
            label = '.'.join(key)
        else:
            #TODO change after change in TreeFrame
            label = '.'.join(key.split('.')[1:])

        ax.plot(value, label=label)
    
    ax.set_xlabel("number of frame")
    ax.set_ylabel(ylabel)
    if legend: fig.legend(loc="outside right upper")

    fig.savefig("_".join(["trj_chemcv", fname]) + ".svg")
    plt.close()

# def trj_chemcv_charges(chemcv: dict, fixmin: bool = False, fixmax: bool = False, threshold: float = 0., legend: bool = True) -> None:
#     fig, ax = plt.subplots()
#     for key, value in chemcv.items():

#         if max(value) - min(value) < threshold:
#             continue

#         if fixmin:
#             value -= min(value)
#             if fixmax:
#                 value /= max(value)
#         elif fixmax:
#             value -= max(value)

#         if isinstance(key, tuple):
#             label = '.'.join(key)
#         else:
#             label = key

#         ax.plot(value, label=label)
    
#     ax.set_xlabel("number of frame")
#     ax.set_ylabel("q [e]")
#     if legend: ax.legend()

#     fig.savefig("trj_chemcv_charges.svg")
#     plt.close()

# def trj_chemcv_populations(chemcv: dict, fixmin: bool = False, fixmax: bool = False, threshold: float = 0., legend: bool = True) -> None:
#     fig, ax = plt.subplots()
#     for key, value in chemcv.items():

#         if max(value) - min(value) < threshold:
#             continue

#         if fixmin:
#             value -= min(value)
#             if fixmax:
#                 value /= max(value)
#         elif fixmax:
#             value -= max(value)

#         if isinstance(key, tuple):
#             label = '.'.join(key)
#         else:
#             label = key
            
#         ax.plot(value, label=label)
    
#     ax.set_xlabel("number of frame")
#     ax.set_ylabel("p [%]")
#     if legend: ax.legend()

#     fig.savefig("trj_chemcv_populations.svg")
#     plt.close()

# def trj_chemcv_energies(chemcv: dict, fixmin: bool = False, fixmax: bool = False, threshold: float = 0., legend: bool = True) -> None:
#     fig, ax = plt.subplots()
#     for key, value in chemcv.items():

#         if max(value) - min(value) < threshold:
#             continue

#         if fixmin:
#             value -= min(value)
#             if fixmax:
#                 value /= max(value)
#         elif fixmax:
#             value -= max(value)

#         if isinstance(key, tuple):
#             label = '.'.join(key)
#         else:
#             label = key
            
#         ax.plot(value, label=label)
    
#     ax.set_xlabel("number of frame")
#     ax.set_ylabel("E [eV]")
#     if legend: ax.legend()

#     fig.savefig("trj_chemcv_energies.svg")
#     plt.close()

def pred(ref, pred):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color='k', linestyle='--')
    ax.plot(ref, pred, 'o', ms=1)
    
    ax.set_xlabel("Reference")
    ax.set_ylabel("Prediction")

    fig.savefig("pred.svg")
    plt.close()