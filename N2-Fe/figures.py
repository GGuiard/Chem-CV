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

def trj_d(time, d):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, d)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$d_{N-N}\ [A]$")

    fig.savefig("trj_d.svg")
    plt.close()

def trj_c(time, c):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, c)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel("Coordination")

    fig.savefig("trj_c.svg")
    plt.close()

def trj_z(time, z):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, z)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel("z [A]")

    fig.savefig("trj_z.svg")
    plt.close()

def trj_2D(time, d, c):
    fig, ax = plt.subplots(layout='tight')
    scatter = ax.scatter(d, c, s=1/100, c=time, cmap="magma")

    ax.set_xlabel(r"$d_{N-N}\ [A]$")
    ax.set_ylabel("Coordination")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(label="t [ps]")

    fig.savefig("trj_2D.png", dpi=300)
    plt.close()

def trj_xy(time, x, y):
    fig, ax = plt.subplots(layout='tight')
    scatter = ax.scatter(x, y, s=1/100, c=time, cmap="magma")

    ax.set_xlabel("x [A]")
    ax.set_ylabel("y [A]")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(label="t [ps]")

    fig.savefig("trj_xy.png", dpi=300)
    plt.close()

def trj_q(time, q):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, q)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$q_N\ [e]$")

    fig.savefig("trj_q.svg")
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

def density_d(grid, density):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(grid, density)

    ax.set_xlabel(r"$d_{N-N}\ [A]$")
    ax.set_ylabel("Density")
    
    fig.savefig("density_d.svg")
    plt.close()

def density_c(grid, density):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(grid, density)

    ax.set_xlabel("Coordination")
    ax.set_ylabel("Density")
    
    fig.savefig("density_c.svg")
    plt.close()

def density_q(grid, density):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(grid, density)

    ax.set_xlabel(r"$q_N\ [e]$")
    ax.set_ylabel("Density")
    
    fig.savefig("density_q.svg")
    plt.close()

def density_2D(grid_d, grid_c, density):
    fig, ax = plt.subplots(layout='tight')
    im = ax.contourf(grid_d, grid_c, density.T, 10, cmap=colormaps['Blues'])
    ax.contour(grid_d, grid_c, density.T, 10, linestyles='-', colors='darkgray', linewidths=1.2)

    ax.set_xlabel(r"$d_{N-N}\ [A]$")
    ax.set_ylabel("Coordination")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label="Density")

    fig.savefig("density_2D.svg")
    plt.close()

def fes_d(grid, fes, err):
    fig, ax = plt.subplots(layout='tight')
    ax.fill_between(grid, fes-err, fes+err, alpha=0.3)
    ax.plot(grid, fes)

    ax.set_xlabel(r"$d_{N-N}\ [A]$")
    ax.set_ylabel("FES [eV]")
    ax.set_ylim(0, 1)
    
    fig.savefig("fes_d.svg")
    plt.close()

def fes_c(grid, fes, err):
    fig, ax = plt.subplots(layout='tight')
    ax.fill_between(grid, fes-err, fes+err, alpha=0.3)
    ax.plot(grid, fes)

    ax.set_xlabel("Coordination")
    ax.set_ylabel("FES [eV]")
    ax.set_ylim(0, 1)
    
    fig.savefig("fes_c.svg")
    plt.close()

def fes_q(grid, fes, err):
    fig, ax = plt.subplots(layout='tight')
    ax.fill_between(grid, fes-err, fes+err, alpha=0.3)
    ax.plot(grid, fes)

    ax.set_xlabel(r"$q_N\ [e]$")
    ax.set_ylabel("FES [eV]")
    ax.set_ylim(0, 1)
    
    fig.savefig("fes_q.svg")
    plt.close()

def fes_2D(grid_d, grid_c, fes):
    fig, ax = plt.subplots(layout='tight')
    im = ax.contourf(grid_d, grid_c, fes.T, np.linspace(0, 1, 11), cmap=colormaps['Blues_r'])
    ax.contour(grid_d, grid_c, fes.T, np.linspace(0, 1, 11), linestyles='-', colors='darkgray', linewidths=1.2)

    ax.set_xlabel(r"$d_{N-N}\ [A]$")
    ax.set_ylabel("Coordination")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label="FES [eV]")

    fig.savefig("fes_2D.svg")
    plt.close()

def err_fes_2D(grid_d, grid_c, err):
    fig, ax = plt.subplots(layout='tight')
    im = ax.contourf(grid_d, grid_c, err.T, np.linspace(0, 0.02, 11), cmap=colormaps['Blues_r'])
    ax.contour(grid_d, grid_c, err.T, np.linspace(0, 0.02, 11), linestyles='-', colors='darkgray', linewidths=1.2)

    ax.set_xlabel(r"$d_{N-N} [A]$")
    ax.set_ylabel("Coordination")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label="Err(FES) [eV]")

    fig.savefig("err_fer_2D.svg")
    plt.close()

def av_d(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$\langle d_{N-N} \rangle\ [A]$")

    fig.savefig("av_d.svg")
    plt.close()

def av_c(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$\langle Coordination \rangle$")

    fig.savefig("av_c.svg")
    plt.close()

def av_q(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$\langle q_N \rangle\ [e]$")

    fig.savefig("av_q.svg")
    plt.close()

def delta_d(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$\Delta d_{N-N}\ [A]$")

    fig.savefig("delta_d.svg")
    plt.close()

def delta_c(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$\Delta Coordination$")

    fig.savefig("delta_c.svg")
    plt.close()

def delta_q(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$\Delta q_N\ [e]$")

    fig.savefig("delta_q.svg")
    plt.close()

def chemiscope(structures, time, d, c):
    properties = {"d": {"target": "structure",
                        "values": d,
                        "description": "Distance between the two atoms of nitrogen"},
                  "c": {"target": "structure",
                        "values": c,
                        "description": "Coordination between the atoms of nitrogen and the atoms of iron"},
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
                "map": {"x": {"property": "d"},
                        "y": {"property": "c"},
                        "color": {"property": "time"}},
                "structure": [{"bonds": False,
                               "spaceFilling": True,
                               "keepOrientation": True,
                               "playbackDelay": 200,
                               "supercell": [3,3,1]}]}
    
    write_input("chemiscope.json.gz", structures=structures, properties=properties, settings=settings)

def chemiscope_charges(structures, d, c, q):
    properties = {"d": {"target": "structure",
                        "values": d,
                        "description": "Distance between the two atoms of nitrogen"},
                  "c": {"target": "structure",
                        "values": c,
                        "description": "Coordination between the atoms of nitrogen and the atoms of iron"},
                  "q": {"target": "structure",
                        "values": (q[:,72]+q[:,73])/2,
                        "description": "charge [e]"},
                  "charge": {"target": "atom",
                        "values": q.ravel(),
                        "description": "charge [e]"}}

    settings = {"target": "structure",
                "map": {"x": {"property": "d"},
                        "y": {"property": "c"},
                        "color": {"property": "q"}},
                "structure": [{"bonds": False,
                               "spaceFilling": True,
                               "keepOrientation": True,
                               "playbackDelay": 200,
                               "supercell": [3,3,1],
                               "color": {"property": "charge", "palette": "bwr", "min":-1, "max":1}}]}

    environments = all_atomic_environments(structures)

    write_input("chemiscope_charges.json.gz", structures=structures, properties=properties, environments=environments, settings=settings)

def chemiscope_charge(structures, d, c, q):
    charge = np.outer(q, np.concatenate((-np.ones(72)/36, np.ones(2)))).ravel()

    properties = {"d": {"target": "structure",
                        "values": d,
                        "description": "Distance between the two atoms of nitrogen"},
                  "c": {"target": "structure",
                        "values": c,
                        "description": "Coordination between the atoms of nitrogen and the atoms of iron"},
                  "q": {"target": "structure",
                        "values": q,
                        "description": "charge [e]"},
                  "charge": {"target": "atom",
                        "values": charge,
                        "description": "charge [e]"}}

    settings = {"target": "structure",
                "map": {"x": {"property": "d"},
                        "y": {"property": "c"},
                        "color": {"property": "q"}},
                "structure": [{"bonds": False,
                               "spaceFilling": True,
                               "keepOrientation": True,
                               "playbackDelay": 200,
                               "supercell": [3,3,1],
                               "color": {"property": "charge", "palette": "bwr", "min":-1, "max":1}}]}

    environments = all_atomic_environments(structures)

    write_input("chemiscope_charge.json.gz", structures=structures, properties=properties, environments=environments, settings=settings)

def pred(ref, pred):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color='k', linestyle='--')
    ax.plot(ref, pred, 'o', ms=1)
    
    ax.set_xlabel("Reference")
    ax.set_ylabel("Prediction")

    fig.savefig("pred.svg")
    plt.close()