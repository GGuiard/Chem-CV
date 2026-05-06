import json
import subprocess
import re
from typing import Tuple

### Orca Output Generation ###

def orca_property(path: str = "ORCA") -> json:
    subprocess.run(f"orca_2json {path}/orca -property", shell=True, stdout=subprocess.DEVNULL)
    with open(f"{path}/orca.property.json", 'r') as f:
        orca_property = json.load(f)
    return orca_property

# useful ?
def orca_gbw(path: str = "ORCA") -> json:
    subprocess.run(f"orca_2json {path}/orca.gbw", shell=True, stdout=subprocess.DEVNULL)
    with open(f"{path}/orca.json", 'r') as f:
        orca_gbw = json.load(f)
    return orca_gbw

def orca_out(path: str = "ORCA") -> str:
    with open(f"{path}/orca.out", 'r') as f:
        return f.read()

### Helping Functions ###

def get_atom_index(atom_key : str) -> int:
    return int(re.match(r'^(\d+)', atom_key).group(1))

def get_ao_key(ao_str: str) -> Tuple[int, str, str]:
    """
    Extract and convert ao notation to sortable components.
    
    Args:
        ao_str: String
    
    Returns:
        Tuple: (n, l, m)
    """
    # Extract quantum numbers
    match = re.match(r'^(\d+)([spdf])(.*)', ao_str)
    
    n = int(match.group(1))
    l = match.group(2)
    m = match.group(3) if match.group(3) else ""
    
    return (n, l, m)

def get_redao_key(redao_str: str) -> Tuple[int, str, str]:
    """
    Extract and convert redao notation to sortable components.
    
    Args:
        redao_str: String
    
    Returns:
        Tuple: (n, l, m)
    """
    # Extract quantum numbers
    match = re.match(r'^([spdf])(.*)', redao_str)
    
    l = match.group(1)
    m = match.group(2) if match.group(2) else ""
    
    return (l, m)

def is_in_list_tuples(tuple: tuple[int|str|None], list_tuples: list[tuple[int|str|None]]) -> bool:
    for test_tuple in list_tuples:
        is_in = True
        for value, test_value in zip(tuple, test_tuple):
            if value!=None and test_value!=None:
                if value != test_value:
                    is_in = False
                    break
        if is_in:
            return True
    return False

def sort(dict: dict, fmt: list[str]) -> dict:
    l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    m_map = {'': 0,
             'x': 1, 'y': 2, 'z': 3,
             'xy': 4, 'xz': 5, 'yz': 6, 'x2y2': 7, 'z2': 8,
             '-3': 9, '-2': 10, '-1': 11, '0': 12, '+1': 13, '+2': 14, '+3': 15}
    
    sort_functions = []
    for key_type in fmt:
        if key_type == 'l':
            sort_functions.append(lambda x: l_map[x])
        elif key_type == 'm':
            sort_functions.append(lambda x: m_map[x])
        else:
            sort_functions.append(lambda x: x)
    func = lambda item: tuple([f(x) for f, x in zip(sort_functions, item[0])])
    
    return {key: value for key, value in sorted(dict.items(), key=func)}
    
def reduce(dict: dict, new_format: list[str], old_format: list[str]) -> dict:
    reduced_dict = {}
    for tuple_key, value in dict.items():
        reduced_tuple_key = tuple([tuple_key[old_format.index(key)] for key in new_format])
        if reduced_tuple_key not in reduced_dict:
            reduced_dict[reduced_tuple_key] = 0.0
        reduced_dict[reduced_tuple_key] += value
    return reduced_dict

### Parsing Functions ###

def q_Mulliken(properties: json, atoms: list[int] | None = None) -> dict:
    q_list = properties["Geometries"][0]["Mulliken_Population_Analysis"][0]["AtomicCharges"]
    q_dict = {atom: q[0] for atom, q in enumerate(q_list)}
    if atoms:
        q_dict = {atom: q for atom, q in q_dict.items() if atom in atoms}
    return q_dict

def q_AO_Mulliken(output_text: str, aos: list[tuple[int|str|None]] = [(None, "p", None, None)], fmt: list[str] = ["atom", "n", "l", "m"]) -> dict:
    pattern = r"MULLIKEN ORBITAL CHARGES.*?\n.*?\n.*?\n(.*?)(?:\nSum of orbital charges)"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        print("Warning: MULLIKEN ORBITAL CHARGES section not found")
        return {}
    
    section = match.group(1)
    charges = {}
    
    for line in section.split('\n'):
        parts = line.split()

        ao = (get_atom_index(parts[1]),) + get_ao_key(parts[2])
        charge = float(parts[3])

        if is_in_list_tuples(ao, aos):
            charges[ao] = charge

    default_format = ["atom", "n", "l", "m"]
    if fmt != default_format:
        charges = reduce(charges, fmt, default_format)
    charges = sort(charges, fmt)

    return charges

def p_MOAt_Mulliken(output_text: str, moats: list[tuple[int|None]] = [(None, None)], fmt: list[str] = ["MO", "atom"], threshold: float = 0.1) -> dict:
    pattern = r"MULLIKEN ATOM POPULATIONS PER MO.*?\n.*?\n.*?\n(.*?)(?:\n{3})"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        print("Warning: MULLIKEN ATOM POPULATIONS PER MO section not found")
        return {}
    
    section = match.group(1)
    blocks = section.split('\n\n')
    
    populations = {}
    
    for block in blocks:
        lines = block.split('\n')

        mo_indices = [int(mo_index) for mo_index in lines[0].split()]

        for line in lines[4:]:
            parts = line.split()

            atom = int(parts[0])
            atom_populations = [float(population) for population in parts[2:]]

            for mo, population in zip(mo_indices, atom_populations):
                moat = (mo, atom)
                if is_in_list_tuples(moat, moats) and threshold <= population:
                    populations[moat] = population

    default_format = ["MO", "atom"]
    if fmt != default_format:
        populations = reduce(populations, fmt, default_format)
    populations = sort(populations, fmt)
    
    return populations

def p_MOAO_Mulliken(output_text: str, moaos: list[tuple[int|str|None]] = [(None, "p", None, None, None)], fmt: list[str] = ["MO", "atom", "n", "l", "m"], threshold: float = 0.1) -> dict:
    pattern = r"MULLIKEN ORBITAL POPULATIONS PER MO.*?\n.*?\n.*?\n(.*?)(?:\n{3})"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        print("Warning: MULLIKEN ORBITAL POPULATIONS PER MO section not found")
        return {}
    
    section = match.group(1)
    blocks = section.split('\n\n')
    
    populations = {}

    for block in blocks:
        lines = block.split('\n')

        mo_indices = [int(mo_index) for mo_index in lines[0].split()]

        for line in lines[4:]:
            parts = line.split()

            ao = (get_atom_index(parts[0]),) + get_ao_key(parts[1])
            ao_populations = [float(ao_population) for ao_population in parts[2:]]

            for mo, population in zip(mo_indices, ao_populations):
                moao = (mo,) + ao
                if is_in_list_tuples(moao, moaos) and threshold <= population:
                    populations[moao] = population

    default_format = ["MO", "atom", "n", "l", "m"]
    if fmt != default_format:
        populations = reduce(populations, fmt, default_format)
    populations = sort(populations, fmt)

    return populations

def q_Loewdin(properties: json, atoms: list[int] | None = None) -> dict:
    q_list = properties["Geometries"][0]["Loewdin_Population_Analysis"][0]["AtomicCharges"]
    q_dict = {i: q[0] for i, q in enumerate(q_list)}
    if atoms:
        q_dict = {atom: q for atom, q in q_dict.items() if atom in atoms}
    return q_dict

def q_AO_Loewdin(output_text: str, aos: list[tuple[int|str|None]] = [(None, "p", None, None)], fmt: list[str] = ["atom", "n", "l", "m"]) -> dict:
    pattern = r"LOEWDIN ORBITAL CHARGES.*?\n.*?\n(.*?)(?:\n\n)"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        print("Warning: LOEWDIN ORBITAL CHARGES section not found")
        return {}

    section = match.group(1)
    charges = {}
    
    for line in section.split('\n'):
        parts = line.split()

        ao = (get_atom_index(parts[1]),) + get_ao_key(parts[2])
        charge = float(parts[3])

        if is_in_list_tuples(ao, aos):
            charges[ao] = charge

    default_format = ["atom", "n", "l", "m"]
    if fmt != default_format:
        charges = reduce(charges, fmt, default_format)
    charges = sort(charges, fmt)

    return charges

def p_MOAt_Loewdin(output_text: str, moats: list[tuple[int|None]] = [(None, None)], fmt: list[str] = ["MO", "atom"], threshold: float = 0.1) -> dict:
    pattern = r"LOEWDIN ATOM POPULATIONS PER MO.*?\n.*?\n.*?\n(.*?)(?:\n{3})"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        print("Warning: LOEWDIN ATOM POPULATIONS PER MO section not found")
        return {}
    
    section = match.group(1)
    blocks = section.split('\n\n')
    
    populations = {}
    
    for block in blocks:
        lines = block.split('\n')

        mo_indices = [int(mo_index) for mo_index in lines[0].split()]

        for line in lines[4:]:
            parts = line.split()

            atom = int(parts[0])
            atom_populations = [float(population) for population in parts[2:]]

            for mo, population in zip(mo_indices, atom_populations):
                moat = (mo, atom)
                if is_in_list_tuples(moat, moats) and threshold <= population:
                    populations[moat] = population

    default_format = ["MO", "atom"]
    if fmt != default_format:
        populations = reduce(populations, fmt, default_format)
    populations = sort(populations, fmt)
    
    return populations

def p_MOAO_Loewdin(output_text: str, moaos: list[tuple[int|str|None]] = [(None, "p", None, None, None)], fmt: list[str] = ["MO", "atom", "n", "l", "m"], threshold: float = 0.1) -> dict:
    pattern = r"LOEWDIN ORBITAL POPULATIONS PER MO.*?\n.*?\n.*?\n(.*?)(?:\n{3})"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        print("Warning: LOEWDIN ORBITAL POPULATIONS PER MO section not found")
        return {}
    
    section = match.group(1)
    blocks = section.split('\n\n')
    
    populations = {}

    for block in blocks:
        lines = block.split('\n')

        mo_indices = [int(mo_index) for mo_index in lines[0].split()]

        for line in lines[4:]:
            parts = line.split()

            ao = (get_atom_index(parts[0]),) + get_ao_key(parts[1])
            ao_populations = [float(ao_population) for ao_population in parts[2:]]

            for mo, population in zip(mo_indices, ao_populations):
                moao = (mo,) + ao
                if is_in_list_tuples(moao, moaos) and threshold <= population:
                    populations[moao] = population

    default_format = ["MO", "atom", "n", "l", "m"]
    if fmt != default_format:
        populations = reduce(populations, fmt, default_format)
    populations = sort(populations, fmt)

    return populations

def q_Mayer(properties: json, atoms: list[int] | None = None) -> dict:
    q_list = properties["Geometries"][0]["Mayer_Population_Analysis"][0]["QA"]
    q_dict = {atom: q[0] for atom, q in enumerate(q_list)}
    if atoms:
        q_dict = {atom: q for atom, q in q_dict.items() if atom in atoms}
    return q_dict

def v_Mayer(properties: json, atoms: list[int] | None = None) -> dict:
    v_list = properties["Geometries"][0]["Mayer_Population_Analysis"][0]["VA"]
    v_dict = {atom: v[0] for atom, v in enumerate(v_list)}
    if atoms:
        v_dict = {atom: v for atom, v in v_dict.items() if atom in atoms}
    return v_dict

def b_Mayer(properties: json, bonds: list[tuple[int|None]] = [(None, None)]) -> dict:
    b_list = properties["Geometries"][0]["Mayer_Population_Analysis"][0]["BondOrders"]
    components_list = properties["Geometries"][0]["Mayer_Population_Analysis"][0]["components"]
    b_dict = {(component[0], component[2]): b[0] for component, b in zip(components_list, b_list)}
    b_dict = {bond: b for bond, b in b_dict.items() if is_in_list_tuples(bond, bonds)}
    return b_dict

def q_Hirshfeld(properties: json, atoms: list[int] | None = None) -> dict:
    q_list = properties["Geometries"][0]["Hirshfeld_Population_Analysis"][0]["AtomicCharges"]
    q_dict = {atom: q[0] for atom, q in enumerate(q_list)}
    if atoms:
        q_dict = {atom: q for atom, q in q_dict.items() if atom in atoms}
    return q_dict

def q_MBIS(properties: json, atoms: list[int] | None = None) -> dict:
    q_list = properties["Geometries"][0]["MBIS_Population_Analysis"][0]["AtomicCharges"]
    q_dict = {atom: q[0] for atom, q in enumerate(q_list)}
    if atoms:
        q_dict = {atom: q for atom, q in q_dict.items() if atom in atoms}
    return q_dict

def npop_MBIS(properties: json, atoms: list[int] | None = None) -> dict:
    npop_list = properties["Geometries"][0]["MBIS_Population_Analysis"][0]["NPOPVAL"]
    npop_dict = {atom: npop[0] for atom, npop in enumerate(npop_list)}
    if atoms:
        npop_dict = {atom: npop for atom, npop in npop_dict.items() if atom in atoms}
    return npop_dict

def sigma_MBIS(properties: json, atoms: list[int] | None = None) -> dict:
    sigma_list = properties["Geometries"][0]["MBIS_Population_Analysis"][0]["SIGMAVAL"]
    sigma_dict = {atom: sigma[0] for atom, sigma in enumerate(sigma_list)}
    if atoms:
        sigma_dict = {atom: sigma for atom, sigma in sigma_dict.items() if atom in atoms}
    return sigma_dict

def q_CHELPG(properties: json, atoms: list[int] | None = None) -> dict:
    q_list = properties["Geometries"][0]["CHELPG_Population_Analysis"][0]["AtomicCharges"]
    q_dict = {atom: q[0] for atom, q in enumerate(q_list)}
    if atoms:
        q_dict = {atom: q for atom, q in q_dict.items() if atom in atoms}
    return q_dict

def q_RESP(output_text: str, atoms: list[int] | None = None) -> dict:
    pattern = r"RESP Charges.*\n-*?\n(.*?)(?:\n-*?\nTotal charge:)"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        print("Warning: RESP Charges section not found")
        return {}
    
    section = match.group(1)
    charges = {}
    
    for line in section.split('\n'):
        parts = line.split()

        atom = int(parts[0])
        if atoms:
            if atom not in atoms:
                continue
        charges[atom] = float(parts[3])
    
    return charges

def E_MO(output_text: str, mos: list[int] | None = None) -> dict:
    pattern = r"ORBITAL ENERGIES(?:.*\n){4}([\s\S]*?)\n.*\n\n"
    match = re.search(pattern, output_text)
    
    if not match:
        print("Warning: ORBITAL ENERGIES section not found")
        return {}
    
    section = match.group(1)
    energies = {}
    
    for line in section.split('\n'):
            parts = line.split()

            mo = int(parts[0])
            if mos:
                if mo not in mos:
                    continue
            energies[mo] = float(parts[3])
    
    return energies

def N_FOD(output_text: str, selection: None = None) -> dict:
    pattern = r"N_FOD =(.*)"
    match = re.search(pattern, output_text)
    
    if not match:
        print("Warning: N_FOD section not found")
        return {}
    
    line = match.group(1)
    
    parts = line.split()

    n_fod = float(parts[0])
    
    return n_fod

def q_AO_FOD(output_text: str, aos: list[tuple[int|None]] = [(None, None)], fmt: list[str] = ["atom", "l", "m"]) -> dict:
    pattern = r"FOD BASED MULLIKEN REDUCED ORBITAL CHARGES.*?\n.*?\n(.*?)(?:\n{3})"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        print("Warning: FOD BASED MULLIKEN REDUCED ORBITAL CHARGES section not found")
        return {}
    
    section = match.group(1)
    blocks = section.split('\n\n')
    
    populations = {}
    
    for block in blocks:
        lines = block.split('\n')

        line = lines[0]

        atom = int(line[:6].split()[0])
        ao = (atom,) + get_redao_key(line[6:13].strip())
        q = float(line[13:].split()[1])

        if is_in_list_tuples(ao, aos):
            populations[ao] = q

        for line in lines[1:]:
            parts = line.split()

            ao = (atom,) + get_redao_key(parts[0])
            q = float(parts[2])

            if is_in_list_tuples(ao, aos):
                populations[ao] = q

    default_format = ["atom", "l", "m"]
    if fmt != default_format:
        populations = reduce(populations, fmt, default_format)
    populations = sort(populations, fmt)
    
    return populations

### Dictionnary of the available ChemCV ###

AVAILABLE_CHEMCVS = {
    "q_Mulliken": {
        "simpleinput": "MULLIKEN",
        "block": "",
        "source": orca_property,
        "parsingfunction": q_Mulliken,
        "selection_keys": ["atom"],
        },
    "q_AO_Mulliken": {
        "simpleinput": "MULLIKEN",
        "block": "%output Print[ P_OrbCharges_M ] 1 end",
        "source": orca_out,
        "parsingfunction": q_AO_Mulliken,
        "selection_keys": ["atom", "n", "l", "m"],
        },
    "p_MOAt_Mulliken": {
        "simpleinput": "MULLIKEN",
        "block": "%output Print[ P_AtPopMO_M ] 1 end",
        "source": orca_out,
        "parsingfunction": p_MOAt_Mulliken,
        "selection_keys": ["MO", "atom"],
        },
    "p_MOAO_Mulliken": {
        "simpleinput": "MULLIKEN",
        "block": "%output Print[ P_OrbPopMO_M ] 1 end",
        "source": orca_out,
        "parsingfunction": p_MOAO_Mulliken,
        "selection_keys": ["MO", "atom", "n", "l", "m"],
        },
    "q_Loewdin": {
        "simpleinput": "LOEWDIN",
        "block": "",
        "source": orca_property,
        "parsingfunction": q_Loewdin,
        "selection_keys": ["atom"],
        },
    "q_AO_Loewdin": {
        "simpleinput": "LOEWDIN",
        "block": "%output Print[ P_OrbCharges_L ] 1 end",
        "source": orca_out,
        "parsingfunction": q_AO_Loewdin,
        "selection_keys": ["atom", "n", "l", "m"],
        },
    "p_MOAt_Loewdin": {
        "simpleinput": "LOEWDIN",
        "block": "%output Print[ P_AtPopMO_L ] 1 end",
        "source": orca_out,
        "parsingfunction": p_MOAt_Loewdin,
        "selection_keys": ["MO", "atom"],
        },
    "p_MOAO_Loewdin": {
        "simpleinput": "LOEWDIN",
        "block": "%output Print[ P_OrbPopMO_L ] 1 end",
        "source": orca_out,
        "parsingfunction": p_MOAO_Loewdin,
        "selection_keys": ["MO", "atom", "n", "l", "m"],
        },
    "q_Mayer": {
        "simpleinput": "MAYER",
        "block": "",
        "source": orca_property,
        "parsingfunction": q_Mayer,
        "selection_keys": ["atom"],
        },
    "v_Mayer": {
        "simpleinput": "MAYER",
        "block": "",
        "source": orca_property,
        "parsingfunction": v_Mayer,
        "selection_keys": ["atom"],
        },
    "b_Mayer": {
        "simpleinput": "MAYER",
        "block": "",
        "source": orca_property,
        "parsingfunction": b_Mayer,
        "selection_keys": ["atom", "atom"],
        },
    "q_Hirshfeld": {
        "simpleinput": "HIRSHFELD",
        "block": "",
        "source": orca_property,
        "parsingfunction": q_Hirshfeld,
        "selection_keys": ["atom"],
        },
    "q_MBIS": {
        "simpleinput": "MBIS",
        "block": "",
        "source": orca_property,
        "parsingfunction": q_MBIS,
        "selection_keys": ["atom"],
        },
    "npop_MBIS": {
        "simpleinput": "MBIS",
        "block": "",
        "source": orca_property,
        "parsingfunction": npop_MBIS,
        "selection_keys": ["atom"],
        },
    "sigma_MBIS": {
        "simpleinput": "MBIS",
        "block": "",
        "source": orca_property,
        "parsingfunction": sigma_MBIS,
        "selection_keys": ["atom"],
        },
    "q_CHELPG": {
        "simpleinput": "CHELPG",
        "block": "",
        "source": orca_property,
        "parsingfunction": q_CHELPG,
        "selection_keys": ["atom"],
        },         
    "q_RESP": {
        "simpleinput": "RESP",
        "block": "",
        "source": orca_out,
        "parsingfunction": q_RESP,
        "selection_keys": ["atom"],
        },
    "E_MO": {
        "simpleinput": "",
        "block": "",
        "source": orca_out,
        "parsingfunction": E_MO,
        "selection_keys": ["MO"],
        },
    }

    # "N_FOD": {
    #     "simpleinput": "FOD",
    #     "block": "",
    #     "source": orca_out,
    #     "parsingfunction": N_FOD,
    #     "selection_keys": [],
    #     },
    # "q_AO_FOD": {
    #     "simpleinput": "FOD",
    #     "block": "",
    #     "source": orca_out,
    #     "parsingfunction": q_AO_FOD,
    #     "selection_keys": ["atom", "l", "m"],
    #     },

# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.chdir("orca_parser")
    sep = "=" * 60

    from ase import Atoms
    from ase.calculators.orca import ORCA

    atoms = Atoms("CClH3Cl", positions=[( 0.000, 0.000, 0.000),
                                        ( 0.000, 0.000, 1.800),
                                        ( 0.000, 1.076, 0.000),
                                        ( 0.935,-0.540, 0.000),
                                        (-0.935,-0.540, 0.000),
                                        ( 0.000, 0.000,-2.300)])
    
    simpleinput = ' '.join(set([chemcv["simpleinput"] for chemcv in AVAILABLE_CHEMCVS.values()]))
    blocks = '\n'.join(set([chemcv["block"] for chemcv in AVAILABLE_CHEMCVS.values()]))
    atoms.calc = ORCA(charge=-1, mult=1, directory="ORCA", 
                      orcasimpleinput=' '.join(["WB97X-D4 def2-TZVPD", simpleinput]), 
                      orcablocks='\n'.join(["%pal nprocs 32 end", blocks]))
    _ = atoms.get_potential_energy()

    # ------------------------------------------------------------------
    print(sep)
    print("1. Output preformating")
    print(sep, '\n')

    property = orca_property()
    if property: print("Property Downloaded")
    gbw = orca_gbw()
    if gbw: print("GBW Downloaded")
    output = orca_out()
    if output: print("Output Downloaded")

    # ------------------------------------------------------------------
    print('\n', sep)
    print("2. Property chemcv")
    print(sep, '\n')

    print("q_Mulliken:", q_Mulliken(property, atoms=[0,1,5]), '\n')
    print("q_Loewdin:", q_Loewdin(property, atoms=[0,1,5]), '\n')
    print("q_Mayer:", q_Mayer(property, atoms=[0,1,5]), '\n')
    print("v_Mayer:", v_Mayer(property, atoms=[0,1,5]), '\n')
    print("b_Mayer:", b_Mayer(property, bonds=[(0,1), (0,5), (1,5)]), '\n')
    print("q_Hirshfeld:", q_Hirshfeld(property, atoms=[0,1,5]), '\n')
    print("q_MBIS:", q_MBIS(property, atoms=[0,1,5]), '\n')
    print("npop_MBIS:", npop_MBIS(property, atoms=[0,1,5]), '\n')
    print("sigma_MBIS:", sigma_MBIS(property, atoms=[0,1,5]), '\n')
    print("q_CHELPG:", q_CHELPG(property, atoms=[0,1,5]), '\n')

    # ------------------------------------------------------------------
    print('\n', sep)
    print("2. Output chemcv")
    print(sep, '\n')

    print("q_AO_Mulliken:", q_AO_Mulliken(output, aos=[(0, None, "p", None), (1, None, "p", None), (5, None, "p", None)], fmt=["atom", "l"]), '\n')
    print("p_MOAt_Mulliken:", p_MOAt_Mulliken(output, moats=[(21, 0), (21, 1), (21, 5), (22, 0), (22, 1), (22, 5)]), '\n')
    print("p_MOAO_Mulliken:", p_MOAO_Mulliken(output, moaos=[(21, 0, None, "p", None), (21, 1, None, "p", None), (21, 5, None, "p", None), (22, 0, None, "p", None), (22, 1, None, "p", None), (22, 5, None, "p", None)], fmt=["MO", "atom", "l"]), '\n')
    print("q_AO_Loewdin:", q_AO_Loewdin(output, aos=[(0, None, "p", None), (1, None, "p", None), (5, None, "p", None)], fmt=["atom", "l"]), '\n')
    print("p_MOAt_Loewdin:", p_MOAt_Loewdin(output, moats=[(21, 0), (21, 1), (21, 5), (22, 0), (22, 1), (22, 5)]), '\n')
    print("p_MOAO_Loewdin:", p_MOAO_Loewdin(output, moaos=[(21, 0, None, "p", None), (21, 1, None, "p", None), (21, 5, None, "p", None), (22, 0, None, "p", None), (22, 1, None, "p", None), (22, 5, None, "p", None)], fmt=["MO", "atom", "l"]), '\n')
    print("q_RESP:", q_RESP(output, atoms=[0,1,5]), '\n')
    print("E_MO:", E_MO(output, mos=[21,22]), '\n')
    # print("N_FOD:", N_FOD(output), '\n')
    # print("q_AO_FOD:", q_AO_FOD(output, aos=[(0, "p", None), (1, "p", None), (5, "p", None)], fmt=["atom", "l"]), '\n')