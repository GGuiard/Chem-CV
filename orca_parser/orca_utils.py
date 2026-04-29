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

def sort_ao(dict: dict, reduction_order: str, key_start: int = 0) -> dict:
    l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    m_map = {'s': {'': 0},
             'p': {'x': 0, 'y': 1, 'z': 2},
             'd': {'xy': 0, 'xz': 1, 'yz': 2, 'x2y2': 3, 'z2': 4},
             'f': {'-3': 0, '-2': 1, '-1': 2, '0': 3, '+1': 4, '+2': 5, '+3': 6}}
    if reduction_order == 'a':
        return {key: value for key, value in sorted(dict.items())}
    elif reduction_order == 'an':
        return {key: value for key, value in sorted(dict.items())}
    elif reduction_order == 'anl':
        return {key: value for key, value in sorted(dict.items(), key=lambda x: tuple(x[0][:key_start]) + (x[0][key_start+0], x[0][key_start+1], l_map[x[0][key_start+2]]))}
    elif reduction_order == 'al':
        return {key: value for key, value in sorted(dict.items(), key=lambda x: tuple(x[0][:key_start]) + (x[0][key_start+0], l_map[x[0][key_start+1]]))}
    else:
        return {key: value for key, value in sorted(dict.items(), key=lambda x: tuple(x[0][:key_start]) + (x[0][key_start+0], x[0][key_start+1], l_map[x[0][key_start+2]], m_map[x[0][key_start+2]][x[0][key_start+3]]))}

def reduce_ao(dict: dict, reduction_order: str, key_start: int = 0) -> dict:
    reduced_dict = {}
    for key, value in dict.items():
        header = tuple(key[:key_start])
        a, n, l, _ = key[key_start:]
        if reduction_order == 'a':
            if key_start == 0:
                reduced_key = a
            else:
                reduced_key = header + tuple([a])
        elif reduction_order == 'an':
            reduced_key = header + (a, n)
        elif reduction_order == 'anl':
            reduced_key = header + (a, n, l)
        elif reduction_order == 'al':
            reduced_key = header + (a, l)
        if reduced_key not in reduced_dict:
            reduced_dict[reduced_key] = 0.0
        reduced_dict[reduced_key] += value
    return reduced_dict

### Parsing Functions ###

def q_Mulliken(properties: json) -> dict:
    q_list = properties["Geometries"][0]["Mulliken_Population_Analysis"][0]["AtomicCharges"]
    q_dict = {i: q[0] for i, q in enumerate(q_list)}
    return q_dict

def q_Orb_Mulliken(output_text: str) -> dict:
    pattern = r"MULLIKEN ORBITAL CHARGES.*?\n.*?\n.*?\n(.*?)(?:\nSum of orbital charges)"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        raise ValueError("MULLIKEN ORBITAL CHARGES section not found")
    
    section = match.group(1)
    charges = {}
    
    for line in section.split('\n'):
        parts = line.split()

        atom_index = get_atom_index(parts[1])
        n, l, m = get_ao_key(parts[2])
        charge = float(parts[3])

        charges[(atom_index, n, l, m)] = charge

    charges = reduce_ao(charges, 'anl')
    charges = sort_ao(charges, 'anl')

    return charges

def p_AtMO_Mulliken(output_text: str) -> dict:
    pattern = r"MULLIKEN ATOM POPULATIONS PER MO.*?\n.*?\n.*?\n(.*?)(?:\n{3})"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        raise ValueError("MULLIKEN ATOM POPULATIONS PER MO section not found")
    
    section = match.group(1)
    blocks = section.split('\n\n')
    
    populations = {}
    
    for block in blocks:
        lines = block.split('\n')

        mo_indices = [int(mo_index) for mo_index in lines[0].split()]

        for line in lines[4:]:
            parts = line.split()

            atom_index, atom_populations = int(parts[0]), [float(population) for population in parts[2:]]

            for mo_index, population in zip(mo_indices, atom_populations):
                if mo_index < 33:
                    populations[(mo_index, atom_index)] = population
    
    return populations

def p_OrbMO_Mulliken(output_text: str) -> dict:
    pattern = r"MULLIKEN ORBITAL POPULATIONS PER MO.*?\n.*?\n.*?\n(.*?)(?:\n{3})"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        raise ValueError("MULLIKEN ORBITAL POPULATIONS PER MO section not found")
    
    section = match.group(1)
    blocks = section.split('\n\n')
    
    populations = {}

    for block in blocks:
        lines = block.split('\n')

        mo_indices = [int(mo_index) for mo_index in lines[0].split()]

        for line in lines[4:]:
            parts = line.split()

            atom_index = get_atom_index(parts[0])
            n, l, m = get_ao_key(parts[1])
            orbital_populations = parts[2:]

            for mo_index, pop in zip(mo_indices, orbital_populations):
                if mo_index < 33:
                    populations[(mo_index, atom_index, n, l, m)] = float(pop)

    populations = reduce_ao(populations, 'al', 1)
    populations = sort_ao(populations, 'al', 1)

    return populations

def q_Loewdin(properties: json) -> dict:
    q_list = properties["Geometries"][0]["Loewdin_Population_Analysis"][0]["AtomicCharges"]
    q_dict = {i: q[0] for i, q in enumerate(q_list)}
    return q_dict

def q_Orb_Loewdin(output_text: str) -> dict:
    pattern = r"LOEWDIN ORBITAL CHARGES.*?\n.*?\n(.*?)(?:\n\n)"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        raise ValueError("LOEWDIN ORBITAL CHARGES section not found")
    
    section = match.group(1)
    charges = {}
    
    for line in section.split('\n'):
        parts = line.split()

        atom_index = get_atom_index(parts[1])
        n, l, m = get_ao_key(parts[2])
        charge = float(parts[3])

        charges[(atom_index, n, l, m)] = charge

    charges = reduce_ao(charges, 'anl')
    charges = sort_ao(charges, 'anl')

    return charges

def p_AtMO_Loewdin(output_text: str) -> dict:
    pattern = r"LOEWDIN ATOM POPULATIONS PER MO.*?\n.*?\n.*?\n(.*?)(?:\n{3})"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        raise ValueError("LOEWDIN ATOM POPULATIONS PER MO section not found")
    
    section = match.group(1)
    blocks = section.split('\n\n')
    
    populations = {}
    
    for block in blocks:
        lines = block.split('\n')

        mo_indices = [int(mo_index) for mo_index in lines[0].split()]

        for line in lines[4:]:
            parts = line.split()

            atom_index, atom_populations = int(parts[0]), [float(population) for population in parts[2:]]

            for mo_index, population in zip(mo_indices, atom_populations):
                if mo_index < 33:
                    populations[(mo_index, atom_index)] = population
    
    return populations

def p_OrbMO_Loewdin(output_text: str) -> dict:
    pattern = r"LOEWDIN ORBITAL POPULATIONS PER MO.*?\n.*?\n.*?\n(.*?)(?:\n{3})"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        raise ValueError("LOEWDIN ORBITAL POPULATIONS PER MO section not found")
    
    section = match.group(1)
    blocks = section.split('\n\n')
    
    populations = {}

    for block in blocks:
        lines = block.split('\n')

        mo_indices = [int(mo_index) for mo_index in lines[0].split()]

        for line in lines[4:]:
            parts = line.split()

            atom_index = get_atom_index(parts[0])
            n, l, m = get_ao_key(parts[1])
            orbital_populations = parts[2:]

            for mo_index, pop in zip(mo_indices, orbital_populations):
                if mo_index < 33:
                    populations[(mo_index, atom_index, n, l, m)] = float(pop)

    populations = reduce_ao(populations, 'al', 1)
    populations = sort_ao(populations, 'al', 1)

    return populations

def q_Mayer(properties: json) -> dict:
    q_list = properties["Geometries"][0]["Mayer_Population_Analysis"][0]["QA"]
    q_dict = {i: q[0] for i, q in enumerate(q_list)}
    return q_dict

def v_Mayer(properties: json) -> dict:
    v_list = properties["Geometries"][0]["Mayer_Population_Analysis"][0]["VA"]
    v_dict = {i: v[0] for i, v in enumerate(v_list)}
    return v_dict

def b_Mayer(properties: json) -> dict:
    b_list = properties["Geometries"][0]["Mayer_Population_Analysis"][0]["BondOrders"]
    components_list = properties["Geometries"][0]["Mayer_Population_Analysis"][0]["components"]
    b_dict = {(component[0], component[2]): b[0] for component, b in zip(components_list, b_list)}
    return b_dict

def q_Hirshfeld(properties: json) -> dict:
    q_list = properties["Geometries"][0]["Hirshfeld_Population_Analysis"][0]["AtomicCharges"]
    q_dict = {i: q[0] for i, q in enumerate(q_list)}
    return q_dict

def q_MBIS(properties: json) -> dict:
    q_list = properties["Geometries"][0]["MBIS_Population_Analysis"][0]["AtomicCharges"]
    q_dict = {i: q[0] for i, q in enumerate(q_list)}
    return q_dict

def npop_MBIS(properties: json) -> dict:
    npop_list = properties["Geometries"][0]["MBIS_Population_Analysis"][0]["NPOPVAL"]
    npop_dict = {i: npop[0] for i, npop in enumerate(npop_list)}
    return npop_dict

def sigma_MBIS(properties: json) -> dict:
    sigma_list = properties["Geometries"][0]["MBIS_Population_Analysis"][0]["SIGMAVAL"]
    sigma_dict = {i: sigma[0] for i, sigma in enumerate(sigma_list)}
    return sigma_dict

def q_CHELPG(properties: json) -> dict:
    q_list = properties["Geometries"][0]["CHELPG_Population_Analysis"][0]["AtomicCharges"]
    q_dict = {i: q[0] for i, q in enumerate(q_list)}
    return q_dict

def q_RESP(output_text: str) -> dict:
    pattern = r"RESP Charges.*\n-*?\n(.*?)(?:\n-*?\nTotal charge:)"
    match = re.search(pattern, output_text, re.DOTALL)
    
    if not match:
        raise ValueError("RESP Charges section not found")
    
    section = match.group(1)
    charges = {}
    
    for line in section.split('\n'):
            parts = line.split()

            atom_index = int(parts[0])
            charges[atom_index] = float(parts[3])
    
    return charges

def E_MO(output_text: str) -> dict:
    pattern = r"ORBITAL ENERGIES(?:.*\n){4}([\s\S]*?)\n.*\n\n"
    match = re.search(pattern, output_text)
    
    if not match:
        raise ValueError("ORBITAL ENERGIES")
    
    section = match.group(1)
    energies = {}
    
    for line in section.split('\n'):
            parts = line.split()

            atom_index = int(parts[0])
            energies[atom_index] = float(parts[3])
    
    return energies

### Dictionnary of the available ChemCV ###

AVAILABLE_CHEMCVS = {
    "q_Mulliken": {
        "simpleinput": "MULLIKEN",
        "block": "",
        "source": orca_property,
        "parsingfunction": q_Mulliken
        },
    "q_Orb_Mulliken": {
        "simpleinput": "MULLIKEN",
        "block": "%output Print[ P_OrbCharges_M ] 1 end",
        "source": orca_out,
        "parsingfunction": q_Orb_Mulliken
        },
    "p_AtMO_Mulliken": {
        "simpleinput": "MULLIKEN",
        "block": "%output Print[ P_AtPopMO_M ] 1 end",
        "source": orca_out,
        "parsingfunction": p_AtMO_Mulliken
        },
    "p_OrbMO_Mulliken": {
        "simpleinput": "MULLIKEN",
        "block": "%output Print[ P_OrbPopMO_M ] 1 end",
        "source": orca_out,
        "parsingfunction": p_OrbMO_Mulliken
        },
    "q_Loewdin": {
        "simpleinput": "LOEWDIN",
        "block": "",
        "source": orca_property,
        "parsingfunction": q_Loewdin
        },
    "q_Orb_Loewdin": {
        "simpleinput": "LOEWDIN",
        "block": "%output Print[ P_OrbCharges_L ] 1 end",
        "source": orca_out,
        "parsingfunction": q_Orb_Loewdin
        },
    "p_AtMO_Loewdin": {
        "simpleinput": "LOEWDIN",
        "block": "%output Print[ P_AtPopMO_L ] 1 end",
        "source": orca_out,
        "parsingfunction": p_AtMO_Loewdin
        },
    "p_OrbMO_Loewdin": {
        "simpleinput": "LOEWDIN",
        "block": "%output Print[ P_OrbPopMO_L ] 1 end",
        "source": orca_out,
        "parsingfunction": p_OrbMO_Loewdin
        },
    "q_Mayer": {
        "simpleinput": "MAYER",
        "block": "",
        "source": orca_property,
        "parsingfunction": q_Mayer
        },
    "v_Mayer": {
        "simpleinput": "MAYER",
        "block": "",
        "source": orca_property,
        "parsingfunction": v_Mayer
        },
    "b_Mayer": {
        "simpleinput": "MAYER",
        "block": "%method MAYER_BONDORDERTHRESH 0.00 end",
        "source": orca_property,
        "parsingfunction": b_Mayer
        },
    "q_Hirshfeld": {
        "simpleinput": "HIRSHFELD",
        "block": "",
        "source": orca_property,
        "parsingfunction": q_Hirshfeld
        },
    "q_MBIS": {
        "simpleinput": "MBIS",
        "block": "",
        "source": orca_property,
        "parsingfunction": q_MBIS
        },
    "npop_MBIS": {
        "simpleinput": "MBIS",
        "block": "",
        "source": orca_property,
        "parsingfunction": npop_MBIS
        },
    "sigma_MBIS": {
        "simpleinput": "MBIS",
        "block": "",
        "source": orca_property,
        "parsingfunction": sigma_MBIS
        },
    "q_CHELPG": {
        "simpleinput": "CHELPG",
        "block": "",
        "source": orca_property,
        "parsingfunction": q_CHELPG
        },         
    "q_RESP": {
        "simpleinput": "RESP",
        "block": "",
        "source": orca_out,
        "parsingfunction": q_RESP
        },
    "E_MO": {
        "simpleinput": "",
        "block": "",
        "source": orca_out,
        "parsingfunction": E_MO
        },
    }

# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.chdir("orca_parser")
    sep = "=" * 60

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

    print("q_Mulliken:", q_Mulliken(property), '\n')
    print("q_Loewdin:", q_Loewdin(property), '\n')
    print("q_Mayer:", q_Mayer(property), '\n')
    print("v_Mayer:", v_Mayer(property), '\n')
    print("b_Mayer:", b_Mayer(property), '\n')
    print("q_Hirshfeld:", q_Hirshfeld(property), '\n')
    print("q_MBIS:", q_MBIS(property), '\n')
    print("npop_MBIS:", npop_MBIS(property), '\n')
    print("sigma_MBIS:", sigma_MBIS(property), '\n')
    print("q_CHELPG:", q_CHELPG(property), '\n')

    # ------------------------------------------------------------------
    print('\n', sep)
    print("2. Output chemcv")
    print(sep, '\n')

    print("q_Orb_Mulliken:", q_Orb_Mulliken(output), '\n')
    print("p_AtMO_Mulliken:", p_AtMO_Mulliken(output), '\n')
    print("p_OrbMO_Mulliken:", p_OrbMO_Mulliken(output), '\n')
    print("q_Orb_Loewdin:", q_Orb_Loewdin(output), '\n')
    print("p_AtMO_Loewdin:", p_AtMO_Loewdin(output), '\n')
    print("p_OrbMO_Loewdin:", p_OrbMO_Loewdin(output), '\n')
    print("q_RESP:", q_RESP(output), '\n')
    print("E_MO:", E_MO(output), '\n')