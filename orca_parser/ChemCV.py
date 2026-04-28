from TreeFrame import TreeFrame
from orca_utils import *

from typing import List, Tuple, Optional

def nest_tuple_dict(d: dict) -> dict:
    """
    Transform a flat dict with tuple keys into a nested dict.
    Works with tuples of any size and mixed types.
    Preserves insertion order (Python 3.7+).
    """
    result = {}

    for keys, value in d.items():
        current = result
        if isinstance(keys, tuple):
            for key in keys[:-1]:
                if key not in current and key != '':
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        else:
            current[keys] = value

    return result

class ChemCV(TreeFrame):
    """Manages Chemical Collective Variables parsed from ORCA outputs."""

    def __init__(self, chemcvs: Optional[List[str]] = None, nb_traj: int = 0, fill_incomplete: str = "zeros") -> None:
        """
        Initialize ChemCV TreeFrame.
        
        Args:
            chemcvs: List of ChemCVs to parse. If None and parse_all=True, parse all.
            nb_traj: Number of structures.
            fill_incomplete : {'zeros', 'drop'}
                              How to handle mismatches between the stored leaves and ``data``
                              passed to ``update``.  See module docstring for full semantics.
        """
        # Initialize TreeFrame
        super().__init__(nb_traj, fill_incomplete)
        
        # Determine which ChemCVs to parse
        if chemcvs is None:
            self.active_chemcvs = list(AVAILABLE_CHEMCVS.keys())
        else:
            self.active_chemcvs = chemcvs
        
        # Validate requested ChemCVs
        invalid = set(self.active_chemcvs) - set(AVAILABLE_CHEMCVS.keys())
        if invalid:
            raise ValueError(f"Unknown ChemCVs: {invalid}")

    def get_orca_input(self) -> Tuple[str, str]:
        """
        Generate ORCA input strings based on active ChemCVs.
        
        Returns:
            Tuple of (orcasimpleinput, orcablocks)
        """
        # Map ChemCVs to ORCA keywords
        simpleinput = set()
        blocks = set()
        for chemcv in self.active_chemcvs:
            simpleinput.add(AVAILABLE_CHEMCVS[chemcv]["simpleinput"])
            blocks.add(AVAILABLE_CHEMCVS[chemcv]["block"])
        
        simpleinput = ' '.join(simpleinput)
        blocks = '\n'.join(blocks)
        
        return simpleinput, blocks
    
    def update(self, step: int | None = None, path: str = "ORCA"):
        """
        Parse and fill ChemCV data from ORCA properties.
        """
        for source in set(AVAILABLE_CHEMCVS[chemcv]["source"] for chemcv in self.active_chemcvs):
            if source == orca_property:
                properties = orca_property(path)
            elif source == orca_gbw:
                gbw = orca_gbw(path)
            elif source == orca_out:
                output = orca_out(path)
            else:
                raise ValueError(f"Unknown source: {source}")
        
        data = {}
        for chemcv in self.active_chemcvs:
            parsing_function = AVAILABLE_CHEMCVS[chemcv]["parsingfunction"]
            source = AVAILABLE_CHEMCVS[chemcv]["source"]
            try:
                if source == orca_property:
                    chemcv_data = parsing_function(properties)
                elif source == orca_gbw:
                    chemcv_data = parsing_function(gbw)
                elif source == orca_out:
                    chemcv_data = parsing_function(output)
                data[chemcv] = nest_tuple_dict(chemcv_data)
            except (KeyError, IndexError, TypeError) as e:
                print(f"Warning: Could not extract {chemcv}: {e}")

        super().update(data, step)