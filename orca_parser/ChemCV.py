if __name__ == "__main__":
    from TreeFrame import TreeFrame
    from orca_utils import *
else:
    from .TreeFrame import TreeFrame
    from .orca_utils import *

from typing import List, Tuple, Optional, Any
from pathlib import Path
import itertools

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
            keys = tuple(key for key in keys if key!='')
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        else:
            current[keys] = value

    return result

class ChemCV(TreeFrame):
    """Manages Chemical Collective Variables parsed from ORCA outputs."""

    def __init__(
        self,
        chemcvs: list[str] | None = None,
        selections_per_cv: dict[str, list] = {},
        selections_per_type: dict[str, list] = {},
        kwargs_per_cv: dict[str, Any] = {},
        nb_traj: int = 0,
        fill_incomplete: str = "zeros",
    ) -> None:
        """
        Initialize ChemCV TreeFrame.

        Args:
            chemcvs:
                - None -> parse all available CVs.
                - list -> parse these CVs, all using global ``selections``.
                - dict -> keys are CV names; values are per-CV selection dicts
                          that are *merged on top of* the global ``selections``
                          (per-CV wins on collision).

            selections:
                Global selection defaults shared by all CVs unless overridden.
                Keys are dimension names matching ``selection_keys`` declared
                in ``AVAILABLE_CHEMCVS``; values are lists of indices/labels.

                If a CV needs a dimension that is absent from both global and
                per-CV selections, the corresponding kwarg is omitted and the
                parsing function falls back to its own default (no filter).

            nb_traj:
                Number of structures.

            fill_incomplete:
                {'zeros', 'drop'} — how to handle leaf/data mismatches in
                ``update``. See TreeFrame docstring for full semantics.
        """
        super().__init__(nb_traj, fill_incomplete)

        # --- ChemCVs ----------------------------------------
        if chemcvs is None:
            self.active_chemcvs = list(AVAILABLE_CHEMCVS.keys())
        else:
            self.active_chemcvs = chemcvs

        invalid = set(self.active_chemcvs) - set(AVAILABLE_CHEMCVS.keys())
        if invalid:
            raise ValueError(f"Unknown ChemCVs: {invalid}")
        
        # --- selection -------------------------------------------------
        self.selections = self._resolve_selection(selections_per_cv, selections_per_type)

        # --- selection -------------------------------------------------
        self.kwargs_per_cv = kwargs_per_cv

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_selection(
        self,
        selections_per_cv: dict[str, list] = {},
        selections_per_type: dict[str, list] = {}
    ) -> dict[str, list]:
        for chemcv in self.active_chemcvs:
            if chemcv in selections_per_cv:
                continue
            else:
                list_product = []

                for selection_key in AVAILABLE_CHEMCVS[chemcv]["selection_keys"]:
                    if selection_key in selections_per_type:
                        selection_value = selections_per_type[selection_key]
                        if isinstance(selection_value, list):
                            list_product.append(selection_value)
                        elif isinstance(selection_value, (str, int)):
                            list_product.append([selection_value])
                        else:
                            print(f"Warning: selection_per_type[{chemcv}] must be a list, str or int, "\
                                  f"{type(selection_value)} was passed. "\
                                  "This ChemCV was ignored for the selection")
                            list_product.append([None])
                    else:
                        list_product.append([None])

                if len(list_product) == 1:
                    if list_product[0] == [None]:
                        selections_per_cv[chemcv] = None
                    else:
                        selections_per_cv[chemcv] = list_product[0]
                else:
                    selections_per_cv[chemcv] = list(itertools.product(*list_product))

        return selections_per_cv

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
        
        simpleinput = ' '.join([txt for txt in simpleinput if txt])
        blocks = '\n'.join([txt for txt in blocks if txt])
        
        return simpleinput, blocks
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, step: int | None = None, path: str = "ORCA"):
        """
        Parse and fill ChemCV data from ORCA outputs.

        Each active CV's selections are resolved independently, so different
        CVs can target different atoms/MOs without conflict.
        """
        # --- load sources (each source loaded at most once) -------------------
        required_sources = {AVAILABLE_CHEMCVS[cv]["source"] for cv in self.active_chemcvs}
        loaded = {}
        for source_fn in required_sources:
            loaded[source_fn] = source_fn(path)

        # --- parse each CV ----------------------------------------------------
        data = {}
        for chemcv in self.active_chemcvs:
            spec = AVAILABLE_CHEMCVS[chemcv]
            parsing_function = spec["parsingfunction"]
            source_fn = spec["source"]
            selection = self.selections[chemcv]
            if chemcv in self.kwargs_per_cv:
                kwargs = self.kwargs_per_cv[chemcv]
            else:
                kwargs = {}

            try:
                chemcv_data = parsing_function(loaded[source_fn], selection, **kwargs)
                if isinstance(chemcv_data, dict):
                    data[chemcv] = nest_tuple_dict(chemcv_data)
                else:
                    data[chemcv] = chemcv_data
            except (KeyError, IndexError, TypeError) as e:
                print(f"Warning: Could not extract {chemcv}: {e}")

        super().update(data, step)

    def save(self, format: str = "hdf5", path: str | Path = "CHEMCV") -> None:
        self._metadata["active_chemcvs"] = self.active_chemcvs
        if format == "json":
            super().save_json(path)
        elif format == "hdf5":
            super().save_hdf5(path)
        else:
            raise ValueError("format must be 'json' or 'hdf5'")
    
    @classmethod
    def load(
        cls,
        format: str = "hdf5",
        path: str | Path = "CHEMCV",
        add_nb_traj: int | None = None,
        chemcvs: Optional[List[str]] = None
    ) -> "ChemCV":
        if format == "auto":
            for fmt in ["json", "hdf5"]:
                try:
                    if fmt == "json":
                        tf = TreeFrame.load_json(path, add_nb_traj)
                    elif fmt == "hdf5":
                        tf = TreeFrame.load_hdf5(path, add_nb_traj)
                    break
                except Exception:
                    continue          
        elif format == "json":
            tf = TreeFrame.load_json(path, add_nb_traj)
        elif format == "hdf5":
            tf = TreeFrame.load_hdf5(path, add_nb_traj)
        else:
            raise ValueError("format must be 'json', 'hdf5' or 'auto'")
        
        # Restore as ChemCV
        if not chemcvs:
            chemcvs = tf._metadata.get("active_chemcvs")
        chemcv = cls(chemcvs=chemcvs, nb_traj=tf.array_size, fill_incomplete=tf.fill_incomplete)
        chemcv._root = tf._root
        chemcv._step_count = tf._step_count
        
        return chemcv

# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.chdir("orca_parser")

    from ase import Atoms
    from ase.calculators.orca import ORCA

    chemcv = ChemCV(nb_traj=1, 
                    selections_per_type={"MO": [21,22], "atom": [0,1,5], "l": "p"},
                    kwargs_per_cv={"q_AO_Mulliken": {"fmt": ["atom", "l"]},
                                   "p_MOAO_Mulliken": {"fmt": ["MO", "atom", "l"]},
                                   "q_AO_Loewdin": {"fmt": ["atom", "l"]},
                                   "p_MOAO_Loewdin": {"fmt": ["MO", "atom", "l"]}})

    atoms = Atoms("CClH3Cl", positions=[( 0.000, 0.000, 0.000),
                                        ( 0.000, 0.000, 1.800),
                                        ( 0.000, 1.076, 0.000),
                                        ( 0.935,-0.540, 0.000),
                                        (-0.935,-0.540, 0.000),
                                        ( 0.000, 0.000,-2.300)])
    
    simpleinput, blocks = chemcv.get_orca_input()
    atoms.calc = ORCA(charge=-1, mult=1, directory="ORCA", 
                      orcasimpleinput=' '.join(["WB97X-D4 def2-TZVPD", simpleinput]), 
                      orcablocks='\n'.join(["%pal nprocs 32 end", blocks]))
    _ = atoms.get_potential_energy()

    chemcv.update()

    print(chemcv.summary())

    # chemcv.save()
    # chemcv.load()