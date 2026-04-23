import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import subprocess

class ChemCV(pd.DataFrame):
    """Manages Chemical Collective Variables parsed from ORCA outputs."""

    _metadata = ['active_chemcvs', 'columns_info', 'current_row', 'nb_atoms', 'nb_traj', 'AVAILABLE_CHEMCVS', 'columns_tuples', 'columns_info']
    
    def __init__(self, nb_traj: int, nb_atoms: Optional[int] = None, chemcvs: Optional[List[str]] = None, df: Optional[pd.DataFrame] = None, *args, **kwargs):
        """
        Initialize ChemCV dictionary.
        
        Args:
            nb_traj: Number of structures
            nb_atoms: Number of atoms
            chemcvs: List of ChemCVs to parse. If None and parse_all=True, parse all.
            parse_all: If True, parse all available ChemCVs by default
        """
        self.nb_traj = nb_traj
        if nb_atoms:
            self.nb_atoms = nb_atoms
        
        # Define available ChemCVs with their properties
        self.AVAILABLE_CHEMCVS = {
            "q_Mulliken": {"type": "atomic", "simpleinput": "MULLIKEN", "block": "", "source": self._orca_property, "parsingfunction": self._q_Mulliken},
            "q_Loewdin": {"type": "atomic", "simpleinput": "LOEWDIN", "block": "%method LOEWDIN_BONDORDERTHRESH 0.00 end", "source": self._orca_property, "parsingfunction": self._q_Loewdin},
            "q_Mayer": {"type": "atomic", "simpleinput": "MAYER", "block": "%method MAYER_BONDORDERTHRESH 0.00 end", "source": self._orca_property, "parsingfunction": self._q_Mayer},
            "v_Mayer": {"type": "atomic", "simpleinput": "MAYER", "block": "%method MAYER_BONDORDERTHRESH 0.00 end", "source": self._orca_property, "parsingfunction": self._v_Mayer},
            "q_Hirshfeld": {"type": "atomic", "simpleinput": "HIRSHFELD", "block": "", "source": self._orca_property, "parsingfunction": self._q_Hirshfeld},
            "q_MBIS": {"type": "atomic", "simpleinput": "MBIS", "block": "", "source": self._orca_property, "parsingfunction": self._q_MBIS},
            "npop_MBIS": {"type": "atomic", "simpleinput": "MBIS", "block": "", "source": self._orca_property, "parsingfunction": self._npop_MBIS},
            "sigma_MBIS": {"type": "atomic", "simpleinput": "MBIS", "block": "", "source": self._orca_property, "parsingfunction": self._sigma_MBIS},
            "q_CHELPG": {"type": "atomic", "simpleinput": "CHELPG", "block": "", "source": self._orca_property, "parsingfunction": self._q_CHELPG},
        }
        
        # Determine which ChemCVs to parse
        if chemcvs is None:
            self.active_chemcvs = list(self.AVAILABLE_CHEMCVS.keys())
        else:
            self.active_chemcvs = chemcvs
        
        # Validate requested ChemCVs
        invalid = set(self.active_chemcvs) - set(self.AVAILABLE_CHEMCVS.keys())
        if invalid:
            raise ValueError(f"Unknown ChemCVs: {invalid}")
        
        # Initialize DataFrame: either from provided df or create new
        if df is not None:
            if self.nb_traj > 0:
                additional_rows = pd.DataFrame(np.zeros((self.nb_traj, len(df.columns)), dtype=np.float32), columns=df.columns)
                df = pd.concat([df, additional_rows], ignore_index=True)
            super().__init__(df, *args, **kwargs)
            self._infer_from_dataframe()
            self.current_row = len(self)
        else:
            df = self._initialize_dataframe()
            super().__init__(df, *args, **kwargs)
            self.current_row = 0

    def __repr__(self) -> str:
        return (f"ChemCV(nb_traj={self.nb_traj}, nb_chemcvs={len(self.columns_tuples)}, active_chemcvs={self.active_chemcvs})")
        
    def __getitem__(self, key):
        """Get ChemCV data with flexible indexing based on number of indices.
        
        For multi-index ChemCVs:
            chemcv["multi_index_prop"]              # all indices -> DataFrame
            chemcv["multi_index_prop", indices]     # specified indices -> DataFrame
            chemcv["multi_index_prop", index]       # specified index -> 1D array
            
        For single-index ChemCVs -> 1D array:
            chemcv["single_index_prop"]
            chemcv["single_index_prop", index]
        """
        # Handle tuple indexing: (chemcv_name, index_key)
        if isinstance(key, tuple) and len(key) == 2:
            chemcv_name, index_key = key
            
            if chemcv_name not in self.active_chemcvs:
                raise KeyError(f"ChemCV '{chemcv_name}' not found")
            
            n_indices = self.columns_info[chemcv_name]
            if n_indices == 1 and key not in self.columns:
                raise ValueError(f"The index {index_key} is not valid for single-index ChemCV '{chemcv_name}'")
            
            # Get all rows for this ChemCV
            subset = super().__getitem__(chemcv_name)
            
            # Handle index_key
            if isinstance(index_key, slice):
                if index_key == slice(None):
                    # [:] - return all indices
                    return subset
                else:
                    # [start:stop] - return selected indices
                    cols = subset.columns
                    indices = [int(c[1]) for c in cols]
                    start = index_key.start if index_key.start is not None else 0
                    stop = index_key.stop if index_key.stop is not None else len(indices)
                    selected_cols = [cols[i] for i, idx in enumerate(indices) if start <= idx < stop]
                    return subset[selected_cols]
            
            elif isinstance(index_key, int):
                # Single index - return as 1D array
                return subset[str(index_key)].to_numpy()
            
            elif isinstance(index_key, (list, np.ndarray)):
                # List of indices - return selected columns
                selected_cols = [str(i) for i in index_key]
                return subset[selected_cols]
            
            else:
                raise TypeError(f"Index key must be int, slice, or list, got {type(index_key)}")
        
        # Handle single string key (ChemCV name)
        elif isinstance(key, str) and key in self.active_chemcvs:
            if key not in self.active_chemcvs:
                raise KeyError(f"ChemCV '{key}' not found")
            
            n_indices = self.columns_info[key]
            subset = super().__getitem__(key)
            
            if n_indices == 1:
                # Single-index ChemCV: return as 1D array
                return subset.to_numpy().squeeze()
            else:
                # Multi-index ChemCV: return as DataFrame
                return subset
        
        # Handle row indexing on the full ChemCV object
        else:
            # This handles chemcv[0], chemcv[0:10], chemcv[[0, 1]], chemcv[:]
            return super().__getitem__(key)
        
    #TODO: Test
    def __setitem__(self, key, value):
        """Set ChemCV data with flexible indexing based on number of indices.
        
        If value is a dict, indices will be inferred from the keys.

        For multi-index ChemCVs:
            chemcv["multi_index_prop"] = array_2d or dict
            chemcv["multi_index_prop", indices] = array_2d or dict
            chemcv["multi_index_prop", index] = array_1d or dict
            
        For single-index ChemCVs:
            chemcv["single_index_prop"] = array_1d or dict
            chemcv["single_index_prop", index] = array_1d or dict
        """
        # Handle tuple indexing: (chemcv_name, index_key)
        if isinstance(key, tuple) and len(key) == 2:
            chemcv_name, index_key = key
            
            if chemcv_name not in self.active_chemcvs:
                raise KeyError(f"ChemCV '{chemcv_name}' not found")
            
            n_indices = self.columns_info[chemcv_name]
            
            # Handle dict: infer indices from keys
            if isinstance(value, (dict, pd.DataFrame)):
                for idx, data in value.items():
                    self[chemcv_name, idx] = data
                return
            
            # Get the columns for this ChemCV
            chemcv_cols = [col for col in self.columns if col[0] == chemcv_name]
            
            if isinstance(index_key, slice):
                if index_key == slice(None):
                    # [:] - set all indices
                    if isinstance(value, np.ndarray):
                        if value.ndim == 1:
                            if n_indices != 1:
                                raise ValueError(f"Expected 2D array for multi-index ChemCV, got 1D")
                            super().__setitem__((chemcv_name, chemcv_cols[0]), value)
                        elif value.ndim == 2:
                            if value.shape[1] != n_indices:
                                raise ValueError(f"Expected shape (n_rows, {n_indices}), got {value.shape}")
                            for i, (_, idx) in enumerate(chemcv_cols):
                                super().__setitem__((chemcv_name, idx), value[:, i])
                    elif isinstance(value, pd.DataFrame):
                        super().__setitem__(chemcv_name, value)
                else:
                    # [start:stop] - set selected indices
                    indices = [int(col[1]) for col in chemcv_cols]
                    start = index_key.start if index_key.start is not None else 0
                    stop = index_key.stop if index_key.stop is not None else len(indices)
                    
                    if isinstance(value, np.ndarray):
                        if value.ndim != 2:
                            raise ValueError(f"Expected 2D array for slice indexing")
                        selected_cols = [col for col in chemcv_cols if start <= int(col[1]) < stop]
                        if value.shape[1] != len(selected_cols):
                            raise ValueError(f"Expected {len(selected_cols)} columns, got {value.shape[1]}")
                        for i, (_, idx) in enumerate(selected_cols):
                            super().__setitem__((chemcv_name, idx), value[:, i])
            
            elif isinstance(index_key, int):
                # Single index
                if isinstance(value, np.ndarray):
                    if value.ndim != 1:
                        raise ValueError(f"Expected 1D array for single index, got {value.ndim}D")
                    super().__setitem__((chemcv_name, str(index_key)), value)
                else:
                    raise TypeError(f"Unsupported value type: {type(value)}")
            
            elif isinstance(index_key, (list, np.ndarray)):
                # List of indices
                if isinstance(value, np.ndarray):
                    if value.ndim != 2:
                        raise ValueError(f"Expected 2D array for list indexing")
                    if value.shape[1] != len(index_key):
                        raise ValueError(f"Expected {len(index_key)} columns, got {value.shape[1]}")
                    for i, idx in enumerate(index_key):
                        super().__setitem__((chemcv_name, str(idx)), value[:, i])
                else:
                    raise TypeError(f"Unsupported value type: {type(value)}")
            
            else:
                raise TypeError(f"Index key must be int, slice, or list, got {type(index_key)}")
        
        # Handle single string key (ChemCV name)
        elif isinstance(key, str):
            if chemcv_name not in self.active_chemcvs:
                raise KeyError(f"ChemCV '{chemcv_name}' not found")
            
            n_indices = self.columns_info[key]
            
            # Handle dict: infer indices from keys
            if isinstance(value, dict):
                for idx, data in value.items():
                    self[key, idx] = data
                return
            
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    if n_indices != 1:
                        raise ValueError(f"1D array only valid for single-index ChemCV, got {n_indices} indices")
                    super().__setitem__(key, value)
                elif value.ndim == 2:
                    if value.shape[1] != n_indices:
                        raise ValueError(f"Expected shape (n_rows, {n_indices}), got {value.shape}")
                    chemcv_cols = [col for col in self.columns if col[0] == key]
                    for i, (_, idx) in enumerate(chemcv_cols):
                        super().__setitem__((key, idx), value[:, i])
                else:
                    raise ValueError(f"Array must be 1D or 2D, got {value.ndim}D")
            
            elif isinstance(value, pd.DataFrame):
                super().__setitem__(key, value)
            
            else:
                raise TypeError(f"Unsupported value type: {type(value)}")
        
        else:
            # Fall back to standard pandas setitem
            super().__setitem__(key, value)
    
    def __setattr__(self, name, value):
        """Handle attribute setting to avoid issues with pandas internals."""
        super().__setattr__(name, value)
  
    def _infer_from_dataframe(self):
        """Infer nb_atoms and columns_info from existing DataFrame."""
        self.columns_tuples = self.columns.tolist()
        self.columns_info = {chemcv: sum(1 for c in self.columns if c[0] == chemcv) for chemcv in self.active_chemcvs} 
        self.nb_traj = len(self)

    def _initialize_dataframe(self):
        """Initialize DataFrame with hierarchical columns."""
        self.columns_tuples = []
        self.columns_info = {}
        
        for chemcv in self.active_chemcvs:
            chemcv_type = self.AVAILABLE_CHEMCVS[chemcv]["type"]
            
            if chemcv_type == "atomic":
                if not hasattr(self, 'nb_atoms'):
                    raise ValueError("nb_atoms is required when creating a new ChemCV object")
                for index in range(self.nb_atoms):
                    self.columns_tuples.append((chemcv, f"{index}"))
                self.columns_info[chemcv] = self.nb_atoms
        
        # Create MultiIndex columns
        return pd.DataFrame(np.zeros((self.nb_traj, len(self.columns_tuples)), dtype=np.float32), columns=pd.MultiIndex.from_tuples(self.columns_tuples, names=["ChemCV", "Index"]))

    def _orca_property(self) -> json:
        subprocess.run("orca_2json ORCA/orca -property", shell=True, stdout=subprocess.DEVNULL)
        with open("ORCA/orca.property.json", 'r') as f:
            orca_property = json.load(f)
        return orca_property
    
    def _orca_gbw(self) -> json:
        subprocess.run(f"orca_2json ORCA/orca.gbw", shell=True, stdout=subprocess.DEVNULL)
        with open("ORCA/orca.gbw.json", 'r') as f:
            orca_gbw = json.load(f)
        return orca_gbw

    def _q_Mulliken(self, properties) -> np.ndarray:
        return np.array(properties["Geometries"][0]["Mulliken_Population_Analysis"][0]["AtomicCharges"], dtype=np.float32).T[0]

    def _q_Loewdin(self, properties) -> np.ndarray:
        return np.array(properties["Geometries"][0]["Loewdin_Population_Analysis"][0]["AtomicCharges"], dtype=np.float32).T[0]

    def _q_Mayer(self, properties) -> np.ndarray:
        return np.array(properties["Geometries"][0]["Mayer_Population_Analysis"][0]["QA"], dtype=np.float32).T[0]

    def _v_Mayer(self, properties) -> np.ndarray:
        return np.array(properties["Geometries"][0]["Mayer_Population_Analysis"][0]["VA"], dtype=np.float32).T[0]

    def _q_Hirshfeld(self, properties) -> np.ndarray:
        return np.array(properties["Geometries"][0]["Hirshfeld_Population_Analysis"][0]["AtomicCharges"], dtype=np.float32).T[0]

    def _q_MBIS(self, properties) -> np.ndarray:
        return np.array(properties["Geometries"][0]["MBIS_Population_Analysis"][0]["AtomicCharges"], dtype=np.float32).T[0]

    def _npop_MBIS(self, properties) -> np.ndarray:
        return np.array(properties["Geometries"][0]["MBIS_Population_Analysis"][0]["NPOPVAL"], dtype=np.float32).T[0]
        
    def _sigma_MBIS(self, properties) -> np.ndarray:
        return np.array(properties["Geometries"][0]["MBIS_Population_Analysis"][0]["SIGMAVAL"], dtype=np.float32).T[0]
        
    def _q_CHELPG(self, properties) -> np.ndarray:
        return np.array(properties["Geometries"][0]["CHELPG_Population_Analysis"][0]["AtomicCharges"], dtype=np.float32).T[0]

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
            simpleinput.add(self.AVAILABLE_CHEMCVS[chemcv]["simpleinput"])
            blocks.add(self.AVAILABLE_CHEMCVS[chemcv]["block"])

        simpleinput.discard("")
        blocks.discard("")
        
        simpleinput = ' '.join(simpleinput)
        blocks = '\n'.join(blocks)

        if simpleinput: simpleinput = ' ' + simpleinput
        if blocks: blocks = '\n' + blocks
        
        return simpleinput, blocks
    
    def update(self):
        """
        Parse and fill ChemCV data from ORCA properties.
        """
        # Check if we need to extend the DataFrame
        if self.current_row >= self.nb_traj:
            print(f"Warning: Extending DataFrame beyond initial nb_traj={self.nb_traj}. "
                  "It is more efficient to specify the correct number of trajectories beforehand.")
            new_row = pd.DataFrame(np.zeros((1, len(self.columns_tuples)), dtype=np.float32), columns=self.columns)
            self = pd.concat([self, new_row], ignore_index=True)
            self.nb_traj += 1
        
        for source in set(self.AVAILABLE_CHEMCVS[chemcv]["source"] for chemcv in self.active_chemcvs):
            if source == self._orca_property:
                properties = self._orca_property()
            elif source == self._orca_gbw:
                properties = self._orca_gbw()
            else:
                raise ValueError(f"Unknown source: {source}")
        
        col_idx = 0
        for chemcv, n_values in zip(self.active_chemcvs, self.columns_info.values()):
            parsing_function = self.AVAILABLE_CHEMCVS[chemcv]["parsingfunction"]
            try:
                data = parsing_function(properties)
                assert len(data) == n_values, f"Expected {n_values} values for {chemcv}, got {len(data)}"
                self.iloc[self.current_row, col_idx:col_idx + n_values] = data
            except (KeyError, IndexError, TypeError) as e:
                print(f"Warning: Could not extract {chemcv}: {e}")
            col_idx += n_values
        
        self.current_row += 1
    
    def save(self, filepath: str = "CHEMCV", format: str = "txt"):
        """
        Save ChemCV with optimal format for large datasets.
        
        Args:
            filepath: Path to save file (without extension)
            format: "hdf5" (best for millions of rows), "txt" (human-readable)
        """
        if format == "hdf5":
            # HDF5: Fast, compressed, supports incremental writes
            self.to_hdf(f"{filepath}.h5", key='chemcv', mode='w', complevel=9, complib='blosc')
        
        elif format == "txt":
            # Text: Human-readable but large file size
            col_width = 13
            with open(filepath, 'w') as f:
                # First header line: ChemCV names
                first_header = ""
                for chemcv, count in self.columns_info.items():
                    first_header += f" {chemcv:>{col_width-1}.{col_width-1}}{'':{col_width*(count-1)}}"
                f.write(first_header + "\n")
                
                # Second header line: indices
                second_header = ""
                for chemcv, index in self.columns_tuples:
                    second_header += f" {index:>{col_width-1}.{col_width-1}}"
                f.write(second_header + "\n")

                # Data lines (write in chunks to avoid memory overflow)
                chunk_size = 10000
                for start in range(0, len(self), chunk_size):
                    end = min(start + chunk_size, len(self))
                    for row in self.iloc[start:end].values:
                        data_line = ""
                        for val in row:
                            data_line += f" {val:>{col_width-1}.6f}"
                        f.write(data_line + "\n")
    
    @staticmethod
    def load(filepath: str = "CHEMCV", nb_additional_traj: int = 0, format: str = "auto") -> 'ChemCV':
        """
        Load ChemCV from file (auto-detects format).
        
        Args:
            filepath: Path to file (without extension)
            nb_atoms: Number of atoms
            nb_additional_traj: Additional rows to allocate
            format: "hdf5", "txt", or "auto"
        
        Returns:
            ChemCV instance
        """
        # Auto-detect format
        if format == "auto":
            if os.path.exists(f"{filepath}.h5"):
                format = "hdf5"
            else:
                format = "txt"
        
        if format == "hdf5":
            df = pd.read_hdf(f"{filepath}.h5", key='chemcv')

        else:  # txt
            col_width = 13
            with open(filepath, 'r') as f:
                first_header = f.readline()
                second_header = f.readline()

            chemcv_names = [first_header[i:i+col_width].strip() for i in range(0, len(first_header), col_width)]
            chemcv_indices = second_header.split()
            
            columns_tuples = []
            current_chemcv = None
            for name, index in zip(chemcv_names, chemcv_indices):
                if name:
                    current_chemcv = name
                columns_tuples.append((current_chemcv, index))
            
            data = np.loadtxt(filepath, skiprows=2, dtype=np.float32)

            df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns_tuples, names=["ChemCV", "Index"]))
        
        instance = ChemCV(nb_additional_traj, df=df)

        return instance