"""
TreeFrame.py
============
A tree-structured container where every leaf holds a 1-D float numpy array
of a fixed length (``array_size``).  Designed for iterative, step-by-step
population: each call to ``update(data, step)`` writes one scalar per leaf
into the pre-allocated arrays at the given step index.

Concepts
--------
* The **tree structure** (which paths exist as leaves) is defined by the first
  ``update`` call.  After that, the set of leaves is the reference.
* Every leaf array is always of length ``array_size``, pre-filled with zeros.
  A scalar written at step ``i`` lands at ``array[i]``.
* When ``array_size`` steps have been filled and more updates arrive, arrays
  grow automatically by appending (one warning is emitted).

fill_incomplete behaviour (applied on every update)
----------------------------------------------------
Leaf in TreeFrame but NOT in data
  ``'zeros'`` → keep the leaf; that step stays zero. Nothing to do.
  ``'drop'``  → drop the leaf (and any empty ancestor nodes). Warns.

Leaf in data but NOT in TreeFrame
  ``'zeros'`` → create a zero-initialised leaf, write the value. Warns.
  ``'drop'``  → ignore silently. Nothing to do.

Quick example
-------------
>>> tf = TreeFrame(array_size=1000)
>>> # First update defines the tree structure
>>> tf.update({"sensors": {"temp": 22.1, "pressure": 1013.0},
...            "model":   {"loss": 0.42}}, step=0)
>>> # Subsequent updates fill the next steps
>>> for i in range(1, 1000):
...     tf.update({"sensors": {"temp": 22.1 + i*0.01,
...                            "pressure": 1013.0 - i*0.1},
...                "model":   {"loss": 0.42 / (i + 1)}}, step=i)
>>> arr, labels = tf.to_numpy()          # shape (3, 1000)
>>> tf.save_json("data.json")
>>> tf.save_hdf5("data.h5")
>>> tf2 = TreeFrame.load_json("data.json", new_array_size=2000)
>>> tf3 = TreeFrame.load_hdf5("data.h5",  new_array_size=2000)
"""

import json
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Internal node
# ---------------------------------------------------------------------------

class _Node:
    """
    An internal node in the tree.  Children are either other _Node objects
    or 1-D numpy arrays (leaves).
    """

    def __init__(self) -> None:
        self._children: dict[str, "_Node | np.ndarray"] = {}

    # ------------------------------------------------------------------
    # dict-like access
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> "_Node | np.ndarray":
        if key not in self._children:
            # auto-vivification: create a new empty node on first access
            self._children[key] = _Node()
        return self._children[key]

    def __setitem__(self, key: str, value: "_Node | np.ndarray") -> None:
        if isinstance(value, (list, tuple)):
            value = np.asarray(value, dtype=float)
        self._children[key] = value

    def __delitem__(self, key: str) -> None:
        del self._children[key]

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __iter__(self) -> Iterator[str]:
        return iter(self._children)

    def keys(self):
        return self._children.keys()

    def items(self):
        return self._children.items()

    def values(self):
        return self._children.values()

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------

    def is_leaf_parent(self) -> bool:
        """True if every child is a numpy array (i.e. this node holds leaves)."""
        return all(isinstance(v, np.ndarray) for v in self._children.values())

    def leaves(self) -> Iterator[tuple[tuple[str, ...], np.ndarray]]:
        """Yield (path_tuple, array) for every leaf in this sub-tree."""
        for key, child in self._children.items():
            if isinstance(child, np.ndarray):
                yield (key,), child
            else:
                for sub_path, arr in child.leaves():
                    yield (key,) + sub_path, arr

    def depth(self) -> int:
        """Maximum depth below this node (0 if all children are arrays)."""
        if not self._children:
            return 0
        depths = []
        for child in self._children.values():
            if isinstance(child, np.ndarray):
                depths.append(0)
            else:
                depths.append(1 + child.depth())
        return max(depths)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_nested_dict(self) -> dict:
        out = {}
        for key, child in self._children.items():
            if isinstance(child, np.ndarray):
                out[key] = child.tolist()
            else:
                out[key] = child.to_nested_dict()
        return out
    
    def to_dict(self) -> dict["str | int | tuple", np.ndarray]:
        dict = {}
        leaves = list(self.leaves())
        for path, array in leaves:
            key = '.'.join(path)
            dict[key] = array
        return dict
        
    
    def to_numpy(self, dtype: type = float) -> tuple[np.ndarray, list[tuple[str, ...]]]:
        """
        Return ``matrix`` with shape ``(n_leaves, array_size)``

        Raises ``ValueError`` if leaf arrays have inconsistent lengths.
        """
        leaves = list(self.leaves())
        if not leaves:
            return np.empty((0, 0), dtype=dtype), []

        _, arrays = zip(*leaves)
        lengths = {len(a) for a in arrays}
        if len(lengths) > 1:
            raise ValueError(
                f"Leaf arrays have inconsistent lengths: {lengths}. "
                "This should not happen under normal usage; "
                "check that all leaves were updated the same number of times."
            )
        matrix = np.vstack([a[np.newaxis, :] for a in arrays]).astype(dtype)
        return matrix

    @classmethod
    def from_dict(cls, d: dict) -> "_Node":
        node = cls()
        for key, value in d.items():
            if isinstance(value, list):
                node._children[key] = np.array(value, dtype=float)
            else:
                node._children[key] = cls.from_dict(value)
        return node


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _flatten(d: dict, parent: tuple = ()) -> dict[tuple[str, ...], float]:
    """Recursively flatten a nested dict of scalars to ``{path_tuple: scalar}``."""
    flat: dict = {}
    for key, value in d.items():
        path = parent + (key,)
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _get_leaf(root: _Node, path: tuple[str, ...]) -> "np.ndarray | None":
    """Return the leaf array at *path*, or None if the path does not exist."""
    node = root
    for k in path:
        if not isinstance(node, _Node) or k not in node:
            return None
        node = node[k]
    return node if isinstance(node, np.ndarray) else None


def _create_leaf(root: _Node, path: tuple[str, ...], size: int) -> np.ndarray:
    """Create intermediate nodes and a zero-filled leaf array at *path*."""
    node = root
    for k in path[:-1]:
        if k not in node:
            node[k] = _Node()
        node = node[k]
    arr = np.zeros(size, dtype=float)
    node[path[-1]] = arr
    return arr


def _delete_leaf(root: _Node, path: tuple[str, ...]) -> None:
    """Delete the leaf at *path* and prune any resulting empty ancestor nodes."""
    trail: list[tuple[_Node, str]] = []
    node = root
    for k in path[:-1]:
        trail.append((node, k))
        node = node[k]
    del node[path[-1]]
    for parent, key in reversed(trail):
        child = parent[key]
        if isinstance(child, _Node) and not child._children:
            del parent[key]
        else:
            break


def _pad_all_leaves(node: _Node, add_size: int) -> None:
    """Pad every leaf array in *node*'s sub-tree to *new_size* with zeros."""
    for key, child in node._children.items():
        if isinstance(child, np.ndarray):
            node[key] = np.pad(child, (0, add_size))
        else:
            _pad_all_leaves(child, add_size)


# ---------------------------------------------------------------------------
# Public TreeFrame
# ---------------------------------------------------------------------------

class TreeFrame:
    """
    Tree-structured, step-by-step accumulator of 1-D float arrays.

    Parameters
    ----------
    array_size : int
        Number of time-steps / samples to pre-allocate per leaf.
    fill_incomplete : {'zeros', 'drop'}
        How to handle mismatches between the stored leaves and ``data``
        passed to ``update``.  See module docstring for full semantics.
    """

    def __init__(self, array_size: int = 0, fill_incomplete: str = "zeros") -> None:
        if fill_incomplete not in ("zeros", "drop"):
            raise ValueError("fill_incomplete must be 'zeros' or 'drop'")
        self.array_size: int = array_size
        self.fill_incomplete: str = fill_incomplete
        self._root: _Node = _Node()
        self._step_count: int = 0         # number of steps written so far
        self._overflow_warned: bool = False  # overflow warning emitted at most once
        self._metadata: dict = {}  # Store arbitrary metadata

    # ------------------------------------------------------------------
    # Direct dict-style access
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """
        Support both single-key and tuple-key access.

        tf["a"]           → _Node or array
        tf["a", "b", "c"] → _Node or array (equivalent to tf["a"]["b"]["c"])
        """
        if isinstance(key, tuple):
            node = self._root
            for k in key:
                node = node[k]
            return node
        return self._root[key]

    def __setitem__(self, key, value) -> None:
        if isinstance(key, tuple):
            *path, last = key
            node = self._root
            for k in path:
                node = node[k]
            node[last] = value
        else:
            self._root[key] = value

    def __contains__(self, key) -> bool:
        return key in self._root

    def __iter__(self) -> Iterator[str]:
        return iter(self._root)

    def keys(self):
        return self._root.keys()

    def items(self):
        return self._root.items()

    # ------------------------------------------------------------------
    # Tree information
    # ------------------------------------------------------------------

    def depth(self) -> int:
        """Current maximum depth (number of key levels above the arrays)."""
        return self._root.depth()

    def n_leaves(self) -> int:
        return sum(1 for _ in self._root.leaves())

    def leaf_paths(self) -> list[tuple[str, ...]]:
        return [path for path, _ in self._root.leaves()]

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, data: dict, step: int | None = None) -> None:
        """
        Write one scalar value per leaf at the given *step* index.

        Parameters
        ----------
        data : dict
            Nested or flat dict whose terminal values are scalars.

        First call
        ----------
        The tree structure is initialised from *data*: a zero array of length
        ``array_size`` is created for every leaf, then the scalar at *step*
        is written.

        Overflow
        --------
        If ``step >= array_size``, arrays are extended by one element.
        A single warning is emitted (subsequent overflows are silent).
        """
        flat_data: dict[tuple[str, ...], float] = _flatten(data)
        is_first = (self.n_leaves() == 0)

        if step is None:
            step = self._step_count

        # ---- overflow warning (emit only once) ---------------------------
        if step >= self.array_size:
            if not self._overflow_warned:
                warnings.warn(
                    f"The array_size={self.array_size} has been reached. "
                    "Leaf arrays will be extended by appending from now on. "
                    "Consider using a larger array_size to avoid reallocation.",
                    stacklevel=2,
                )
                self._overflow_warned = True
            _pad_all_leaves(self._root, step-self.array_size+1)
            self.array_size = step+1

        # ---- first call: build tree from data ----------------------------
        if is_first:
            for path, value in flat_data.items():
                arr = _create_leaf(self._root, path, self.array_size)
                arr[step] = float(value)
            if step == self._step_count:
                self._step_count += 1
            return

        # ---- subsequent calls: reconcile leaves --------------------------
        tree_paths = set(self.leaf_paths())
        data_paths = set(flat_data.keys())

        # In tree but missing from data
        for path in tree_paths - data_paths:
            if self.fill_incomplete == "drop":
                warnings.warn(f"Leaf {path} is present in the TreeFrame but not in data. ", stacklevel=2)
                _delete_leaf(self._root, path)
            # 'zeros': do nothing, slot stays zero

        # In data but absent from tree
        for path in data_paths - tree_paths:
            if self.fill_incomplete == "zeros":
                warnings.warn(f"Leaf {path} is present in data but not in the TreeFrame. ", stacklevel=2)
                arr = _create_leaf(self._root, path, self.array_size)
                arr[step] = float(flat_data[path])
            # 'drop': ignore silently

        # Write values for leaves present in both
        for path in data_paths & tree_paths:
            arr = _get_leaf(self._root, path)
            if arr is not None:
                arr[step] = float(flat_data[path])

        if step == self._step_count:
            self._step_count += 1
        
    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return _Node.to_dict(self._root)

    def to_numpy(self, dtype: type = float) -> tuple[np.ndarray, list[tuple[str, ...]]]:
        return _Node.to_numpy(self._root, dtype)

    # ------------------------------------------------------------------
    # Save - JSON (human-readable)
    # ------------------------------------------------------------------

    def save_json(self, path: str | Path) -> None:
        """Save the TreeFrame to a human-readable JSON file."""
        payload = {
            "_meta": {
                "array_size":     self.array_size,
                "fill_incomplete": self.fill_incomplete,
                "step_count":     self._step_count,
                **self._metadata,  # Include any custom metadata
            },
            "data": self._root.to_nested_dict(),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------------
    # Load - JSON
    # ------------------------------------------------------------------

    @classmethod
    def load_json(cls, path: str | Path, new_array_size: int | None = None) -> "TreeFrame":
        """
        Load a TreeFrame saved with ``save_json``.

        Parameters
        ----------
        new_array_size : int or None
            If given, every leaf is zero-padded to this length so you can
            resume filling the TreeFrame.  Must be >= the stored array_size.
        """
        payload = json.loads(Path(path).read_text())
        meta = payload.get("_meta", {})
        stored_size = meta.get("array_size", 0)

        if new_array_size is not None and new_array_size < stored_size:
            raise ValueError(
                f"new_array_size={new_array_size} is smaller than the stored "
                f"array_size={stored_size}."
            )

        target_size = new_array_size if new_array_size is not None else stored_size
        tf = cls(array_size=target_size, fill_incomplete=meta.get("fill_incomplete", "zeros"))
        tf._root = _Node.from_dict(payload["data"])
        tf._step_count = meta.get("step_count", 0)

        # Restore custom metadata (everything except reserved keys)
        reserved = {"array_size", "fill_incomplete", "step_count"}
        tf._metadata = {k: v for k, v in meta.items() if k not in reserved}

        if new_array_size is not None and new_array_size > stored_size:
            _pad_all_leaves(tf._root, new_array_size)

        return tf

    # ------------------------------------------------------------------
    # Save - HDF5 (compact)
    # ------------------------------------------------------------------

    def save_hdf5(self, path: str | Path) -> None:
        """
        Save to an HDF5 file.  Each leaf becomes a gzip-compressed dataset
        whose path mirrors the tree (e.g. ``/sensors/temp``).

        Requires ``h5py``  (``pip install h5py``).
        """
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required for HDF5 support.") from exc

        with h5py.File(path, "w") as f:
            f.attrs["array_size"]     = self.array_size
            f.attrs["fill_incomplete"] = self.fill_incomplete
            f.attrs["step_count"]     = self._step_count
            if self._metadata:
                f.attrs["_metadata_json"] = json.dumps(self._metadata)
            for path_tuple, arr in self._root.leaves():
                path_tuple = [str(path_key) for path_key in path_tuple]
                f.create_dataset("/" + "/".join(path_tuple), data=arr, compression="gzip")

    # ------------------------------------------------------------------
    # Load - HDF5
    # ------------------------------------------------------------------

    @classmethod
    def load_hdf5(cls, path: str | Path, new_array_size: int | None = None) -> "TreeFrame":
        """
        Load a TreeFrame saved with ``save_hdf5``.

        Parameters
        ----------
        new_array_size : int or None
            If given, every leaf is zero-padded to this length.
            Must be >= the stored array_size.
        """
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required for HDF5 support.") from exc

        with h5py.File(path, "r") as f:
            stored_size     = int(f.attrs["array_size"])
            fill_incomplete  = str(f.attrs["fill_incomplete"])
            step_count      = int(f.attrs.get("step_count", 0))

            if new_array_size is not None and new_array_size < stored_size:
                raise ValueError(
                    f"new_array_size={new_array_size} is smaller than the stored "
                    f"array_size={stored_size}."
                )

            target_size = new_array_size if new_array_size is not None else stored_size
            tf = cls(array_size=target_size, fill_incomplete=fill_incomplete)
            tf._step_count = step_count

            if "_metadata_json" in f.attrs:
                tf._metadata = json.loads(str(f.attrs["_metadata_json"]))

            def _visit(hdf5_name: str, obj) -> None:
                if not isinstance(obj, h5py.Dataset):
                    return
                keys = hdf5_name.strip("/").split("/")
                node = tf._root
                for k in keys[:-1]:
                    if k not in node:
                        node[k] = _Node()
                    node = node[k]
                arr = np.array(obj, dtype=float)
                if new_array_size is not None and new_array_size > len(arr):
                    arr = np.pad(arr, (0, new_array_size - len(arr)))
                node[keys[-1]] = arr

            f.visititems(_visit)

        return tf

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"TreeFrame(n_leaves={self.n_leaves()}, array_size={self.array_size}, step_count={self._step_count}')"

    def summary(self) -> str:
        """Return a text tree showing structure and array shapes."""
        lines = [repr(self)]
        self._summarise_node(self._root, lines, prefix="", is_last=True)
        return "\n".join(lines)

    def _summarise_node(self, node: _Node, lines: list[str], prefix: str, is_last: bool) -> None:
        children = list(node._children.items())
        for i, (key, child) in enumerate(children):
            connector = "└── " if i == len(children) - 1 else "├── "
            if isinstance(child, np.ndarray):
                if self.array_size >2:
                    lines.append(f"{prefix}{connector}{key}  [{child[0]:f}, {child[1]:f}, {child[2]:f}, ...]")
                else:
                    lines.append(f"{prefix}{connector}{key}  {child}")
            else:
                lines.append(f"{prefix}{connector}{key}")
                extension = "    " if i == len(children) - 1 else "│   "
                self._summarise_node(child, lines, prefix + extension, i == len(children) - 1)



# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sep = "=" * 60

    # ------------------------------------------------------------------
    print(sep)
    print("1. Basic step-by-step fill")
    print(sep)

    tf = TreeFrame(array_size=8)
    for i in range(8):
        tf.update({"a": {"one": rng.random(), "two": rng.random()}, "b": rng.random()})

    print(tf.summary())

    arr = tf.to_numpy()
    print(f"to_numpy -> shape {arr.shape}")
    print(arr)

    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("2. Extra leaf in data  (fill_incomplete='zeros') → added + warn")
    print(sep)

    tf2 = TreeFrame(array_size=2, fill_incomplete="zeros")
    tf2.update({"a": rng.random(), "b": rng.random()})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tf2.update({"a": rng.random(), "b": rng.random(), "c": rng.random()})

    for w in caught:
        print(w.message)

    print(tf2.summary())

    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("3. Leaf missing from data  (fill_incomplete='drop') → dropped + warn")
    print(sep)

    tf3 = TreeFrame(array_size=2, fill_incomplete="drop")
    tf3.update({"a": rng.random(), "b": rng.random()})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tf3.update({"a": rng.random()}, step=1)

    for w in caught:
        print(w.message)

    print(tf3.summary())

    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("4. Overflow beyond array_size → single warning, arrays extend")
    print(sep)

    tf4 = TreeFrame(array_size=3, fill_incomplete="zeros")
    for i in range(3):
        tf4.update({"x": rng.random()})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tf4.update({"x": rng.random()})   # overflow → warn once
        tf4.update({"x": rng.random()})   # no second warning

    for w in caught:
        print(w.message)

    print(tf4.summary())
    print("x:", tf4["x"])

    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("5. JSON round-trip + new_array_size")
    print(sep)

    tf.save_json("/tmp/tf_test.json")
    tf_j = TreeFrame.load_json("/tmp/tf_test.json", new_array_size=16)
    print(tf_j.summary())

    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("6. HDF5 round-trip + new_array_size")
    print(sep)

    try:
        tf.save_hdf5("/tmp/tf_test.h5")
        tf_h = TreeFrame.load_hdf5("/tmp/tf_test.h5", new_array_size=16)
        print(tf_h.summary())
    except ImportError:
        print("(h5py not installed - skipping)")
