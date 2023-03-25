import torch
import numpy as np
from typing import List, Union
from scipy.spatial import distance_matrix
from . import Dictionary

Matrix = Union[torch.Tensor, np.ndarray]  # to separate from list


def conformer_sampling(coordinates: List[Matrix]):
    """
    Sample one conformer from the generated conformers (#=11)

    Notice that this is a simplified version of the original implementation
    without considering the seeding difference within the *continue training* setup

    Returns
    -------
    Sampled coordinates
    """
    assert len(coordinates) == 11  # number of conformations defined in the paper

    size = len(coordinates)
    sample_idx = np.random.randint(size)
    coordinates = coordinates[sample_idx]
    return coordinates


def check_atom_types(atoms: Matrix, coordinates: Matrix):
    # for low rdkit version
    if len(atoms) != len(coordinates):
        min_len = min(atoms, coordinates)
        atoms = atoms[:min_len]
        coordinates = coordinates[:min_len]

    return atoms, coordinates


def tta():
    raise NotImplementedError("Should be implemented in other ways.")


def remove_hydrogen(atoms: Matrix, coordinates: Matrix, remove_hydrogen_flag=False, remove_polar_hydrogen_flag=False):

    if remove_hydrogen_flag:
        mask_hydrogen = atoms != "H"
        atoms = atoms[mask_hydrogen]
        coordinates = coordinates[mask_hydrogen]
    if not remove_hydrogen_flag and remove_polar_hydrogen_flag:
        end_idx = 0
        for i, atom in enumerate(atoms[::-1]):
            if atom != "H":
                break
            else:
                end_idx = i + 1
        if end_idx != 0:
            atoms = atoms[:-end_idx]
            coordinates = coordinates[:-end_idx]

    return atoms, coordinates


def cropping(atoms: Matrix, coordinates: Matrix, max_atoms=256):

    if max_atoms and len(atoms) > max_atoms:
        index = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = np.array(atoms)[index]
        coordinates = coordinates[index]
    return atoms, coordinates


def normalize_coordinates(coordinates: Matrix):
    coordinates = coordinates - coordinates.mean(axis=0)
    coordinates = coordinates.astype(np.float32)
    return coordinates


def tokenize_atoms(atoms: np.ndarray, dictionary: Dictionary, max_seq_len=512):
    assert max_seq_len > len(atoms) > 0
    tokens = torch.from_numpy(dictionary.vec_index(atoms)).long()
    return tokens


def prepend_and_append(item: torch.Tensor, prepend_value, append_value):
    """
    Parameters
    ----------
    item: could be both atom and coordinates, as torch.Tensor
    prepend_value: value to prepend
    append_value: value to append

    """
    item = torch.cat([torch.full_like(item[0], prepend_value).unsqueeze(0), item], dim=0)
    item = torch.cat([item, torch.full_like(item[0], append_value).unsqueeze(0)], dim=0)
    return item


def edit_edge_type(atoms: Matrix, num_types: int):
    offset = atoms.view(-1, 1) * num_types + atoms.view(1, -1)
    return offset


def from_numpy(coordinates: np.ndarray):
    coordinates = torch.from_numpy(coordinates)
    return coordinates


def get_distance(coordinates: np.ndarray):
    pos = coordinates.view(-1, 3).numpy()
    dist = distance_matrix(pos, pos).astype(np.float32)
    return torch.from_numpy(dist)
