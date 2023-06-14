import torch
import numpy as np
from typing import List, Union, Optional
from scipy.spatial import distance_matrix
from .dictionary import Dictionary

Matrix = Union[torch.Tensor, np.ndarray]  # to separate from list


class ProcessingPipeline:
    def __init__(self,
                 dictionary: Dictionary,
                 coordinate_padding: Optional[float] = 0.0,
                 max_atoms: Optional[int] = 256,
                 max_seq_len: Optional[int] = 512,
                 remove_hydrogen_flag: Optional[bool] = False,
                 remove_polar_hydrogen_flag: Optional[bool] = False):

        self._dictionary = dictionary
        self._coordinate_padding = coordinate_padding
        self._max_atoms = max_atoms
        self._max_seq_len = max_seq_len
        self._remove_hydrogen_flag = remove_hydrogen_flag
        self._remove_polar_hydrogen_flag = remove_polar_hydrogen_flag

    def process_instance(self, atoms: np.ndarray, coordinates: np.ndarray):
        atoms, coordinates = check_atom_types(atoms, coordinates)
        atoms, coordinates = remove_hydrogen(
            atoms, coordinates, self._remove_hydrogen_flag, self._remove_polar_hydrogen_flag
        )
        atoms, coordinates = cropping(atoms, coordinates, self._max_atoms)

        atoms = tokenize_atoms(atoms, self._dictionary, self._max_seq_len)
        atoms = prepend_and_append(atoms, self._dictionary.bos(), self._dictionary.eos())

        edge_types = get_edge_type(atoms, len(self._dictionary))

        coordinates = normalize_coordinates(coordinates)
        coordinates = from_numpy(coordinates)
        coordinates = prepend_and_append(coordinates, self._coordinate_padding, self._coordinate_padding)

        distances = get_distance(coordinates)

        return atoms, coordinates, distances, edge_types

    def process_training(self, atoms: List[str], coordinates: List[np.ndarray]):
        coordinates = conformer_sampling(coordinates)
        atoms = np.array(atoms)
        atoms, coordinates, distances, edge_types = self.process_instance(atoms, coordinates)

        return atoms.unsqueeze(0), coordinates.unsqueeze(0), distances.unsqueeze(0), edge_types.unsqueeze(0)

    def process_inference(self, atoms: List[str], coordinates: List[np.ndarray]):
        atoms = np.array(atoms)
        atoms_ = list()
        coordinates_ = list()
        distances_ = list()
        edge_types_ = list()

        for coord in coordinates:
            a, c, d, e = self.process_instance(atoms, coord)
            atoms_.append(a)
            coordinates_.append(c)
            distances_.append(d)
            edge_types_.append(e)

        atoms_ = torch.stack(atoms_)
        coordinates_ = torch.stack(coordinates_)
        distances_ = torch.stack(distances_)
        edge_types_ = torch.stack(edge_types_)

        return atoms_, coordinates_, distances_, edge_types_


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


def get_edge_type(atoms: Matrix, num_types: int):
    offset = atoms.view(-1, 1) * num_types + atoms.view(1, -1)
    return offset


def from_numpy(coordinates: np.ndarray):
    coordinates = torch.from_numpy(coordinates)
    return coordinates


def get_distance(coordinates: torch.Tensor):
    pos = coordinates.view(-1, 3).numpy()
    dist = distance_matrix(pos, pos).astype(np.float32)
    return torch.from_numpy(dist)
