"""
# Author: Yinghao Li
# Modified: February 28th, 2024
# ---------------------------------------
# Description: Pre-process the data
# Reference: https://github.com/dptech-corp/Uni-Mol/tree/main/unimol
"""

import torch
import numpy as np
from typing import List, Union, Optional
from scipy.spatial import distance_matrix
from .dictionary import DictionaryUniMol

Matrix = Union[torch.Tensor, np.ndarray]  # to separate from list


class ProcessingPipeline:
    """
    Data processing pipeline for molecular data.

    Attributes
    ----------
    _dictionary : Dictionary
        Dictionary for atom tokenization.
    _coordinate_padding : float, optional
        Padding value for coordinates, by default 0.0.
    _max_atoms : int, optional
        Maximum number of atoms to consider, by default 256.
    _max_seq_len : int, optional
        Maximum sequence length, by default 512.
    _remove_hydrogen_flag : bool, optional
        Flag to remove hydrogen, by default False.
    _remove_polar_hydrogen_flag : bool, optional
        Flag to remove polar hydrogen, by default False.
    """

    def __init__(
        self,
        dictionary: DictionaryUniMol,
        coordinate_padding: Optional[float] = 0.0,
        max_atoms: Optional[int] = 256,
        max_seq_len: Optional[int] = 512,
        remove_hydrogen_flag: Optional[bool] = False,
        remove_polar_hydrogen_flag: Optional[bool] = False,
    ):
        self._dictionary = dictionary
        self._coordinate_padding = coordinate_padding
        self._max_atoms = max_atoms
        self._max_seq_len = max_seq_len
        self._remove_hydrogen_flag = remove_hydrogen_flag
        self._remove_polar_hydrogen_flag = remove_polar_hydrogen_flag

    def process_instance(self, atoms: np.ndarray, coordinates: np.ndarray):
        """
        Process a single instance of atom and coordinates data.

        Parameters
        ----------
        atoms : np.ndarray
            Atom data.
        coordinates : np.ndarray
            Molecular coordinates.

        Returns
        -------
        tuple
            Updated atoms, normalized coordinates, distances, and edge types.
        """
        atoms, coordinates = check_atom_types(atoms, coordinates)
        atoms, coordinates = remove_hydrogen(
            atoms,
            coordinates,
            self._remove_hydrogen_flag,
            self._remove_polar_hydrogen_flag,
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
        """
        Instance processing during the training phase.

        Parameters
        ----------
        atoms : np.ndarray
            Atom data.
        coordinates : np.ndarray
            Molecular coordinates.

        Returns
        -------
        tuple
            Updated atoms, normalized coordinates, distances, and edge types.
        """
        coordinates = conformer_sampling(coordinates)
        atoms = np.array(atoms)
        atoms, coordinates, distances, edge_types = self.process_instance(atoms, coordinates)

        return (
            atoms.unsqueeze(0),
            coordinates.unsqueeze(0),
            distances.unsqueeze(0),
            edge_types.unsqueeze(0),
        )

    def process_inference(self, atoms: List[str], coordinates: List[np.ndarray]):
        """
        Instance processing during the inference phase.

        Parameters
        ----------
        atoms : np.ndarray
            Atom data.
        coordinates : np.ndarray
            Molecular coordinates.

        Returns
        -------
        tuple
            Updated atoms, normalized coordinates, distances, and edge types.
        """
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
    Sample one conformer from the generated conformers.

    Notice: This is a simplified version of the original implementation without considering
    the seeding difference within the *continue training* setup.

    Parameters
    ----------
    coordinates : List[Matrix]
        List of conformer coordinates.

    Returns
    -------
    Matrix
        Sampled coordinates.
    """
    assert len(coordinates) == 11  # number of conformations defined in the paper

    size = len(coordinates)
    sample_idx = np.random.randint(size)
    coordinates = coordinates[sample_idx]
    return coordinates


def check_atom_types(atoms: Matrix, coordinates: Matrix):
    """
    Ensure the atoms and coordinates have the same length.

    In the case of a mismatch in length (possibly due to versions of RDKit),
    the longer list is truncated to match the shorter one.

    Parameters
    ----------
    atoms : Matrix
        List or array of atom types.
    coordinates : Matrix
        List or array of atomic coordinates.

    Returns
    -------
    tuple
        A tuple containing the matched atom types and their corresponding coordinates.
    """
    # for low rdkit version
    if len(atoms) != len(coordinates):
        min_len = min(atoms, coordinates)
        atoms = atoms[:min_len]
        coordinates = coordinates[:min_len]

    return atoms, coordinates


def remove_hydrogen(
    atoms: Matrix,
    coordinates: Matrix,
    remove_hydrogen_flag=False,
    remove_polar_hydrogen_flag=False,
):
    """
    Remove hydrogen atoms based on flags.

    Parameters
    ----------
    atoms : Matrix
        Array or Tensor of atom types.
    coordinates : Matrix
        Array or Tensor of atomic coordinates.
    remove_hydrogen_flag : bool, optional
        If True, remove all hydrogen atoms. Default is False.
    remove_polar_hydrogen_flag : bool, optional
        If True and remove_hydrogen_flag is False, remove only polar hydrogen atoms. Default is False.

    Returns
    -------
    tuple
        A tuple containing atom types and coordinates after removal of hydrogens.
    """
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
    """
    Crop the atoms and coordinates to a maximum specified size.

    Parameters
    ----------
    atoms : Matrix
        Array or Tensor of atom types.
    coordinates : Matrix
        Array or Tensor of atomic coordinates.
    max_atoms : int, optional
        The maximum number of atoms to keep. Default is 256.

    Returns
    -------
    tuple
        A tuple containing cropped atom types and their corresponding coordinates.
    """

    if max_atoms and len(atoms) > max_atoms:
        index = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = np.array(atoms)[index]
        coordinates = coordinates[index]
    return atoms, coordinates


def normalize_coordinates(coordinates: Matrix):
    """
    Normalize the atomic coordinates by centering them around the mean.

    Parameters
    ----------
    coordinates : Matrix
        Array or Tensor of atomic coordinates.

    Returns
    -------
    Matrix
        The normalized coordinates.
    """
    coordinates = coordinates - coordinates.mean(axis=0)
    coordinates = coordinates.astype(np.float32)
    return coordinates


def tokenize_atoms(atoms: np.ndarray, dictionary: DictionaryUniMol, max_seq_len=512):
    """
    Convert atom names to tokens using a provided dictionary.

    Parameters
    ----------
    atoms : np.ndarray
        Array of atom names.
    dictionary : Dictionary
        The dictionary used for tokenization.
    max_seq_len : int, optional
        The maximum sequence length. Default is 512.

    Returns
    -------
    torch.Tensor
        The tokenized atoms.
    """
    assert max_seq_len > len(atoms) > 0
    tokens = torch.from_numpy(dictionary.vec_index(atoms)).long()
    return tokens


def prepend_and_append(item: torch.Tensor, prepend_value, append_value):
    """
    Prepend and append values to a torch.Tensor.

    Given an input tensor, this function will prepend the tensor with a given value and append
    another given value at the end of the tensor.

    Parameters
    ----------
    item : torch.Tensor
        The input tensor to which values need to be prepended and appended.
    prepend_value : int or float
        The value to be prepended to the tensor.
    append_value : int or float
        The value to be appended to the tensor.

    Returns
    -------
    torch.Tensor
        The modified tensor after prepending and appending the specified values.
    """
    item = torch.cat([torch.full_like(item[0], prepend_value).unsqueeze(0), item], dim=0)
    item = torch.cat([item, torch.full_like(item[0], append_value).unsqueeze(0)], dim=0)
    return item


def get_edge_type(atoms: Matrix, num_types: int):
    """
    Generate the edge type for a given set of atoms.

    Parameters
    ----------
    atoms : Matrix
        The input tensor/array representing atom types.
    num_types : int
        The number of unique atom types.

    Returns
    -------
    torch.Tensor
        A tensor representing the edge types between pairs of atoms.
    """
    offset = atoms.view(-1, 1) * num_types + atoms.view(1, -1)
    return offset


def from_numpy(coordinates: np.ndarray):
    """
    Convert a numpy array to a torch tensor.

    Parameters
    ----------
    coordinates : np.ndarray
        The numpy array to be converted.

    Returns
    -------
    torch.Tensor
        The corresponding torch tensor.
    """
    coordinates = torch.from_numpy(coordinates)
    return coordinates


def get_distance(coordinates: torch.Tensor):
    """
    Calculate the pairwise distances between points in a set of coordinates.

    Parameters
    ----------
    coordinates : torch.Tensor
        A tensor of shape (N, 3) where N is the number of points and each point is represented in 3D space.

    Returns
    -------
    torch.Tensor
        A tensor of shape (N, N) representing the pairwise distances.
    """
    pos = coordinates.view(-1, 3).numpy()
    dist = distance_matrix(pos, pos).astype(np.float32)
    return torch.from_numpy(dist)
