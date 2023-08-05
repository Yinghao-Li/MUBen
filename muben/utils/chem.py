"""
# Author: Yinghao Li
# Modified: August 4th, 2023
# ---------------------------------------
# Description: Molecular descriptors and features
"""


import warnings
import logging
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore')

logger = logging.getLogger(__name__)

__all__ = ["smiles_to_coords",
           "smiles_to_atom_ids",
           "rdkit_2d_features_normalized_generator",
           "morgan_binary_features_generator"]


def smiles_to_2d_coords(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    assert len(mol.GetAtoms()) == len(coordinates), f"2D coordinates shape is not align with {smiles}"
    return coordinates


def smiles_to_3d_coords(smiles, n_conformer):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    coordinate_list = []
    for seed in range(n_conformer):
        coordinates = list()
        try:
            # will random generate conformer with seed equal to -1. else fixed random seed.
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)  # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except Exception as e:
                    logger.warning(f"Failed to generate 3D, replace with 2D: {e}")
                    coordinates = smiles_to_2d_coords(smiles)

            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smiles)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)  # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except Exception as e:
                    logger.warning(f"Failed to generate 3D, replace with 2D: {e}")
                    coordinates = smiles_to_2d_coords(smiles)
        except Exception as e:
            logger.warning(f"Failed to generate 3D, replace with 2D: {e}")
            coordinates = smiles_to_2d_coords(smiles)

        assert len(mol.GetAtoms()) == len(coordinates), f"3D coordinates shape is not align with {smiles}"
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


def smiles_to_coords(smiles, n_conformer=10):
    """

    Parameters
    ----------
    smiles: the smile string
    n_conformer: conformer num, default (uni-mol) all==11, 10 3d + 1 2d

    Returns
    -------
    atoms and coordinates
    """

    mol = Chem.MolFromSmiles(smiles)
    if len(mol.GetAtoms()) > 400:
        coordinates = [smiles_to_2d_coords(smiles)] * (n_conformer + 1)
        logger.warning("atom num > 400, use 2D coords", smiles)
    else:
        coordinates = smiles_to_3d_coords(smiles, n_conformer)
        coordinates.append(smiles_to_2d_coords(smiles).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H
    return atoms, coordinates


def rdkit_2d_features_normalized_generator(mol) -> np.ndarray:
    """
    Generates RDKit 2D normalized features for a molecule.

    Parameters
    ----------
    mol: A molecule (i.e. either a SMILES string or an RDKit molecule).

    Returns
    -------
    An 1D numpy array containing the RDKit 2D normalized features.
    """
    from rdkit import Chem
    from descriptastorus.descriptors import rdNormalizedDescriptors

    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smiles)[1:]
    # replace nan values
    features = np.where(np.isnan(features), 0, features)
    return features


def morgan_binary_features_generator(mol, radius: int = 2, num_bits: int = 1024) -> np.ndarray:
    """
    Generates a binary Morgan fingerprint for a molecule.

    Parameters
    ----------
    mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    radius: Morgan fingerprint radius.
    num_bits: Number of bits in Morgan fingerprint.

    Returns
    -------
    An 1-D numpy array containing the binary Morgan fingerprint.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    # replace nan values
    features = np.where(np.isnan(features), 0, features)
    return features


def smiles_to_atom_ids(smiles: str) -> list[int]:
    """
    Convert smiles strings to a list of atom ids with H included

    Parameters
    ----------
    smiles: a smiles string

    Returns
    -------
    atom ids
    """

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    atom_ids = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return atom_ids
