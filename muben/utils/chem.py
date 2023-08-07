"""
# Author: Yinghao Li
# Modified: August 7th, 2023
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
           "atom_to_atom_ids",
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


ATOMIC_NUMBER_MAP = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96,
    'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
    'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114,
    'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}


def atom_to_atom_ids(atoms: list[str]) -> list[int]:
    """
    Convert a list of atoms strings to a list of atom ids

    Parameters
    ----------
    atoms: a list of atoms strings

    Returns
    -------
    atom ids
    """
    atoms_ids = [ATOMIC_NUMBER_MAP[atom] for atom in atoms]
    return atoms_ids
