import logging
import numpy as np

logger = logging.getLogger(__name__)


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
