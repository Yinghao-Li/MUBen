import logging
import numpy as np
from typing import Union

from rdkit import Chem
from descriptastorus.descriptors import rdNormalizedDescriptors


Molecule = Union[str, Chem.Mol]
logger = logging.getLogger(__name__)


def rdkit_2d_features_normalized_generator(mol: Molecule) -> np.ndarray:
    """
    Generates RDKit 2D normalized features for a molecule.

    Parameters
    ----------
    mol: A molecule (i.e. either a SMILES string or an RDKit molecule).

    Returns
    -------
    An 1D numpy array containing the RDKit 2D normalized features.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smiles)[1:]
    # replace nan values
    features = np.where(np.isnan(features), 0, features)
    return features


# Fix nans in features
# if self.features is not None:
#     replace_token = 0
#     self.features = np.where(np.isnan(self.features), replace_token, self.features)

