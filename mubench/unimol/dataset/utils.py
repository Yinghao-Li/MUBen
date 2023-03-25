import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm.auto import tqdm
from rdkit.Chem import AllChem
from rdkit import RDLogger
from multiprocessing import Pool
import warnings

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore')

logger = logging.getLogger(__name__)

__all__ = ["smiles_to_coords"]


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


def smiles_to_coords(smiles):
    n_conformer = 10  # conformer num,all==11, 10 3d + 1 2d

    mol = Chem.MolFromSmiles(smiles)
    if len(mol.GetAtoms()) > 400:
        coordinates = [smiles_to_2d_coords(smiles)] * (n_conformer + 1)
        logger.warning("atom num > 400, use 2D coords", smiles)
    else:
        coordinates = smiles_to_3d_coords(smiles, n_conformer)
        coordinates.append(smiles_to_2d_coords(smiles).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H
    return atoms, coordinates, mol
    # return {'atoms': atoms, 'coordinates': coordinates, 'mol': mol, 'smi': smiles, 'target': target}


def load_csv(file_path, n_threads=16):

    df = pd.read_csv(file_path)
    content_list = zip(*[df[c].values.tolist() for c in df])
    with Pool(n_threads) as pool:
        i = 0
        outputs = list()
        for inner_output in tqdm(pool.imap(smiles_to_coords, content_list), total=len(df)):
            if inner_output is not None:
                outputs.append(inner_output)
                i += 1
