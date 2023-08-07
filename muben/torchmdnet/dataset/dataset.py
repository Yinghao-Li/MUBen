"""
# Author: Yinghao Li
# Modified: August 7th, 2023
# ---------------------------------------
# Description: TorchMD-NET dataset.
"""


import os
import os.path as op
import ssl
import urllib
import zipfile
import lmdb
import pickle
import logging
from functools import partial
from multiprocessing import get_context
from tqdm.auto import tqdm

from muben.utils.chem import smiles_to_coords, smiles_to_atom_ids, atom_to_atom_ids
from muben.base.dataset import (
    pack_instances,
    Dataset as BaseDataset
)

logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._partition = None
        self._atoms = None
        self._cooridnates = None

    def prepare(self, config, partition, **kwargs):
        self._partition = partition
        super().prepare(config, partition, **kwargs)
        return self

    def create_features(self, config):
        """
        Create data features

        Returns
        -------
        self
        """

        self._atoms = list()
        self._cooridnates = list()

        # load feature if UniMol LMDB file exists else generate feature
        unimol_feature_path = op.join(config.unimol_feature_dir, f"{self._partition}.lmdb")

        if op.exists(unimol_feature_path):

            logger.info("Loading features form pre-processed Uni-Mol LMDB")

            unimol_atoms = list()

            env = lmdb.open(unimol_feature_path,
                            subdir=False,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False,
                            max_readers=256)
            txn = env.begin()
            keys = list(txn.cursor().iternext(values=False))
            for idx in tqdm(keys):
                datapoint_pickled = txn.get(idx)
                data = pickle.loads(datapoint_pickled)
                self._cooridnates.append(data['coordinates'][0])
                unimol_atoms.append(data['atoms'])

        else:
            logger.info("Generating 3D Coordinates.")
            s2c = partial(smiles_to_coords, n_conformer=1)
            with get_context('fork').Pool(config.num_preprocess_workers) as pool:
                for outputs in tqdm(pool.imap(s2c, self._smiles), total=len(self._smiles)):
                    _, coordinates = outputs
                    self._cooridnates.append(coordinates[0])
            unimol_atoms = [None] * len(self._cooridnates)

        logger.info("Generating atom ids")
        for smiles, coords, atoms in tqdm(zip(self._smiles, self._cooridnates, unimol_atoms), total=len(self._smiles)):
            atom_ids = smiles_to_atom_ids(smiles)
            if len(atom_ids) != len(coords):
                assert atoms is not None
                atom_ids = atom_to_atom_ids(atoms)
                assert len(atom_ids) == len(coords)
            self._atoms.append(atom_ids)

        return self

    def get_instances(self):
        data_instances = pack_instances(
            atoms=self._atoms, coords=self._cooridnates, lbs=self.lbs, masks=self.masks
        )

        return data_instances


def download_qm9(raw_dir):
    """
    Download qm9 raw data.
    Modified from `torch_geometric` implementation.

    Parameters
    ----------
    raw_dir

    Returns
    -------

    """
    raw_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip'
    file_path = download_url(raw_url, raw_dir)
    extract_zip(file_path, raw_dir)
    os.unlink(file_path)
    return None


def download_url(url: str, folder: str):
    r"""Downloads the content of a URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
    """

    filename = url.rpartition('/')[2]
    filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = op.join(folder, filename)

    if op.exists(path):  # pragma: no cover
        logger.info(f'Using existing file {filename}')
        return path

    logger.info(f'Downloading {url}')

    os.makedirs(folder, exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def extract_zip(path: str, folder: str):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (str): The path to the tar archive.
        folder (str): The folder.
    """
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)
