"""
# Author: Yinghao Li
# Modified: February 27th, 2024
# ---------------------------------------
# Description: GROVER Dataset.

This module defines a Dataset class for handling molecular graph data
in the GROVER format.
"""

from tqdm.auto import tqdm
from multiprocessing import get_context

from ..dataset import pack_instances, Dataset as BaseDataset
from .molgraph import MolGraph, MolGraphAttrs


class Dataset(BaseDataset):
    def __init__(self) -> None:
        """
        Initialize the Dataset class.

        Attributes
        ----------
        _molecule_graphs : list or None
            A list of molecule graphs, initialized as None.
        """
        super().__init__()
        self._molecule_graphs = None

    def create_features(self, config: object) -> "Dataset":
        """
        Create data features for molecules using multiple processes.

        This method processes SMILES strings to convert them to molecular graph representations.
        The processing is performed in parallel using a specified number of worker processes.

        Parameters
        ----------
        config : object
            The configuration object with an attribute `num_preprocess_workers`.

        Returns
        -------
        Dataset
            Self instance of Dataset with features created.
        """

        with get_context("fork").Pool(config.num_preprocess_workers) as pool:
            self._molecule_graphs = [
                g
                for g in tqdm(
                    pool.imap(self.get_mol_attr, self.smiles),
                    total=len(self._smiles),
                )
            ]
        return self

    @staticmethod
    def get_mol_attr(smiles: str) -> MolGraphAttrs:
        """
        Convert a SMILES string to its molecular graph attributes representation.

        Parameters
        ----------
        smiles : str
            The SMILES string representation of the molecule.

        Returns
        -------
        MolGraphAttrs
            Molecular graph attributes of the given SMILES string.
        """
        return MolGraphAttrs().from_mol_graph(MolGraph(smiles))

    def get_instances(self) -> list[dict]:
        """
        Get data instances packed with molecule graphs, labels, and masks.

        Returns
        -------
        list of dicts
            Data instances containing molecule graphs, labels, and masks.
        """
        data_instances = pack_instances(
            molecule_graphs=self._molecule_graphs,
            lbs=self.lbs,
            masks=self.masks,
        )
        return data_instances
