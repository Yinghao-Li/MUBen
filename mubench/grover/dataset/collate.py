import torch
import logging
import numpy as np

from mubench.utils.data import (
    Batch,
    unpack_instances
)
from .molgraph import BatchMolGraph
from .molgraph import MolGraph, MolGraphAttrs

logger = logging.getLogger(__name__)


class Collator:

    def __init__(self, config):
        self._task = config.task_type
        self._lbs_type = torch.float \
            if config.task_type == 'regression' or not config.binary_classification_with_softmax \
            else torch.long
        self._bond_drop_rate = config.bond_drop_rate

    def __call__(self, instance_list: list, *args, **kwargs) -> Batch:
        """
        function call

        Parameters
        ----------
        instance_list: a list of instance

        Returns
        -------
        a Batch of instances
        """
        smiles, lbs, masks = unpack_instances(instance_list)

        molecule_graphs = [MolGraphAttrs().from_mol_graph(MolGraph(s, self._bond_drop_rate)) for s in smiles]
        molecule_graphs_batch = BatchMolGraph(molecule_graphs)
        lbs_batch = torch.from_numpy(np.stack(lbs)).to(self._lbs_type)
        masks_batch = torch.from_numpy(np.stack(masks))

        return Batch(molecule_graphs=molecule_graphs_batch, lbs=lbs_batch, masks=masks_batch)
