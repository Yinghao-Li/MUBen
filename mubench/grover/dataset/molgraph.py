"""
The data structure of Molecules.
This implementation is adapted from
https://github.com/chemprop/chemprop/blob/master/chemprop/features/featurization.py
"""
from typing import List, Tuple, Union

import logging
import numpy as np
import torch
from dataclasses import dataclass
from rdkit import Chem
from mubench.utils.data import Batch

logger = logging.getLogger(__name__)

MAX_ATOMIC_NUM = 100

ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


def get_atom_fdim() -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM + 18


def get_bond_fdim() -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    if min(choices) < 0:
        index = value
    else:
        index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        """
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond

        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(smiles)

        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
            "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
            "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

        self.hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
        self.hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
        self.acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
        self.basic_match = sum(mol.GetSubstructMatches(self.basic), ())
        self.ring_info = mol.GetRingInfo()

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = mol.GetNumAtoms()

        # Get atom features
        for _, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(self.atom_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = self.bond_features(bond)

                # Always treat the bond as directed.
                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2

    def atom_features(self, atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
        """
        Builds a feature vector for an atom.

        :param atom: An RDKit atom.
        :return: A list containing the atom features.
        """
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
            onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
            onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
            onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01]
        atom_idx = atom.GetIdx()
        features = features + \
            onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
            [atom_idx in self.hydrogen_acceptor_match] + \
            [atom_idx in self.hydrogen_donor_match] + \
            [atom_idx in self.acidic_match] + \
            [atom_idx in self.basic_match] + \
            [self.ring_info.IsAtomInRingOfSize(atom_idx, 3),
             self.ring_info.IsAtomInRingOfSize(atom_idx, 4),
             self.ring_info.IsAtomInRingOfSize(atom_idx, 5),
             self.ring_info.IsAtomInRingOfSize(atom_idx, 6),
             self.ring_info.IsAtomInRingOfSize(atom_idx, 7),
             self.ring_info.IsAtomInRingOfSize(atom_idx, 8)]
        return features

    @staticmethod
    def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
        """
        Builds a feature vector for a bond.

        :param bond: A RDKit bond.
        :return: A list containing the bond features.
        """

        if bond is None:
            fbond = [1] + [0] * (BOND_FDIM - 1)
        else:
            bt = bond.GetBondType()
            fbond = [
                0,  # bond is not None
                bt == Chem.rdchem.BondType.SINGLE,
                bt == Chem.rdchem.BondType.DOUBLE,
                bt == Chem.rdchem.BondType.TRIPLE,
                bt == Chem.rdchem.BondType.AROMATIC,
                (bond.GetIsConjugated() if bt is not None else 0),
                (bond.IsInRing() if bt is not None else 0)
            ]
            fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
        return fbond


@dataclass
class MolGraphAttrs:
    n_atoms = None
    n_bonds = None
    f_atoms = None
    f_bonds = None
    a2b = None
    b2a = None
    b2revb = None

    def from_mol_graph(self, graph: MolGraph):
        attrs = ("n_atoms", "n_bonds", "f_atoms", "f_bonds", "a2b", "b2a", "b2revb")
        for attr in attrs:
            setattr(self, attr, getattr(graph, attr))
        return self


class BatchMolGraph(Batch):
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraphAttrs]):
        super().__init__()

        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim() + self.atom_fdim

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond

        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))

        self.f_atoms = torch.tensor(f_atoms, dtype=torch.float)
        self.f_bonds = torch.tensor(f_bonds, dtype=torch.float)
        self.a2b = torch.tensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)],
                                dtype=torch.long)
        self.b2a = torch.tensor(b2a, dtype=torch.long)
        self.b2revb = torch.tensor(b2revb, dtype=torch.long)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = self.b2a[self.a2b]  # only needed if using atom messages
        self.a_scope = torch.tensor(self.a_scope, dtype=torch.long)
        self.b_scope = torch.tensor(self.b_scope, dtype=torch.long)

        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.register_tensor_members(k, v)

    @property
    def components(self):
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.a2a
