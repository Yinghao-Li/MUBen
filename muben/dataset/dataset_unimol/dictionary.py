# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

UNIMOL_DICT = [
    "[PAD]",
    "[CLS]",
    "[SEP]",
    "[UNK]",
    "C",
    "N",
    "O",
    "S",
    "H",
    "Cl",
    "F",
    "Br",
    "I",
    "Si",
    "P",
    "B",
    "Na",
    "K",
    "Al",
    "Ca",
    "Sn",
    "As",
    "Hg",
    "Fe",
    "Zn",
    "Cr",
    "Se",
    "Gd",
    "Au",
    "Li",
]


class DictionaryUniMol:
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # begin keyword-only arguments
        bos="[CLS]",
        pad="[PAD]",
        eos="[SEP]",
        unk="[UNK]",
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.specials = set()
        self.specials.add(bos)
        self.specials.add(unk)
        self.specials.add(pad)
        self.specials.add(eos)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def vec_index(self, a):
        return np.vectorize(self.index)(a)

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.indices[self.unk_word]

    def special_index(self):
        return [self.index(x) for x in self.specials]

    def add_symbol(self, word, n=1, overwrite=False, is_special=False):
        """Adds a word to the dictionary"""
        if is_special:
            self.specials.add(word)
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.index(self.bos_word)

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.index(self.pad_word)

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.index(self.eos_word)

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.index(self.unk_word)

    @classmethod
    def load(cls, f=None):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        _ = d.add_from_file(f) if f else d.add_from_macro(UNIMOL_DICT)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(f"Incorrect encoding detected in {f}, please rebuild the dataset")
            return

        lines = f.readlines()

        for line_idx, line in enumerate(lines):
            try:
                splits = line.rstrip().rsplit(" ", 1)
                line = splits[0]
                field = splits[1] if len(splits) > 1 else str(len(lines) - line_idx)
                if field == "#overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    logger.info(
                        f"Duplicate word found when loading Dictionary: '{word}', index is {self.indices[word]}."
                    )
                else:
                    self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt> [flags]'")

    def add_from_macro(self, token_list):
        """
        Loads a pre-existing dictionary from a macro defined in a py script
        to this instance.
        """
        for line_idx, token in enumerate(token_list):
            count = len(token_list) - line_idx
            self.add_symbol(token, n=count, overwrite=False)
