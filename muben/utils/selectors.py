from muben.args import (
    Arguments,
    Config,
    Arguments2D,
    Config2D,
    Arguments3D,
    Config3D,
    ArgumentsRDKit,
    ConfigRDKit,
    ArgumentsLinear,
    ConfigLinear,
    ArgumentsGrover,
    ConfigGrover,
    ArgumentsUniMol,
    ConfigUniMol,
)
from muben.dataset import (
    Dataset2D,
    Collator2D,
    Dataset3D,
    Collator3D,
    DatasetRDKit,
    CollatorRDKit,
    DatasetLinear,
    CollatorLinear,
    DatasetUniMol,
    CollatorUniMol,
    DictionaryUniMol,
    DatasetGrover,
    CollatorGrover,
)
from muben.model import (
    DNN,
    GIN,
    LinearTransformer,
    GROVER,
    TorchMDNET,
    UniMol,
)


def argument_selector(descriptor_type: str):
    if descriptor_type == "2D":
        return Arguments2D
    elif descriptor_type == "3D":
        return Arguments3D
    elif descriptor_type == "RDKit":
        return ArgumentsRDKit
    elif descriptor_type == "Linear":
        return ArgumentsLinear
    elif descriptor_type == "Grover":
        return ArgumentsGrover
    elif descriptor_type == "UniMol":
        return ArgumentsUniMol
    else:
        return Arguments


def configure_selector(descriptor_type: str):
    if descriptor_type == "2D":
        return Config2D
    elif descriptor_type == "3D":
        return Config3D
    elif descriptor_type == "RDKit":
        return ConfigRDKit
    elif descriptor_type == "Linear":
        return ConfigLinear
    elif descriptor_type == "Grover":
        return ConfigGrover
    elif descriptor_type == "UniMol":
        return ConfigUniMol
    else:
        return Config


def dataset_selector(descriptor_type: str):
    if descriptor_type == "2D":
        return Dataset2D, Collator2D
    elif descriptor_type == "3D":
        return Dataset3D, Collator3D
    elif descriptor_type == "RDKit":
        return DatasetRDKit, CollatorRDKit
    elif descriptor_type == "Linear":
        return DatasetLinear, CollatorLinear
    elif descriptor_type == "Grover":
        return DatasetGrover, CollatorGrover
    elif descriptor_type == "UniMol":
        return DatasetUniMol, CollatorUniMol, DictionaryUniMol
    else:
        raise ValueError("Invalid descriptor type")


def model_selector(descriptor_type: str):
    if descriptor_type == "2D":
        return GIN
    elif descriptor_type == "3D":
        return TorchMDNET
    elif descriptor_type == "RDKit":
        return DNN
    elif descriptor_type == "Linear":
        return LinearTransformer
    elif descriptor_type == "Grover":
        return GROVER
    elif descriptor_type == "UniMol":
        return UniMol
    else:
        raise ValueError("Invalid descriptor type")
