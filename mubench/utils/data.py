from typing import Optional, List


def unzip_list_of_dicts(instance_list: List[dict], feature_names: Optional[List[str]] = None):
    if not feature_names:
        feature_names = list(instance_list[0].keys())

    features_lists = list()
    for name in feature_names:
        features_lists.append([inst[name] for inst in instance_list])

    return features_lists


