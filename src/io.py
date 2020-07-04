import pandas as pd
from glob import glob
from typing import List, Tuple

names_simple_model = [
    "svm_rbf",
    "svm_linear",
    "knn",
    "nusvm",
    "ridge",
    "ridge_bagging",
    "lgbm_gnn_feature",
    "lgbm_2plus1dcnn_feature",
    "xgb_gnn_feature",
    "xgb_3dcnn_feature",
]
use_feature_selection = [
    False,
    False,
    False,
    False,
    False,
    False,
    True,
    True,
    True,
    True,
]

assert len(names_simple_model) == len(use_feature_selection)

names_nn = [
    "1d_densenet121",
    "3dresnet18",
    "1dresnet18",
    "simple_3dcnn",
    "transformer",
    "simple_3dcnn_3label",
    "gin",
    "1dresnest14d",
    "2plus1dcnn_resnet10",
]


def load_type_a(
    workdir: str, model_name: str, use_feature_selection: bool, is_train: bool
) -> pd.DataFrame:
    """
    except NN model
    """
    if not is_train:
        suffix = "test"
    else:
        suffix = "train"
    if not use_feature_selection:
        path = f"{workdir}/output/{model_name}/result/{model_name}_{suffix}.csv"
    else:
        path = f"{workdir}/output/{model_name}/result/{model_name}_feature_section1024_{suffix}.csv"
    df = pd.read_csv(path).sort_values("Id").reset_index(drop=True)
    return df


def load_type_b(workdir: str, model_name: str, is_train: bool) -> pd.DataFrame:
    """
    NN model
    """
    if not is_train:
        paths = glob(f"{workdir}/output/{model_name}/fold*/result/test_result.csv")
        df = pd.concat([pd.read_csv(path).sort_values("Id") for path in paths], axis=1,)
        df = (
            df.loc[:, ~df.columns.duplicated()].sort_values("Id").reset_index(drop=True)
        )
    else:
        paths = glob(f"{workdir}/output/{model_name}/fold*/result/valid_result_all.csv")
        df = (
            pd.concat([pd.read_csv(path) for path in paths], axis=0,)
            .sort_values("Id")
            .drop_duplicates()
            .reset_index(drop=True)
        )
    return df


def load_train_data(workdir: str) -> Tuple[List[pd.DataFrame], List[str]]:
    dfs = []
    dfs.extend(
        [
            load_type_a(workdir, name, use_pseudo, is_train=True)
            for name, use_pseudo in zip(names_simple_model, use_feature_selection)
        ]
    )
    dfs.extend([load_type_b(workdir, name, is_train=True) for name in names_nn])
    names = names_simple_model + names_nn
    return dfs, names


def load_test_data(workdir: str) -> List[pd.DataFrame]:
    dfs = []
    dfs.extend(
        [
            load_type_a(workdir, name, use_pseudo, is_train=False)
            for name, use_pseudo in zip(names_simple_model, use_feature_selection)
        ]
    )
    dfs.extend([load_type_b(workdir, name, is_train=False) for name in names_nn])
    return dfs
