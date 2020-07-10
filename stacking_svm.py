import logging
import os
import random
import pickle
from glob import glob
from itertools import combinations
from typing import List, Optional, Tuple, Union, Dict

import hydra
import cudf
from cuml import SVR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from google.cloud import storage
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.metrics import weighted_normalized_absolute_errors, normalized_absolute_errors
from src.io import load_train_data, load_test_data
from src.randomize import randomize_age


plt.style.use("ggplot")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepair_dir(config: DictConfig) -> None:
    """
    Logの保存先を作成
    """
    for path in [
        config.store.result_path,
        config.store.log_path,
        config.store.model_path,
    ]:
        os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    os.environ.PYTHONHASHSEED = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_submission(test_df: cudf.DataFrame) -> cudf.DataFrame:
    for label_col in [
        "age",
        "domain1_var1",
        "domain1_var2",
        "domain2_var1",
        "domain2_var2",
    ]:
        test_df[label_col] = test_df[
            [f"{label_col}_pred_fold{i}" for i in range(5)]
        ].mean(1)
    sub_df = cudf.melt(
        test_df[
            [
                "Id",
                "age",
                "domain1_var1",
                "domain1_var2",
                "domain2_var1",
                "domain2_var2",
            ]
        ],
        id_vars=["Id"],
        value_name="Predicted",
    )
    sub_df["Id"] = sub_df["Id"].astype("str") + "_" + sub_df["variable"].astype("str")

    sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
    assert sub_df.shape[0] == test_df.shape[0] * 5
    return sub_df


def upload_directory(store_config: DictConfig) -> None:
    storage_client = storage.Client(store_config.gcs_project)
    bucket = storage_client.get_bucket(store_config.bucket_name)
    filenames = glob(os.path.join(store_config.save_path, "**"), recursive=True)
    for filename in filenames:
        if os.path.isdir(filename):
            continue
        destination_blob_name = os.path.join(
            store_config.gcs_path, filename.split(store_config.save_path)[-1][1:]
        )
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(filename)


def make_domain_feature(df: pd.DataFrame, mode: str, name):
    domain_cols = ["domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]
    feat_dict = {}
    if mode != "test":
        for col1, col2 in combinations(domain_cols, 2):
            if f"{col1}_pred" in df.columns and f"{col2}_pred" in df.columns:
                feat_dict[f"{name}_{col1}_{col2}_sum"] = (
                    df[f"{col1}_pred"] + df[f"{col2}_pred"]
                )
                feat_dict[f"{name}_{col1}_{col2}_diff"] = np.abs(
                    df[f"{col1}_pred"] - df[f"{col2}_pred"]
                )
                feat_dict[f"{name}_{col1}_{col2}_multiply"] = (
                    df[f"{col1}_pred"] * df[f"{col2}_pred"]
                )
                # feat_dict[f"{name}_{col1}_{col2}_div"] = df[f"{col1}_pred"] / (
                #     df[f"{col2}_pred"] + 1e-8
                # )
                # feat_dict[f"{name}_{col2}_{col1}_div"] = df[f"{col2}_pred"] / (
                #     df[f"{col1}_pred"] + 1e-8
                # )
    else:
        for col1, col2 in combinations(domain_cols, 2):
            for n_fold in range(5):
                if (
                    f"{col1}_pred_fold{n_fold}" in df.columns
                    and f"{col2}_pred_fold{n_fold}" in df.columns
                ):
                    feat_dict[f"{name}_{col1}_{col2}_sum_fold{n_fold}"] = (
                        df[f"{col1}_pred_fold{n_fold}"]
                        + df[f"{col2}_pred_fold{n_fold}"]
                    )
                    feat_dict[f"{name}_{col1}_{col2}_diff_fold{n_fold}"] = np.abs(
                        df[f"{col1}_pred_fold{n_fold}"]
                        - df[f"{col2}_pred_fold{n_fold}"]
                    )
                    feat_dict[f"{name}_{col1}_{col2}_multiply_fold{n_fold}"] = (
                        df[f"{col1}_pred_fold{n_fold}"]
                        * df[f"{col2}_pred_fold{n_fold}"]
                    )
                    # feat_dict[f"{name}_{col1}_{col2}_div_fold{n_fold}"] = df[
                    #     f"{col1}_pred_fold{n_fold}"
                    # ] / (df[f"{col2}_pred_fold{n_fold}"] + 1e-8)
                    # feat_dict[f"{name}_{col2}_{col1}_div_fold{n_fold}"] = df[
                    #     f"{col2}_pred_fold{n_fold}"
                    # ] / (df[f"{col1}_pred_fold{n_fold}"] + 1e-8)
    return feat_dict


def preprocess(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    std = StandardScaler()
    train_df[feature_cols] = std.fit_transform(train_df[feature_cols])
    for col in feature_cols:
        test_df[col] = test_df[[f"{col}_fold{i}" for i in range(5)]].mean(1)
    test_df[feature_cols] = std.transform(test_df[feature_cols])

    return train_df, test_df


@hydra.main(config_path="yamls/stacking.yaml")
def main(config: DictConfig) -> None:
    prepair_dir(config)
    set_seed(config.data.seed)
    label_cols = ["age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]
    train_dfs, names = load_train_data(config.store.workdir)
    test_dfs = load_test_data(config.store.workdir)
    remove_cols = [
        "knn_age_pred",
        "knn_domain1_var1",
        "densenet121_age_pred",
        "densenet121_domain1_var1_pred",
        "densenet121_domain1_var2_pred",
        "densenet121_domain2_var2_pred",
        "3dcnn_resnet18_domain1_var2_pred",
        "3dcnn_resnet18_domain2_var1_pred",
        "3dcnn_resnet18_domain2_var2_pred",
        "1dresnet18_domain1_var1_pred",
        "1dresnet18_domain1_var2_pred",
        "1dresnet18_domain2_var2_pred",
        "simple_3dcnn_domain1_var1_pred",
        "simple_3dcnn_domain1_var2_pred",
        "simple_3dcnn_domain2_var2_pred",
        "transformer_domain2_var1_pred",
        "transformer_domain2_var2_pred",
        "transformer_domain1_var1_pred",
        "transformer_domain1_var2_pred",
        "lgbm_gnn_feature_domain1_var2_pred",
        "lgbm_gnn_feature_domain2_var2_pred",
        "lgbm_gnn_featured_domain1_var2_pred",
        "lgbm_gnn_featured_domain2_var2_pred",
        "lgbm_cnn_feature_domain1_var2_pred",
        "lgbm_cnn_feature_domain2_var2_pred",
        "lgbm_2plus1dcnn_feature_domain1_var2_pred",
        "lgbm_2plus1dcnn_feature_domain2_var2_pred",
        "xgb_2plus1dcnn_feature_age_pred",
        "xgb_2plus1dcnn_feature_domain1_var2_pred",
        "xgb_2plus1dcnn_feature_domain2_var2_pred",
        "simple_3dcnn_domain2_var1_pred",
        "simple_3dcnn_3label_domain1_var2_pred",
        "gin_domain1_var1_pred",
        "gin_domain2_var1_pred",
        "2plus1dcnn_resnet10_domain1_var2_pred",
        "resnest14d_domain1_var1_pred",
        "resnest14d_domain1_var2_pred",
        "resnest14d_domain2_var2_pred",
    ]
    train_ft_dict = {}
    test_ft_dict = {}
    feature_cols = []
    train_ft_dict["Id"] = train_dfs[0]["Id"]
    test_ft_dict["Id"] = test_dfs[0]["Id"]
    for label_col in label_cols:
        train_ft_dict[label_col] = train_dfs[0][label_col]
    for name, df in zip(names, train_dfs):
        for label_col in label_cols:
            if (
                f"{label_col}_pred" in df.columns
                and f"{name}_{label_col}_pred" not in remove_cols
            ):
                train_ft_dict[f"{name}_{label_col}_pred"] = df[f"{label_col}_pred"]
                feature_cols += [f"{name}_{label_col}_pred"]
            elif f"{name}_{label_col}_pred" in remove_cols:
                df.drop(f"{label_col}_pred", axis=1, inplace=True)

        feat_dict = make_domain_feature(df, mode="train", name=name)
        train_ft_dict.update(feat_dict)
        feature_cols += list(feat_dict.keys())

    for name, df in zip(names, test_dfs):
        for label_col in label_cols:
            for i in range(5):
                if (
                    f"{label_col}_pred_fold{i}" in df.columns
                    and f"{name}_{label_col}_pred" not in remove_cols
                ):
                    test_ft_dict[f"{name}_{label_col}_pred_fold{i}"] = df[
                        f"{label_col}_pred_fold{i}"
                    ]
                elif (
                    f"{name}_{label_col}_pred" in remove_cols
                    and f"{label_col}_pred_fold{i}" in df.columns
                ):
                    df.drop(f"{label_col}_pred_fold{i}", axis=1, inplace=True)
        feat_dict = make_domain_feature(df, mode="test", name=name)
        test_ft_dict.update(feat_dict)
    train_df = pd.DataFrame(train_ft_dict)
    test_df = pd.DataFrame(test_ft_dict)
    train_df["age"] = (
        pd.read_csv(f"{config.store.workdir}/input/train_scores.csv")
        .sort_values("Id")
        .reset_index(drop=True)["age"]
    )
    age_rank = train_df["age"].values // 10 * 10
    skf = StratifiedKFold(n_splits=5, random_state=777, shuffle=True)

    train_df, test_df = preprocess(train_df, test_df, feature_cols)
    for feature_col in feature_cols:
        train_df[feature_col].fillna(0, inplace=True)
        test_df[feature_col].fillna(0, inplace=True)
    train_df = cudf.from_pandas(train_df)
    test_df = cudf.from_pandas(test_df)
    if config.randomize_age:
        set_seed(777_777_777)
        train_df["age"] += np.array(
            [randomize_age(age) for age in train_df["age"].values]
        )
    skf = StratifiedKFold(n_splits=5, random_state=777, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    logger.info("=" * 10 + "parameter search" + "=" * 10)
    best_c = {}
    for label_col in label_cols:
        best = np.inf
        if label_col == "age":
            feature_cols_ = [col for col in feature_cols if f"{label_col}" in col]
        else:
            feature_cols_ = feature_cols
        for c in [2 ** (i) for i in range(-14, 1)]:
            y_oof = np.zeros(train_df.shape[0])
            for n_fold, (train_index, val_index) in enumerate(
                skf.split(age_rank, age_rank)
            ):
                train_df_fold = train_df.iloc[train_index]
                valid_df_fold = train_df.iloc[val_index]
                train_df_fold = train_df_fold[train_df_fold[label_col].notnull()]
                model = SVR(kernel="linear", C=c, cache_size=3000.0)
                model.fit(train_df_fold[feature_cols_], train_df_fold[label_col])
                y_oof[val_index] = model.predict(
                    valid_df_fold[feature_cols_]
                ).to_array()
                test_df[f"{label_col}_pred_fold{n_fold}"] = model.predict(
                    test_df[feature_cols_]
                )
            train_df[f"{label_col}_pred"] = y_oof
            notnull_idx = train_df[label_col].notnull()
            score = normalized_absolute_errors(
                train_df[notnull_idx][label_col].values,
                train_df[notnull_idx][f"{label_col}_pred"].values,
            )
            logger.info(f"c={c}, {label_col}: {score}")
            if score <= best:
                best = score
                best_c[label_col] = c
    logger.info("=" * 10 + "prediction" + "=" * 10)
    for label_col in label_cols:
        y_oof = np.zeros(train_df.shape[0])
        if label_col == "age":
            feature_cols_ = [col for col in feature_cols if f"{label_col}" in col]
        else:
            feature_cols_ = feature_cols
        for n_fold, (train_index, val_index) in enumerate(
            skf.split(age_rank, age_rank)
        ):
            train_df_fold = train_df.iloc[train_index]
            valid_df_fold = train_df.iloc[val_index]
            train_df_fold = train_df_fold[train_df_fold[label_col].notnull()]
            model = SVR(kernel="linear", C=best_c[label_col], cache_size=3000.0)
            model.fit(train_df_fold[feature_cols_], train_df_fold[label_col])
            y_oof[val_index] = model.predict(valid_df_fold[feature_cols_]).to_array()
            test_df[f"{label_col}_pred_fold{n_fold}"] = model.predict(
                test_df[feature_cols_]
            )
        train_df[f"{label_col}_pred"] = y_oof
        notnull_idx = train_df[label_col].notnull()
        score = normalized_absolute_errors(
            train_df[notnull_idx][label_col].values,
            train_df[notnull_idx][f"{label_col}_pred"].values,
        )
        logger.info(f"c={c}, {label_col}: {score}")
    score = 0
    for label_col, weight in zip(label_cols, [0.3, 0.175, 0.175, 0.175, 0.175]):
        notnull_idx = train_df[label_col].notnull()
        score += (
            normalized_absolute_errors(
                train_df[notnull_idx][label_col].to_array(),
                train_df[notnull_idx][f"{label_col}_pred"].to_array(),
            )
            * weight
        )
    logger.info(f"all: {score}")
    train_df.to_csv(
        os.path.join(config.store.result_path, f"{config.store.model_name}_train.csv"),
        index=False,
    )
    test_df.to_csv(
        os.path.join(config.store.result_path, f"{config.store.model_name}_test.csv"),
        index=False,
    )
    if config.store.gcs_project is not None:
        upload_directory(config.store)

    sub_df = make_submission(test_df)
    sub_df.to_csv(
        os.path.join(
            config.store.result_path, f"{config.store.model_name}_submission.csv"
        ),
        index=False,
    )


if __name__ == "__main__":
    main()
