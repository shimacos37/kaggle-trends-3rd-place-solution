import logging
import os
import random
import pickle
from glob import glob
from itertools import combinations
from typing import List, Optional, Tuple, Union, Dict

import hydra
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from google.cloud import storage
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.metrics import (
    weighted_normalized_absolute_errors,
    normalized_absolute_errors,
)
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


class LGBMModel(object):
    """
    label_col毎にlightgbm modelを作成するようのクラス
    """

    def __init__(
        self,
        feature_cols: Union[List, np.ndarray],
        label_col: Union[List, np.ndarray],
        params: Dict,
    ):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.params = params
        self.model_dicts: Dict[int, lgb.Booster] = {}

    def custom_metric(
        self, y_pred: np.ndarray, dtrain: lgb.basic.Dataset
    ) -> Tuple[str, np.ndarray, bool]:
        y_true = dtrain.get_label().astype(float)
        loss = normalized_absolute_errors(y_true, y_pred)
        return "normalized_mae", loss, False

    def store_model(self, bst: lgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def cv(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        pseudo_df: Optional[pd.DataFrame] = None,
    ) -> lgb.Booster:
        importances = []
        for n_fold in range(5):
            bst = self.fit(train_df, n_fold, pseudo_df)
            valid = train_df.query("fold == @n_fold")
            train_df.loc[valid.index, f"{self.label_col}_pred"] = bst.predict(
                valid[self.feature_cols]
            )
            for feature_col in self.feature_cols:
                test_df[feature_col] = test_df[f"{feature_col}_fold{n_fold}"]

            test_df[f"{self.label_col}_pred_fold{n_fold}"] = bst.predict(
                test_df[self.feature_cols]
            )
            self.store_model(bst, n_fold)
            importances.append(bst.feature_importance())
        importances = np.mean(importances, axis=0)
        importance_df = pd.DataFrame(
            {"importance": importances}, index=self.feature_cols
        ).sort_values(by="importance", ascending=True)
        self.store_importance(importance_df)
        return train_df, test_df

    def fit(
        self,
        train_df: pd.DataFrame,
        n_fold: int,
        pseudo_df: Optional[pd.DataFrame] = None,
    ) -> lgb.Booster:
        notnull_idx = train_df[self.label_col].notnull()
        if pseudo_df is not None:
            X_train = pd.concat(
                [
                    train_df.loc[notnull_idx].query("fold!=@n_fold")[self.feature_cols],
                    pseudo_df[self.feature_cols],
                ]
            )
            y_train = pd.concat(
                [
                    train_df.loc[notnull_idx].query("fold!=@n_fold")[self.label_col],
                    pseudo_df[self.label_col],
                ]
            )
        else:
            X_train = train_df.loc[notnull_idx].query("fold!=@n_fold")[
                self.feature_cols
            ]
            y_train = train_df.loc[notnull_idx].query("fold!=@n_fold")[self.label_col]

        X_valid = train_df.loc[notnull_idx].query("fold==@n_fold")[self.feature_cols]
        y_valid = train_df.loc[notnull_idx].query("fold==@n_fold")[self.label_col]
        print("=" * 10, self.label_col, n_fold, "=" * 10)
        lgtrain = lgb.Dataset(
            X_train, label=np.array(y_train), feature_name=self.feature_cols,
        )
        lgvalid = lgb.Dataset(
            X_valid, label=np.array(y_valid), feature_name=self.feature_cols,
        )
        evals_result = {}
        bst = lgb.train(
            self.params,
            lgtrain,
            num_boost_round=20000,
            valid_sets=[lgtrain, lgvalid],
            valid_names=["train", "valid"],
            early_stopping_rounds=300,
            verbose_eval=None,
            evals_result=evals_result,
            feval=self.custom_metric,
        )
        logger.info(
            f"train: {evals_result['train']['normalized_mae'][-1]}, valid: {evals_result['valid']['normalized_mae'][-1]}"
        )
        return bst

    def save_model(self) -> None:
        with open(f"./output/lightgbm_models/{self.label_col}.pkl", "wb") as f:
            pickle.dump(self.model_dicts, f)


def make_submission(test_df: pd.DataFrame) -> pd.DataFrame:
    sub_df = pd.melt(
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


def save_importance(
    importance_df: pd.DataFrame,
    label_col: str,
    store_config: DictConfig,
    suffix: str = "",
) -> None:
    importance_df.iloc[-50:].plot.barh(figsize=(10, 20))
    plt.tight_layout()
    plt.savefig(
        os.path.join(store_config.result_path, f"importance_{label_col + suffix}.png")
    )
    plt.close()
    importance_df.name = "feature_name"
    importance_df = importance_df.reset_index().sort_values(
        by="importance", ascending=False
    )
    importance_df.to_csv(
        os.path.join(store_config.result_path, f"importance_{label_col + suffix}.csv"),
        index=False,
    )


def upload_directory(store_config: DictConfig) -> None:
    storage_client = storage.Client(store_config.gcs_project)
    bucket = storage_client.get_bucket(store_config.bucket_name)
    filenames = glob(os.path.join(store_config.save_path, "**"), recursive=True)
    for filename in filenames:
        if os.path.isdir(filename):
            continue
        destination_blob_name = os.path.join(
            store_config.gcs_path, filename.split(store_config.save_path)[-1][1:],
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


@hydra.main(config_path="yamls/stacking.yaml")
def main(config: DictConfig) -> None:
    prepair_dir(config)
    set_seed(config.data.seed)
    label_cols = ["age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]
    train_dfs, names = load_train_data(config.store.workdir)
    test_dfs = load_test_data(config.store.workdir)
    params = {
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "num_leaves": 2,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.6,
        "bagging_freq": 1,
        "min_child_samples": 10,
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "mae",
        "metric": "normalized_mae",
        "max_depth": 7,
        "learning_rate": 0.01,
        "num_thread": 4,
        "max_bin": 256,
        "verbose": -1,
        "device": "cpu",
    }
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
                elif f"{name}_{label_col}_pred" in remove_cols:
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
    train_df["age_rank"] = train_df["age"] // 10 * 10
    skf = StratifiedKFold(n_splits=5, random_state=777, shuffle=True)
    for i, (train_index, val_index) in enumerate(
        skf.split(train_df, train_df["age_rank"])
    ):
        train_df.loc[val_index, "fold"] = i

    if config.randomize_age:
        set_seed(777777777)
        train_df["age"] += np.array(
            [randomize_age(age) for age in train_df["age"].values]
        )
    if (
        not config.is_null_importance
        and not config.is_adversarial_validation
        and not config.is_quantile
    ):
        for label_col in label_cols:
            if not config.is_split_label:
                model = LGBMModel(feature_cols, label_col, params)
            else:
                model = LGBMModel(
                    [col for col in feature_cols if f"{label_col}" in col],
                    label_col,
                    params,
                )
            train_df, test_df = model.cv(train_df, test_df)
            score = normalized_absolute_errors(
                train_df[label_col].values, train_df[f"{label_col}_pred"].values
            )
            logger.info(f"{label_col} score: {score}")
            test_df[label_col] = test_df[
                [f"{label_col}_pred_fold{i}" for i in range(5)]
            ].mean(1)
            save_importance(model.importance_df, label_col, config.store)
        score = weighted_normalized_absolute_errors(
            train_df[label_cols].values,
            train_df[[f"{label_col}_pred" for label_col in label_cols]].values.copy(),
        )
        logger.info(f"{names} all score: {score}")
        train_df.to_csv(
            os.path.join(
                config.store.result_path, f"{config.store.model_name}_train.csv"
            ),
            index=False,
        )
        test_df.to_csv(
            os.path.join(
                config.store.result_path, f"{config.store.model_name}_test.csv"
            ),
            index=False,
        )
        if config.store.gcs_project is not None:
            upload_directory(config.store)
        sub_df = make_submission(test_df)
        sub_df.to_csv(
            os.path.join(
                config.store.result_path, f"{config.store.model_name}_submission.csv",
            ),
            index=False,
        )
    elif config.is_quantile:
        params = {
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "num_leaves": 13,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.6,
            "bagging_freq": 1,
            "min_child_samples": 10,
            "task": "train",
            "boosting_type": "gbdt",
            "objective": "quantile",
            "alpha": 0.75,
            "metric": None,
            "max_depth": 7,
            "learning_rate": 0.01,
            "num_thread": 4,
            "max_bin": 256,
            "verbose": -1,
            "device": "cpu",
        }
        for label_col in label_cols:
            model = LGBMModel(feature_cols, label_col, params)
            train_df, test_df = model.cv(train_df, test_df)
            train_df = train_df.rename(
                columns={f"{label_col}_pred": f"{label_col}_pred_upper"}
            )
            for i in range(5):
                test_df = test_df.rename(
                    columns={
                        f"{label_col}_pred_fold{i}": f"{label_col}_pred_fold{i}_upper"
                    }
                )
            test_df[f"{label_col}_pred_upper"] = test_df[
                [f"{label_col}_pred_fold{i}_upper" for i in range(5)]
            ].mean(1)
        params["alpha"] = 0.25
        for label_col in label_cols:
            model = LGBMModel(feature_cols, label_col, params)
            train_df, test_df = model.cv(train_df, test_df)
            train_df = train_df.rename(
                columns={f"{label_col}_pred": f"{label_col}_pred_lower"}
            )
            for i in range(5):
                test_df = test_df.rename(
                    columns={
                        f"{label_col}_pred_fold{i}": f"{label_col}_pred_fold{i}_lower"
                    }
                )
            test_df[f"{label_col}_pred_lower"] = test_df[
                [f"{label_col}_pred_fold{i}_lower" for i in range(5)]
            ].mean(1)
        params["alpha"] = 0.5
        for label_col in label_cols:
            model = LGBMModel(feature_cols, label_col, params)
            train_df, test_df = model.cv(train_df, test_df)
            score = normalized_absolute_errors(
                train_df[label_col].values, train_df[f"{label_col}_pred"].values
            )
            logger.info(f"{label_col} score: {score}")
            test_df[label_col] = test_df[
                [f"{label_col}_pred_fold{i}" for i in range(5)]
            ].mean(1)
            save_importance(
                model.importance_df, label_col, config.store, suffix="_quantile"
            )
            test_df[f"{label_col}_pred"] = test_df[
                [f"{label_col}_pred_fold{i}" for i in range(5)]
            ].mean(1)
        score = weighted_normalized_absolute_errors(
            train_df[label_cols].values,
            train_df[[f"{label_col}_pred" for label_col in label_cols]].values.copy(),
        )
        logger.info(f"{names} all score: {score}")
        train_df.to_csv(
            os.path.join(
                config.store.result_path,
                f"{config.store.model_name}_quantile_train.csv",
            ),
            index=False,
        )
        test_df.to_csv(
            os.path.join(
                config.store.result_path, f"{config.store.model_name}_quantile_test.csv"
            ),
            index=False,
        )

    elif config.is_adversarial_validation:
        skf = StratifiedKFold(n_splits=5, random_state=config.data.seed, shuffle=True)
        if True:
            train_df["is_train"] = 1
            test_df["is_train"] = 0
            label_col = "is_train"
        train_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        if False:
            site2_ids = pd.read_csv(
                f"{config.store.workdir}/input/reveal_ID_site2.csv"
            )["Id"].values
            train_df.loc[train_df.query("Id in @site2_ids").index, "is_site2"] = 1
            train_df["is_site2"].fillna(0, inplace=True)
            label_col = "is_site2"

        for i, (_, val_index) in enumerate(skf.split(train_df, train_df[label_col])):
            train_df.loc[val_index, "fold"] = i
        param = {
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "num_leaves": 32,
            "feature_fraction": 0.4,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_child_samples": 20,
            "task": "train",
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "max_depth": 10,
            "learning_rate": 0.01,
            "num_thread": -1,
            "max_bin": 256,
            "verbose": -1,
            "device": "cpu",
        }
        model = LGBMModel(feature_cols, label_col, param)
        train_df, test_df = model.cv(train_df, test_df)
        test_df[label_col] = test_df[
            [f"{label_col}_pred_fold{i}" for i in range(5)]
        ].mean(1)
        score = roc_auc_score(
            train_df[label_col].values, train_df[f"{label_col}_pred"].values
        )
        logger.info(f"{label_col} score: {score}")
        save_importance(model.importance_df, label_col, config.store)
        train_df.to_csv(
            os.path.join(
                config.store.result_path, f"{config.store.model_name}_adv_val.csv"
            ),
            index=False,
        )


if __name__ == "__main__":
    main()
