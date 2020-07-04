import logging
import os
import random
import pickle
from glob import glob
from typing import List, Optional, Tuple, Union, Dict

import hydra
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from src.metrics import (
    weighted_normalized_absolute_errors,
    normalized_absolute_errors,
)
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


class XGBModel(object):
    """
    label_col毎にxgboost modelを作成するようのクラス
    """

    def __init__(
        self,
        feature_cols: Union[List, np.ndarray],
        label_col: Union[List, np.ndarray],
        params: Dict,
    ):
        self.feature_cols = feature_cols
        self.cnn_feature_cols = [
            col for col in feature_cols if col.startswith("feature")
        ]
        self.label_col = label_col
        self.params = params
        self.model_dicts: Dict[int, xgb.Booster] = {}

    def custom_metric(
        self, y_pred: np.ndarray, dtrain: xgb.DMatrix
    ) -> Tuple[str, np.ndarray]:
        y_true = dtrain.get_label().astype(float)
        loss = normalized_absolute_errors(y_true, y_pred)
        return "normalized_mae", loss

    def store_model(self, bst: xgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def cv(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        pseudo_df: Optional[pd.DataFrame] = None,
    ) -> xgb.Booster:
        importances = []
        for n_fold in range(5):
            for col in self.cnn_feature_cols:
                test_df[col] = test_df[f"{col}_fold{n_fold}"]
                if pseudo_df is not None:
                    pseudo_df[col] = pseudo_df[f"{col}_fold{n_fold}"]
            bst = self.fit(train_df, n_fold, pseudo_df)
            valid = train_df.query("fold == @n_fold")
            train_df.loc[valid.index, f"{self.label_col}_pred"] = bst.predict(
                xgb.DMatrix(valid[self.feature_cols])
            )
            test_df[f"{self.label_col}_pred_fold{n_fold}"] = bst.predict(
                xgb.DMatrix(test_df[self.feature_cols])
            )
            self.store_model(bst, n_fold)
            importance_dict = bst.get_fscore()
            importances.append(
                [
                    importance_dict[col] if col in importance_dict else 0
                    for col in self.feature_cols
                ]
            )
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
    ) -> xgb.Booster:
        notnull_idx = train_df[self.label_col].notnull()
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

        dtrain = xgb.DMatrix(
            X_train, label=np.array(y_train), feature_names=self.feature_cols,
        )
        dvalid = xgb.DMatrix(
            X_valid, label=np.array(y_valid), feature_names=self.feature_cols,
        )
        bst = xgb.train(
            self.params,
            dtrain,
            num_boost_round=50000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=300,
            verbose_eval=300,
            feval=self.custom_metric,
        )
        return bst

    def save_model(self) -> None:
        with open(f"./output/xgboost_models/{self.label_col}.pkl", "wb") as f:
            pickle.dump(self.model_dicts, f)


def load_data(config: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    loading = pd.read_csv(f"{config.store.workdir}/input/loading.csv")
    fnc = pd.read_csv(f"{config.store.workdir}/input/fnc.csv")
    train_df = pd.read_csv(f"{config.store.workdir}/input/train_scores.csv")
    submissoin = pd.read_csv(f"{config.store.workdir}/input/sample_submission.csv")
    train_df = train_df.merge(loading, on="Id", how="left")
    train_df = train_df.merge(fnc, on="Id", how="left")
    test_df = pd.DataFrame({"Id": submissoin["Id"].str[:5].unique().astype(int)})
    test_df = test_df.merge(loading, on="Id", how="left")
    test_df = test_df.merge(fnc, on="Id", how="left")
    voxel_stats = pd.read_csv(f"{config.store.workdir}/input/voxel_stats3.csv")
    train_df = train_df.merge(voxel_stats, on="Id", how="left")
    test_df = test_df.merge(voxel_stats, on="Id", how="left")
    train_df["IC_20"] += 0.0022449734660541093
    if config.use_gnn_feature:
        feat = pd.read_csv(f"{config.store.workdir}/input/gnn_feature.csv")
        train_df = train_df.merge(feat, on="Id", how="left")
        feat = pd.read_csv(f"{config.store.workdir}/input/gnn_feature_test.csv")
        test_df = test_df.merge(feat, on="Id", how="left")
    elif config.use_3dcnn_feature:
        feat = pd.read_csv(f"{config.store.workdir}/input/3dcnn_feature.csv")
        train_df = train_df.merge(feat, on="Id", how="left")
        feat = pd.read_csv(f"{config.store.workdir}/input/3dcnn_feature_test.csv")
        test_df = test_df.merge(feat, on="Id", how="left")
    elif config.use_2plus1dcnn_feature:
        feat = pd.read_csv(f"{config.store.workdir}/input/2plus1dcnn_feature.csv")
        train_df = train_df.merge(feat, on="Id", how="left")
        feat = pd.read_csv(f"{config.store.workdir}/input/2plus1dcnn_feature_test.csv")
        test_df = test_df.merge(feat, on="Id", how="left")
    return train_df, test_df


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
    importance_df.name = "feature_name"
    importance_df = importance_df.reset_index().sort_values(
        by="importance", ascending=False
    )
    importance_df.to_csv(
        os.path.join(store_config.result_path, f"importance_{label_col + suffix}.csv"),
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


@hydra.main(config_path="yamls/xgboost.yaml")
def main(config: DictConfig) -> None:
    prepair_dir(config)
    set_seed(777777)
    train_df, test_df = load_data(config)
    label_cols = ["age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]
    feature_cols = [col for col in train_df.columns if col not in label_cols + ["Id"]]
    # From adversarial valiadtion
    feature_cols.remove("IC_20")
    train_df["age_rank"] = train_df["age"] // 10 * 10
    skf = StratifiedKFold(n_splits=5, random_state=config.data.seed, shuffle=True)
    for i, (_, val_index) in enumerate(skf.split(train_df, train_df["age_rank"])):
        train_df.loc[val_index, "fold"] = i
    if config.randomize_age:
        train_df["age"] += np.array(
            [randomize_age(age) for age in train_df["age"].values]
        )

    params = {}
    params["age"] = {
        "alpha": 0.8,
        "reg_lambda": 0.8,
        "max_leaves": 3,
        "colsample_bytree": 0.6,
        "subsample": 0.8,
        "min_child_weight": 50,
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 10,
        "learning_rate": 0.01,
        "nthread": -1,
        "max_bin": 256,
        "tree_method": "gpu_hist",
        "disable_default_eval_metric": 1,
    }
    params["domain1_var1"] = {
        "alpha": 0.8,
        "reg_lambda": 0.8,
        "max_leaves": 4,
        "colsample_bytree": 0.5,
        "subsample": 0.8,
        "min_child_weight": 5,
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 7,
        "learning_rate": 0.01,
        "nthread": -1,
        "max_bin": 256,
        "tree_method": "gpu_hist",
        "disable_default_eval_metric": 1,
    }
    params["domain1_var2"] = {
        "alpha": 0.8,
        "reg_lambda": 0.8,
        "max_leaves": 4,
        "colsample_bytree": 0.8,
        "subsample": 0.6,
        "min_child_weight": 5,
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 7,
        "learning_rate": 0.01,
        "nthread": -1,
        "max_bin": 256,
        "tree_method": "gpu_hist",
        "disable_default_eval_metric": 1,
    }
    params["domain2_var1"] = {
        "alpha": 0.8,
        "reg_lambda": 0.8,
        "max_leaves": 4,
        "colsample_bytree": 0.7,
        "subsample": 0.4,
        "min_child_weight": 5,
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 7,
        "learning_rate": 0.01,
        "nthread": -1,
        "max_bin": 256,
        "tree_method": "gpu_hist",
        "disable_default_eval_metric": 1,
    }
    params["domain2_var2"] = {
        "alpha": 0.8,
        "reg_lambda": 0.8,
        "max_leaves": 4,
        "colsample_bytree": 0.7,
        "subsample": 0.4,
        "min_child_weight": 5,
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 7,
        "learning_rate": 0.01,
        "nthread": -1,
        "max_bin": 256,
        "tree_method": "gpu_hist",
        "disable_default_eval_metric": 1,
    }
    for label_col in label_cols:
        model = XGBModel(feature_cols, label_col, params[label_col])
        train_df, test_df = model.cv(train_df, test_df)
        score = normalized_absolute_errors(
            train_df[label_col].values, train_df[f"{label_col}_pred"].values
        )
        logger.info(f"{label_col} score: {score}")
        # test_df[label_col] = test_df[
        #     [f"{label_col}_pred_fold{i}" for i in range(5)]
        # ].mean(1)
        save_importance(model.importance_df, label_col, config.store)
    score = weighted_normalized_absolute_errors(
        train_df[label_cols].values,
        train_df[[f"{label_col}_pred" for label_col in label_cols]].values,
    )
    logger.info(f"all score: {score}")
    train_df.to_csv(
        os.path.join(config.store.result_path, f"{config.store.model_name}_train.csv"),
        index=False,
    )
    test_df.to_csv(
        os.path.join(config.store.result_path, f"{config.store.model_name}_test.csv"),
        index=False,
    )
    # with pseudo label
    for label_col in label_cols:
        feature_cols = list(
            pd.read_csv(
                f"{config.store.workdir}/output/{config.store.model_name}/result/importance_{label_col}.csv"
            )["index"].values[:1024]
        )
        model = XGBModel(feature_cols, label_col, params[label_col])
        train_df, test_df = model.cv(train_df, test_df)
        score = normalized_absolute_errors(
            train_df[label_col].values, train_df[f"{label_col}_pred"].values
        )
        logger.info(f"{label_col} score: {score}")
        test_df[label_col] = test_df[
            [f"{label_col}_pred_fold{i}" for i in range(5)]
        ].mean(1)
        save_importance(
            model.importance_df,
            label_col,
            config.store,
            suffix=f"_feature_section{config.n_feature}",
        )
    score = weighted_normalized_absolute_errors(
        train_df[label_cols].values,
        train_df[[f"{label_col}_pred" for label_col in label_cols]].values.copy(),
    )
    logger.info(f"all score: {score}")
    train_df.to_csv(
        os.path.join(
            config.store.result_path,
            f"{config.store.model_name}_feature_section{config.n_feature}_train.csv",
        ),
        index=False,
    )
    test_df.to_csv(
        os.path.join(
            config.store.result_path,
            f"{config.store.model_name}_feature_section{config.n_feature}_test.csv",
        ),
        index=False,
    )
    sub_df = make_submission(test_df)
    sub_df.to_csv(
        os.path.join(
            config.store.result_path,
            f"{config.store.model_name}_feature_section{config.n_feature}_submission.csv",
        ),
        index=False,
    )
    if config.store.gcs_project is not None:
        upload_directory(config.store)


if __name__ == "__main__":
    main()
