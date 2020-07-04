import logging
import os
import random
from glob import glob
from typing import Tuple


import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor

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
    # mean shift
    train_df["IC_20"] += 0.0022449734660541093
    # Scaling
    train_df[fnc.columns[1:]] /= 500
    test_df[fnc.columns[1:]] /= 500

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


@hydra.main(config_path="yamls/ridge.yaml")
def main(config: DictConfig) -> None:
    prepair_dir(config)
    train_df, test_df = load_data(config)
    label_cols = ["age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]
    feature_cols = [col for col in train_df.columns if col not in label_cols + ["Id"]]
    train_df["age_rank"] = train_df["age"] // 10 * 10
    skf = StratifiedKFold(n_splits=5, random_state=config.data.seed, shuffle=True)
    for i, (_, val_index) in enumerate(skf.split(train_df, train_df["age_rank"])):
        train_df.loc[val_index, "fold"] = i
    if config.randomize_age:
        set_seed(100)
        train_df["age"] += [randomize_age(age) for age in train_df["age"]]

    for label_col in label_cols:
        best_score = np.inf
        best_alpha = 0.0
        best_pred = np.zeros([train_df.shape[0]])
        for alpha in [0.01, 0.001, 0.0003, 0.0001]:
            for n_fold in range(5):
                if not config.use_bagging:
                    model = Ridge(alpha=alpha)
                else:
                    model = BaggingRegressor(
                        Ridge(alpha=alpha),
                        n_estimators=30,
                        random_state=42,
                        max_samples=0.3,
                        max_features=0.3,
                    )
                X_train = train_df.query("fold!=@n_fold")[feature_cols]
                y_train = train_df.query("fold!=@n_fold")[label_col]
                X_train = X_train[y_train.notnull()]
                y_train = y_train[y_train.notnull()]
                model.fit(X_train, y_train)
                train_df.loc[
                    train_df.query("fold==@n_fold").index, f"{label_col}_pred"
                ] = model.predict(train_df.query("fold==@n_fold")[feature_cols])
            score = normalized_absolute_errors(
                train_df[label_col].values, train_df[f"{label_col}_pred"].values
            )
            logger.info(f"{label_col} alpha: {alpha}, score: {score}")
            if score <= best_score:
                best_score = score
                best_alpha = alpha
                best_pred[:] = train_df[f"{label_col}_pred"].values
        train_df[f"{label_col}_pred"] = best_pred
        for n_fold in range(5):
            if not config.use_bagging:
                model = Ridge(alpha=best_alpha)
            else:
                model = BaggingRegressor(
                    Ridge(alpha=best_alpha),
                    n_estimators=30,
                    random_state=42,
                    max_samples=0.3,
                    max_features=0.3,
                )
            X_train = train_df.query("fold!=@n_fold")[feature_cols]
            y_train = train_df.query("fold!=@n_fold")[label_col]
            X_train = X_train[y_train.notnull()]
            y_train = y_train[y_train.notnull()]
            model.fit(X_train, y_train)
            test_df[f"{label_col}_pred_fold{n_fold}"] = model.predict(
                test_df[feature_cols]
            )

        score = normalized_absolute_errors(
            train_df[label_col].values, train_df[f"{label_col}_pred"].values
        )
        logger.info(f"{label_col} alpha: {best_alpha}, score: {score}")
        test_df[label_col] = test_df[
            [f"{label_col}_pred_fold{i}" for i in range(5)]
        ].mean(1)
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
    if config.store.gcs_project is not None:
        upload_directory(config.store)


if __name__ == "__main__":
    main()
