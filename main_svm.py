import logging
import os
import random
from glob import glob
from typing import Tuple

import cudf
import hydra
import matplotlib.pyplot as plt
import numpy as np
from cuml import SVR
from google.cloud import storage
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

from src.metrics import normalized_absolute_errors
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


def load_data(config: DictConfig) -> Tuple[cudf.DataFrame, cudf.DataFrame]:
    loading = cudf.read_csv(f"{config.store.workdir}/input/loading.csv")
    fnc = cudf.read_csv(f"{config.store.workdir}/input/fnc.csv")
    train_df = cudf.read_csv(f"{config.store.workdir}/input/train_scores.csv")
    submissoin = cudf.read_csv(f"{config.store.workdir}/input/sample_submission.csv")
    train_df = train_df.merge(loading, on="Id", how="left")
    train_df = train_df.merge(fnc, on="Id", how="left")
    test_df = submissoin["Id"].str.split("_")[0].unique().astype(int)
    test_df = cudf.DataFrame({"Id": test_df})
    test_df = test_df.merge(loading, on="Id", how="left")
    test_df = test_df.merge(fnc, on="Id", how="left")
    # mean shift
    train_df["IC_20"] += 0.0022449734660541093
    # Scaling
    train_df[fnc.columns[1:]] /= 500
    test_df[fnc.columns[1:]] /= 500

    return train_df, test_df


def make_submission(test_df: cudf.DataFrame) -> cudf.DataFrame:
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
            store_config.gcs_path, filename.split(store_config.save_path)[-1][1:],
        )
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(filename)


@hydra.main(config_path="yamls/svm.yaml")
def main(config: DictConfig) -> None:
    prepair_dir(config)
    train_df, test_df = load_data(config)
    label_cols = ["age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]
    feature_cols = [col for col in train_df.columns if col not in label_cols + ["Id"]]
    # From adversarial valiadtion
    # feature_cols.remove("IC_20")
    train_df["age_rank"] = train_df["age"] // 10 * 10
    age_rank = train_df["age_rank"].to_array()
    if config.randomize_age:
        set_seed(100)
        train_df["age"] += [randomize_age(age) for age in train_df["age"]]

    skf = StratifiedKFold(n_splits=5, random_state=config.data.seed, shuffle=True)
    for label_col, c, epsilon in zip(label_cols, [50, 5, 5, 5, 5], [1, 1, 1, 1, 1]):
        y_oof = np.zeros(train_df.shape[0])
        for n_fold, (train_index, val_index) in enumerate(
            skf.split(age_rank, age_rank)
        ):
            train_df_fold = train_df.iloc[train_index]
            valid_df_fold = train_df.iloc[val_index]
            train_df_fold = train_df_fold[train_df_fold[label_col].notnull()]
            model = SVR(kernel=config.kernel, C=c, cache_size=3000.0)
            model.fit(train_df_fold[feature_cols], train_df_fold[label_col])
            y_oof[val_index] = model.predict(valid_df_fold[feature_cols]).to_array()
            test_df[f"{label_col}_pred_fold{n_fold}"] = model.predict(
                test_df[feature_cols]
            )
        train_df[f"{label_col}_pred"] = y_oof
        notnull_idx = train_df[label_col].notnull()
        score = normalized_absolute_errors(
            train_df[notnull_idx][label_col].values,
            train_df[notnull_idx][f"{label_col}_pred"].values,
        )
        logger.info(f"{label_col}, score: {score}")
        test_df[label_col] = test_df[
            [f"{label_col}_pred_fold{i}" for i in range(5)]
        ].mean(1)
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
