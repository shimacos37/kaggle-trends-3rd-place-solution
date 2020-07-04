import logging
from typing import Tuple
import pandas as pd
import numpy as np
from src.metrics import weighted_normalized_absolute_errors


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lgbm_train = (
        pd.read_csv("./output/lgbm_stacking/result/lgbm_stacking_train.csv")
        .sort_values("Id")
        .reset_index(drop=True)
    )
    svm_train = (
        pd.read_csv("./output/svm_stacking/result/svm_stacking_train.csv")
        .sort_values("Id")
        .reset_index(drop=True)
    )

    lgbm_sub = (
        pd.read_csv("./output/lgbm_stacking/result/lgbm_stacking_submission.csv")
        .sort_values("Id")
        .reset_index(drop=True)
    )
    svm_sub = (
        pd.read_csv("./output/svm_stacking/result/svm_stacking_submission.csv")
        .sort_values("Id")
        .reset_index(drop=True)
    )
    return lgbm_train, svm_train, lgbm_sub, svm_sub


def main() -> None:
    lgbm_train, svm_train, lgbm_sub, svm_sub = load_data()
    label_cols = ["age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]
    pred_cols = [f"{label_col}_pred" for label_col in label_cols]

    target = lgbm_train[label_cols].values

    pred = np.average(
        [lgbm_train[pred_cols].values, svm_train[pred_cols].values],
        axis=0,
        weights=[0.5, 0.5],
    )
    score = weighted_normalized_absolute_errors(target, pred)
    logger.info(f"simple mean: {score}")

    best = np.inf
    for weight1 in np.linspace(0, 1, 101):
        pred = np.average(
            [lgbm_train.loc[:, pred_cols].values, svm_train.loc[:, pred_cols].values],
            axis=0,
            weights=[weight1, 1 - weight1],
        )
        score = weighted_normalized_absolute_errors(target[:], pred,)
        if best >= score:
            best = score
            best_weight = [weight1, 1 - weight1]

    pred = np.average(
        [lgbm_train[pred_cols].values, svm_train[pred_cols].values],
        axis=0,
        weights=best_weight,
    )
    score = weighted_normalized_absolute_errors(target, pred)
    logger.info(f"weighted mean: {score}")

    lgbm_sub["Predicted"] = np.average(
        [lgbm_sub["Predicted"].values, svm_sub["Predicted"].values],
        axis=0,
        weights=best_weight,
    )

    lgbm_sub[["Id", "Predicted"]].to_csv(
        "./output/blending_lgbm_svm_stacking.csv", index=False,
    )


if __name__ == "__main__":
    main()
