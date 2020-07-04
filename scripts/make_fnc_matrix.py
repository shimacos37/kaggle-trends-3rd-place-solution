# from functools import partial
from typing import List, Dict
import numpy as np
import pandas as pd

# from joblib import Parallel, delayed


def save_fnc_matrix(
    fnc: pd.DataFrame,
    fnc_cols: List[str],
    name_matrix: Dict[str, List[int]],
    index: int,
) -> None:
    con_matrix = np.zeros((53, 53))
    for col in fnc_cols:
        r_, c_ = name_matrix[col]
        con_matrix[c_, r_] = fnc.iloc[index, :][col]
    # And now add the transpose - its symmetrix
    con_matrix += con_matrix.T
    np.save(f'./input/fnc_mat/{fnc.iloc[index]["Id"]}_corr.npy', con_matrix)


def main():
    fnc = pd.read_csv("./input/fnc.csv")
    icn_number = pd.read_csv("./input/ICN_numbers.csv")
    fnc_cols = fnc.columns.to_list()[1:]
    fnc_cols_filtered = [col.split("_")[0] for col in fnc_cols]
    # Network index:
    ntwk_idx = {}
    network_names = np.unique([i[:3] for i in fnc_cols_filtered])
    for ii in network_names:
        ntwk_idx[ii] = np.unique(
            [
                np.int(i.split("(")[-1].split(")")[0])
                for i in fnc_cols_filtered
                if ii in i
            ]
        )

    # Look up matrix index
    icn_idx = {}
    for key in ntwk_idx.keys():
        icn_idx[key] = np.array(
            icn_number.index[icn_number.ICN_number.isin(ntwk_idx[key])]
        )

    # This is probably totally inefficient - but let's try it
    icn_mat_idx = icn_number.T.to_dict("list")
    # Reverse the matrix:
    icn_mat_idx = {value[0]: key for key, value in icn_mat_idx.items()}
    # Map names to indices
    name_matrix = {}

    for col in fnc_cols:
        name_matrix[col] = [
            np.int(icn_mat_idx[np.int(i.split(")")[0])]) for i in col.split("(")[1:]
        ]

    # save_fnc_matrix_fn = partial(
    #     save_fnc_matrix, fnc=fnc, fnc_cols=fnc_cols, name_matrix=name_matrix
    # )
    # _ = Parallel(n_jobs=-1, verbose=1)(
    #     [delayed(save_fnc_matrix_fn)(index=i) for i in range(fnc.shape[0])]
    # )
    fnc_mean = fnc[fnc_cols].mean(0)
    con_matrix = np.zeros((53, 53))
    for col in fnc_cols:
        r_, c_ = name_matrix[col]
        con_matrix[c_, r_] = fnc_mean[col]
    # And now add the transpose - its symmetrix
    con_matrix += con_matrix.T
    np.save("./input/fnc_adj.npy", con_matrix)


if __name__ == "__main__":
    main()
