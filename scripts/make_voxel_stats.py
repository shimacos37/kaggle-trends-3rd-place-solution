import re
from glob import glob

import h5py
import nilearn as nl
import numpy as np
import pandas as pd
import scipy
from nilearn.input_data import NiftiMasker
from tqdm import tqdm


def load_subject(filename, mask_niimg):
    """
    Load a subject saved in .mat format with
        the version 7.3 flag. Return the subject
        niimg, using a mask niimg as a template
        for nifti headers.
        
    Args:
        filename    <str>            the .mat filename for the subject data
        mask_niimg  niimg object     the mask niimg object used for nifti headers
    """
    subject_data = None
    with h5py.File(filename, "r") as f:
        subject_data = f["SM_feature"][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
    subject_niimg = nl.image.new_img_like(
        mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True
    )
    return subject_niimg


def main():
    image_paths = glob("./input/fMRI/*")
    # numbers = pd.read_csv("./input/ICN_numbers.csv")
    mask_niimg = nl.image.load_img("./input/fMRI_mask.nii")
    ids = [re.split("[/.]", path)[-2] for path in image_paths]
    masker = NiftiMasker(
        mask_niimg,
        target_affine=mask_niimg.affine,
        mask_strategy="epi",
        mask_args=dict(upper_cutoff=0.9, lower_cutoff=0.8, opening=False),
    )
    # inc_dict = {
    #     "ADN": np.array([5, 6]),
    #     "CBN": np.array([49, 50, 51, 52]),
    #     "CON": np.array(
    #         [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    #     ),
    #     "DMN": np.array([42, 43, 44, 45, 46, 47, 48]),
    #     "SCN": np.array([1, 2, 3, 4]),
    #     "SMN": np.array([7, 8, 9, 10, 11, 12, 13, 14, 15]),
    #     "VSN": np.array([16, 17, 18, 19, 20, 21, 22, 23, 24]),
    # }
    stats = ["mean", "std", "max", "min", "kurt", "skew"]
    # df_dict = {f"{key}_{stat}": [] for key, index in inc_dict.items() for stat in stats}
    df_dict = {}
    for i in range(53):
        for stat in stats:
            df_dict[f"COMPONENTS_{i}_{stat}"] = []

    for path in tqdm(image_paths):
        img = load_subject(path, mask_niimg)
        img = masker.fit_transform(img)
        for index in range(img.shape[0]):
            df_dict[f"COMPONENTS_{index}_mean"].append(img[index].mean())
            df_dict[f"COMPONENTS_{index}_std"].append(img[index].std())
            df_dict[f"COMPONENTS_{index}_max"].append(img[index].max())
            df_dict[f"COMPONENTS_{index}_min"].append(img[index].min())
            df_dict[f"COMPONENTS_{index}_kurt"].append(scipy.stats.kurtosis(img[index]))
            df_dict[f"COMPONENTS_{index}_skew"].append(scipy.stats.skew(img[index]))

    # img = img[:, 1:49, 4:60, 3:48]
    # for key, index in inc_dict.items():
    #     df_dict[f"{key}_mean"].append(img[index].mean())
    #     df_dict[f"{key}_std"].append(img[index].std())
    #     df_dict[f"{key}_max"].append(img[index].max())
    #     df_dict[f"{key}_min"].append(img[index].min())
    #     df_dict[f"{key}_kurt"].append(scipy.stats.kurtosis(img[index]))
    #     df_dict[f"{key}_skew"].append(scipy.stats.skew(img[index]))
    df_dict["Id"] = ids
    result = pd.DataFrame(df_dict)
    result.to_csv("./input/voxel_stats.csv", index=False)


if __name__ == "__main__":
    main()
