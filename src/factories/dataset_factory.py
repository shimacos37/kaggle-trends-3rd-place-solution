import albumentations as A
import h5py
import nilearn as nl
import numpy as np
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker
from rising.loading import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from src.randomize import randomize_age


class NeuroDataset(Dataset):
    def __init__(self, data_config, mode="train"):

        loading = pd.read_csv(f"{data_config.workdir}/input/loading.csv")
        fnc = pd.read_csv(f"{data_config.workdir}/input/fnc.csv")
        voxel_stats = pd.read_csv(f"{data_config.workdir}/input/voxel_stats.csv")
        icn_number = pd.read_csv(f"{data_config.workdir}/input/ICN_numbers.csv")

        if mode != "test":
            df = pd.read_csv(data_config.csv_path)
            df["age_rank"] = df["age"] // 10 * 10
            skf = StratifiedKFold(
                n_splits=5, random_state=data_config.seed, shuffle=True
            )
            for i, (train_index, val_index) in enumerate(skf.split(df, df["age_rank"])):
                df.loc[val_index, "fold"] = i
        else:
            submission = pd.read_csv(
                f"{data_config.workdir}/input/sample_submission.csv"
            )
            df = pd.DataFrame({"Id": submission["Id"].str[:5].unique().astype(int)})
            train_df = pd.read_csv(data_config.csv_path)
            train_df = train_df.merge(loading, on="Id", how="left")
            train_df = train_df.merge(fnc, on="Id", how="left")
            train_df = train_df.merge(voxel_stats, on="Id", how="left")
            train_df["age_rank"] = train_df["age"] // 10 * 10
            skf = StratifiedKFold(
                n_splits=5, random_state=data_config.seed, shuffle=True
            )
            for i, (train_index, val_index) in enumerate(
                skf.split(train_df, train_df["age_rank"])
            ):
                train_df.loc[val_index, "fold"] = i

        df = df.merge(loading, on="Id", how="left")
        df = df.merge(fnc, on="Id", how="left")
        df = df.merge(voxel_stats, on="Id", how="left")

        self.mask_niimg = nl.image.load_img(
            f"{data_config.workdir}/input/fMRI_mask.nii"
        )
        self.label_cols = data_config.label_cols
        self.fnc_cols = fnc.columns[1:]
        self.loading_cols = [col for col in loading.columns[1:]]
        self.voxel_stat_cols = [col for col in voxel_stats.columns[:-1]]
        self.feature_cols = [
            col
            for col in df.columns
            if col not in self.label_cols + ["Id", "age_rank", "fold", "IC_20"]
        ]
        # Reverse the matrix:
        icn_mat_idx = icn_number.T.to_dict("list")
        icn_mat_idx = {
            i[0]: j for i, j in zip(icn_mat_idx.values(), icn_mat_idx.keys())
        }
        # Map names to indices
        self.name_matrix = {}

        for fnco in self.fnc_cols:
            self.name_matrix[fnco] = [
                np.int(icn_mat_idx[np.int(i.split(")")[0])])
                for i in fnco.split("(")[1:]
            ]

        self.config = data_config
        self.mode = mode
        self.n_components = 53
        std = StandardScaler()

        if mode == "train":
            df[self.feature_cols] = std.fit_transform(df[self.feature_cols])
            if data_config.n_fold != "all":
                self.df = df.query(f"fold!={data_config.n_fold}").reset_index(drop=True)
                self.df[self.feature_cols] = std.fit_transform(
                    self.df[self.feature_cols]
                )
            else:
                self.df = df
        elif mode == "valid":
            df[self.feature_cols] = std.fit_transform(df[self.feature_cols])
            if data_config.n_fold != "all":
                std.fit(df.query(f"fold!={data_config.n_fold}")[self.feature_cols])
                self.df = df.query(f"fold=={data_config.n_fold}").reset_index(drop=True)
                self.df[self.feature_cols] = std.transform(self.df[self.feature_cols])
            else:
                self.df = df.sample(df.shape[0] // 2)

        elif mode == "test":
            std.fit(train_df.query(f"fold!={data_config.n_fold}")[self.feature_cols])
            df[self.feature_cols] = std.transform(df[self.feature_cols])
            self.df = df

    def load_img(self, path):
        with h5py.File(path, "r") as f:
            img = f["SM_feature"][()]
        img = img[:, 1:49, 4:60, 3:48]
        return img

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        imgs : [n_components, height, width, n_components]
        """
        id_ = self.df.loc[idx, "Id"]
        fnc_feature = self.df.loc[idx, self.fnc_cols].values
        loading_feature = self.df.loc[idx, self.loading_cols].values
        img_path = f"{self.config.workdir}/input/fMRI/{id_}.mat"
        imgs = self.load_img(img_path)
        if self.mode != "test":
            labels = self.df.loc[idx, self.label_cols].values
            return {
                "Id": id_,
                "data": imgs.astype(np.float32),
                "fnc": fnc_feature,
                "loading": loading_feature,
                "label": labels,
            }
        else:
            return {
                "Id": id_,
                "data": imgs.astype(np.float32),
                "fnc": fnc_feature,
                "loading": loading_feature,
            }


class MlpDataset(NeuroDataset):
    def __init__(self, data_config, mode="train"):
        super(MlpDataset, self).__init__(data_config, mode)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, "Id"]
        fnc_feature = self.df.loc[idx, self.fnc_cols].values
        loading_feature = self.df.loc[idx, self.loading_cols].values
        voxel_feature = self.df.loc[idx, self.voxel_stat_cols].values
        if self.mode != "test":
            labels = self.df.loc[idx, self.label_cols].values
            if self.mode == "train":
                labels[0] = randomize_age(labels[0])
            return {
                "Id": id_,
                "fnc": fnc_feature,
                "loading": loading_feature,
                "voxel": voxel_feature,
                "label": labels,
            }
        else:
            return {
                "Id": id_,
                "fnc": fnc_feature,
                "loading": loading_feature,
                "voxel": voxel_feature,
            }


class AutoEncoderDataset(Dataset):
    def __init__(self, data_config, mode="train"):
        super(AutoEncoderDataset, self).__init__()
        self.config = data_config
        self.mode = mode
        df = pd.read_csv(f"{data_config.workdir}/input/loading.csv")[["Id"]]
        train_ids = pd.read_csv(f"{data_config.workdir}/input/train_scores.csv")[
            "Id"
        ].values
        df.loc[df.query("Id in @train_ids").index, "is_train"] = 1
        df["is_train"].fillna(0, inplace=True)
        skf = StratifiedKFold(n_splits=5, random_state=data_config.seed, shuffle=True)
        for i, (train_index, val_index) in enumerate(skf.split(df, df["is_train"])):
            df.loc[val_index, "fold"] = i
        if mode == "train":
            self.df = df.query(f"fold!={data_config.n_fold}").reset_index(drop=True)
        elif mode == "valid":
            self.df = df.query(f"fold=={data_config.n_fold}").reset_index(drop=True)

    def __len__(self):
        return self.df.shape[0]

    def load_img(self, path):
        with h5py.File(path, "r") as f:
            img = f["SM_feature"][()]
        img = img[:, 1:49, 4:60, 3:48]
        return img

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, "Id"]
        img_path = f"{self.config.workdir}/input/fMRI/{id_}.mat"
        imgs = self.load_img(img_path).astype(np.float32)

        if self.mode != "test":
            return {
                "Id": id_,
                "data": imgs,
                "label": imgs,
            }
        else:
            return {
                "Id": id_,
                "label": imgs,
            }


class TransoformerDataset(NeuroDataset):
    def __init__(self, data_config, mode="train"):
        super(TransoformerDataset, self).__init__(data_config, mode)
        self.masker = NiftiLabelsMasker(
            labels_img=f"{data_config.workdir}/input/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz",
            standardize=True,
        )

    def __len__(self):
        return self.df.shape[0]

    def _mask(self, img):
        """
        return : 
            shape: [53, 400] ([n_component, n_label])
        """
        img = nl.image.new_img_like(
            self.mask_niimg, img, affine=self.mask_niimg.affine, copy_header=True
        )
        masked_data = self.masker.fit_transform(img)
        return masked_data

    def load_img(self, path):
        with h5py.File(path, "r") as f:
            img = f["SM_feature"][()]
        img = np.moveaxis(img, [0, 1, 2, 3], [3, 2, 1, 0])
        return img

    def __getitem__(self, idx):
        """
        imgs : [n_components, height, width, n_components]
        """
        id_ = self.df.loc[idx, "Id"]
        feature = self.df.loc[idx, self.feature_cols].values
        fnc_feature = self.df.loc[idx, self.fnc_cols].values
        loading_feature = self.df.loc[idx, self.loading_cols].values
        if self.mode != "test":
            img_path = f"{self.config.workdir}/input/fMRI/{id_}.mat"
        else:
            img_path = f"{self.config.workdir}/input/fMRI/{id_}.mat"
        imgs = self.load_img(img_path)
        masked_data = self._mask(imgs)
        if self.mode != "test":
            labels = self.df.loc[idx, self.label_cols].values
            return {
                "Id": id_,
                "data": masked_data.astype(np.float32),
                "fnc": fnc_feature,
                "loading": loading_feature,
                "feature": feature,
                "label": labels,
            }
        else:
            return {
                "Id": id_,
                "data": masked_data.astype(np.float32),
                "loading": loading_feature,
                "fnc": fnc_feature,
                "feature": feature,
            }


class ImgDataset(NeuroDataset):
    def __init__(self, data_config, mode="train"):
        super(ImgDataset, self).__init__(data_config, mode)
        self.icn_dict = {
            "ADN": np.array([5, 6]),
            "CBN": np.array([49, 50, 51, 52]),
            "CON": np.array(
                [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
            ),
            "DMN": np.array([42, 43, 44, 45, 46, 47, 48]),
            "SCN": np.array([0, 1, 2, 3, 4]),
            "SMN": np.array([7, 8, 9, 10, 11, 12, 13, 14, 15]),
            "VSN": np.array([16, 17, 18, 19, 20, 21, 22, 23, 24]),
        }

    def __len__(self):
        return self.df.shape[0]

    def _load_augmentaion(self):
        if self.mode == "train":
            if not self.refinement_step:
                p = 0.3
                aug = A.Compose(
                    [
                        A.Blur(p=p, blur_limit=(3, 7)),
                        A.GaussNoise(always_apply=False, p=p, var_limit=1),
                        A.RandomBrightnessContrast(
                            p=p,
                            brightness_limit=(-0.2, 0.2),
                            contrast_limit=(-0.2, 0.2),
                            brightness_by_max=True,
                        ),
                    ],
                )
            else:
                aug = None
        else:
            aug = None
        return aug

    def _split_img(self, img):
        """
        return :
            shape: [7, 52, 63, 53]
        """
        img = np.array([img[index].mean(0) for index in self.icn_dict.values()])
        return img

    def load_img(self, path):
        with h5py.File(path, "r") as f:
            img = f["SM_feature"][()]
        img = img[:, 1:49, 4:60, 3:48]
        img = img.sum(0)
        return img

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, "Id"]
        feature = self.df.loc[idx, self.feature_cols].values
        fnc_feature = self.df.loc[idx, self.fnc_cols].values
        loading_feature = self.df.loc[idx, self.loading_cols].values
        if self.mode != "test":
            img_path = f"{self.config.workdir}/input/fMRI/{id_}.mat"
        else:
            img_path = f"{self.config.workdir}/input/fMRI/{id_}.mat"
        imgs = self.load_img(img_path).astype(np.float32)
        aug = self._load_augmentaion()
        if aug is not None:
            imgs = aug(image=imgs)["image"]
        imgs = (imgs - imgs.mean()) / imgs.std()
        if self.mode != "test":
            labels = self.df.loc[idx, self.label_cols].values
            return {
                "Id": id_,
                "data": imgs.astype(np.float32),
                "fnc": fnc_feature,
                "loading": loading_feature,
                "feature": feature,
                "label": labels,
            }
        else:
            return {
                "Id": id_,
                "data": imgs.astype(np.float32),
                "loading": loading_feature,
                "fnc": fnc_feature,
                "feature": feature,
            }


def get_normal_dataset(data_config, mode):
    dataset = NeuroDataset(data_config, mode)
    return dataset


def get_mlp_dataset(data_config, mode):
    dataset = MlpDataset(data_config, mode)
    return dataset


def get_transformer_dataset(data_config, mode):
    dataset = TransoformerDataset(data_config, mode)
    return dataset


def get_img_dataset(data_config, mode):
    dataset = ImgDataset(data_config, mode)
    return dataset


def get_auto_encoder_dataset(data_config, mode):
    dataset = AutoEncoderDataset(data_config, mode)
    return dataset


def get_dataset(data_config, mode):
    print("dataset name:", data_config.dataset_name)
    f = globals().get("get_" + data_config.dataset_name)
    return f(data_config, mode)
