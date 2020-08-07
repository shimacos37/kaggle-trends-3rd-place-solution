import hashlib
import multiprocessing as mp
import os
import random
import re
import shutil
from glob import glob
from itertools import chain
from typing import Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rising.transforms as rtr
import torch
import torch.distributed as dist
import wandb
import yaml
from matplotlib import gridspec
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from rising.loading import DataLoader
from rising.random.abstract import AbstractParameter
from torch.distributions import Distribution as TorchDistribution

from src.factories import (
    MyCallback,
    WandbLogger,
    get_dataset,
    get_loss,
    get_model,
    get_optimizer,
    get_scheduler,
)
from src.sync_batchnorm import convert_model
from src.metrics import (
    normalized_absolute_errors,
    weighted_normalized_absolute_errors,
)

plt.style.use("ggplot")


def set_seed(seed):
    os.environ.PYTHONHASHSEED = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


def prepair_dir(config):
    """
    Logの保存先を作成
    """
    for path in [
        config.store.result_path,
        config.store.log_path,
        config.store.model_path,
    ]:
        if (
            os.path.exists(path)
            and config.train.warm_start is False
            and config.data.is_train
        ):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def set_up(config):
    # Setup
    prepair_dir(config)
    set_seed(config.train.seed)
    for device in config.base.gpu_id:
        torch.cuda.set_device(f"cuda:{device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(config.base.gpu_id)


class ContinuousParameter(AbstractParameter):
    """Class to perform parameter sampling from torch distributions"""

    def __init__(self, distribution: TorchDistribution):
        """
        Args:
            distribution : the distribution to sample from
        """
        super().__init__()
        self.dist = distribution

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Samples from the internal distribution

        Args:
            n_samples : the number of elements to sample

        Returns
            torch.Tensor: samples
        """
        return self.dist.sample((n_samples,)).cuda()


class UniformParameter(ContinuousParameter):
    """
    Samples Parameters from a uniform distribution.
    For details have a look at :class:`torch.distributions.Uniform`
    """

    def __init__(
        self, low: Union[float, torch.Tensor], high: Union[float, torch.Tensor]
    ):
        """
        Args:
            low : the lower range (inclusive)
            high : the higher range (exclusive)
        """
        super().__init__(torch.distributions.Uniform(low=low, high=high))


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # Setting
        self.hparams = hparams
        self.base_config = hparams.base
        self.data_config = hparams.data
        self.model_config = hparams.model
        self.train_config = hparams.train
        self.test_config = hparams.test
        self.store_config = hparams.store
        self.cpu_count = mp.cpu_count() // len(self.base_config.gpu_id)
        # load from factories
        self.model = get_model(hparams.model).cuda()
        weight_params = [
            p for name, p in self.model.named_parameters() if "bias" not in name
        ]
        bias_params = [p for name, p in self.model.named_parameters() if "bias" in name]
        if len(self.base_config.gpu_id) > 1:
            self.model = convert_model(self.model)
        self.optimizer = get_optimizer(
            opt_name=self.base_config.opt_name,
            params=[
                {"params": weight_params, "weight_decay": 0},
                {"params": bias_params, "weight_decay": 0},
            ],
            lr=self.train_config.learning_rate,
        )
        if self.data_config.is_train:
            self.train_dataset = get_dataset(data_config=self.data_config, mode="train")
            self.valid_dataset = get_dataset(data_config=self.data_config, mode="valid")
        else:
            if self.test_config.is_validation:
                self.test_dataset = get_dataset(
                    data_config=self.data_config, mode="valid"
                )
                self.prefix = "valid"
            else:
                self.test_dataset = get_dataset(
                    data_config=self.data_config, mode="test"
                )
                self.prefix = "test"
        self.scheduler = get_scheduler(
            scheduler_name="plateau_scheduler", optimizer=self.optimizer
        )

        if self.base_config.loss_name != "arcface":
            self.loss = get_loss(
                loss_name=self.base_config.loss_name, weights=self.data_config.weights
            )
        else:
            self.loss = get_loss(
                loss_name=self.base_config.loss_name, in_features=self.model.in_features
            )
        # path setting
        self.initialize_variables()
        self.save_flg = False
        self.refinement_step = False

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_nb):

        pred = self.forward(**batch)
        loss = self.loss(pred, batch["label"]) * self.train_config.accumulation_steps
        metrics = {}
        metrics["loss"] = loss
        metrics["log"] = {
            "train_loss": loss,
        }

        return metrics

    def validation_step(self, batch, batch_nb):
        pred = self.forward(**batch)
        loss = self.loss(pred, batch["label"])
        if isinstance(pred, list):
            pred = pred[-1]
        metrics = {
            "Id": batch["Id"],
            "preds": pred,
            "labels": batch["label"],
            "loss": loss,
        }
        if self.store_config.save_feature:
            feature = self.model.get_feature()
            metrics.update({"feature": feature})

        return metrics

    def test_step(self, batch, batch_nb):
        pred = self.forward(**batch)
        if isinstance(pred, list):
            pred = pred[-1]
        metrics = {
            "Id": batch["Id"],
            "preds": pred,
        }
        if self.store_config.save_feature:
            feature = self.model.get_feature()
            metrics.update({"feature": feature})

        return metrics

    def validation_epoch_end(self, outputs):
        preds = np.concatenate(
            [x["preds"].detach().cpu().numpy() for x in outputs], axis=0
        )
        labels = np.concatenate(
            [x["labels"].detach().cpu().numpy() for x in outputs], axis=0
        )
        loss = np.mean([x["loss"].detach().cpu().numpy() for x in outputs])
        ids = list(
            chain.from_iterable([x["Id"].detach().cpu().numpy() for x in outputs])
        )
        label_cols = self.valid_dataset.label_cols
        df_dict = {"Id": ids}
        for i, label_col in enumerate(label_cols):
            df_dict[f"{label_col}_pred"] = preds[:, i]
            df_dict[label_col] = labels[:, i]
        df = pd.DataFrame(df_dict)
        if self.store_config.save_feature:
            feature = np.concatenate(
                [x["feature"].detach().cpu().numpy() for x in outputs], axis=0
            )
            for i in range(feature.shape[-1]):
                df[f"feature{i}"] = feature[:, i]
        # For handling log_loss None Error
        results = {
            f"{label_col}_nae": normalized_absolute_errors(
                df[label_col].values, df[f"{label_col}_pred"].values
            )
            for label_col in label_cols
        }
        avg_score = weighted_normalized_absolute_errors(
            df[label_cols].values,
            df[[f"{col}_pred" for col in label_cols]].values,
            weights=self.data_config.weights,
        ).astype(np.float32)
        if self.use_ddp:
            metrics = {"avg_loss": loss, "avg_score": avg_score}
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            aggregated_metrics = {}
            for metric_name, metric_val in metrics.items():
                metric_tensor = torch.tensor(metric_val).to(f"cuda:{rank}")
                dist.barrier()
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                reduced_metric = metric_tensor.item() / world_size
                aggregated_metrics[metric_name] = reduced_metric
            loss = aggregated_metrics["avg_loss"]
            avg_score = aggregated_metrics["avg_score"]
        else:
            rank = 0
        res = {}
        res["step"] = int(self.global_step)
        res["epoch"] = int(self.current_epoch)
        if avg_score <= self.best_score:
            self.best_score = avg_score
            self.save_flg = True
            res["best_score"] = float(self.best_score)
            df.to_csv(
                os.path.join(self.store_config.result_path, f"valid_result_{rank}.csv"),
                index=False,
            )
            with open(
                os.path.join(self.store_config.log_path, "best_score.yaml"), "w"
            ) as f:
                yaml.dump(res, f, default_flow_style=False)
        metrics = {}
        metrics["progress_bar"] = {
            "val_loss": avg_score,
            "avg_val_score": torch.tensor(avg_score),
            "best_score": self.best_score,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        metrics["progress_bar"].update(results)
        metrics["log"] = {
            "val_loss": avg_score,
            "avg_val_score": torch.tensor(avg_score),
            "best_score": self.best_score,
        }
        metrics["log"].update(results)
        return metrics

    def on_epoch_end(self):
        if self.use_ddp:
            rank = dist.get_rank()
        else:
            rank = 0
        if rank == 0:
            paths = sorted(
                glob(
                    os.path.join(
                        self.store_config.result_path, "valid_result_[0-9].csv"
                    )
                )
            )
            if self.save_flg:
                label_cols = self.valid_dataset.label_cols
                all_df = pd.concat([pd.read_csv(path) for path in paths])
                all_df = all_df.drop_duplicates()
                all_df.to_csv(
                    os.path.join(self.store_config.result_path, "valid_result_all.csv"),
                    index=False,
                )
                if self.logger is not None:
                    gs = gridspec.GridSpec(1, 5)
                    fig = plt.figure(figsize=(20, 4))
                    for n, label_col in enumerate(label_cols):
                        ax = fig.add_subplot(gs[n])
                        all_df.plot.scatter(
                            x=label_col, y=f"{label_col}_pred", ax=ax, alpha=0.5
                        )
                        ax.plot(
                            np.arange(all_df[label_col].min(), all_df[label_col].max()),
                            np.arange(all_df[label_col].min(), all_df[label_col].max()),
                        )
                    fig.tight_layout()
                    self.logger.experiment.log(
                        {"predict vs label scatter plot": wandb.Image(fig)},
                        step=self.global_step,
                    )
                    plt.close(fig)
                    gs = gridspec.GridSpec(1, 5)
                    fig = plt.figure(figsize=(20, 4))
                    for n, label_col in enumerate(label_cols):
                        ax = fig.add_subplot(gs[n])
                        all_df[label_col].hist(bins=100, ax=ax, alpha=0.5, label="true")
                        all_df[f"{label_col}_pred"].hist(
                            bins=100, ax=ax, alpha=0.5, label="pred"
                        )
                        plt.legend()
                        ax.set_title(label_col)
                    fig.tight_layout()
                    self.logger.experiment.log(
                        {"predict distribution": wandb.Image(fig)},
                        step=self.global_step,
                    )
                    self.save_flg = False
                    plt.close(fig)
        if self.current_epoch >= self.train_config.refinement_step:
            self.refinement_step = True

    def test_epoch_end(self, outputs):
        preds = np.concatenate(
            [x["preds"].detach().cpu().numpy() for x in outputs], axis=0
        )
        ids = list(
            chain.from_iterable([x["Id"].detach().cpu().numpy() for x in outputs])
        )
        label_cols = self.test_dataset.label_cols
        if not self.test_config.is_validation:
            df_dict = {"Id": ids}
            for i, label_col in enumerate(label_cols):
                df_dict[f"{label_col}_pred_fold{self.data_config.n_fold}"] = preds[:, i]
            if self.store_config.save_feature:
                feature = np.concatenate(
                    [x["feature"].detach().cpu().numpy() for x in outputs], axis=0
                )
                for i in range(feature.shape[-1]):
                    df_dict[f"feature{i}_fold{self.data_config.n_fold}"] = feature[:, i]
            df = pd.DataFrame(df_dict)
            df.to_csv(
                os.path.join(self.store_config.result_path, "test_result.csv"),
                index=False,
            )
            if self.data_config.n_fold == 4:
                sub_dict = {}
                dfs = [
                    pd.read_csv(f"../fold{i}/result/test_result.csv") for i in range(5)
                ]
                sub_dict["Id"] = dfs[0]["Id"].values
                ids = []
                preds = []
                for label_col in label_cols:
                    preds.append(
                        np.mean(
                            [
                                dfs[i][f"{label_col}_pred_fold{i}"].values
                                for i in range(5)
                            ],
                            axis=0,
                        )
                    )
                    ids.append([f"{id_}_{label_col}" for id_ in dfs[0]["Id"].values])
                sub = pd.DataFrame(
                    {"Id": np.concatenate(ids), "Predicted": np.concatenate(preds)}
                )
                sub.to_csv(
                    f"../{self.store_config.model_name}_submission.csv", index=False
                )
            return {}
        else:
            df = self.test_dataset.df
            for i, label_col in enumerate(label_cols):
                df[f"{label_col}_pred"] = preds[:, i]
            if self.store_config.save_feature:
                feature = np.concatenate(
                    [x["feature"].detach().cpu().numpy() for x in outputs], axis=0
                )
                for i in range(feature.shape[-1]):
                    df[f"feature{i}"] = feature[:, i]
            df.to_csv(
                os.path.join(
                    self.store_config.result_path, f"{self.prefix}_result.csv"
                ),
                index=False,
            )
            result = {}
            if self.data_config.n_fold == 4:
                dfs = pd.concat(
                    [
                        pd.read_csv(f"../fold{i}/result/valid_result.csv")
                        for i in range(5)
                    ],
                    axis=0,
                )
                dfs.to_csv(f"../{self.store_config.model_name}_train.csv", index=False)
                score = self.weighted_normalized_absolute_errors(
                    dfs[label_cols].values.copy(),
                    dfs[[f"{col}_pred" for col in label_cols]].values.copy(),
                    weights=self.data_config.weights,
                )
                result = {"weighted_normalized_absolute_error": score}

            return result

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def _3d_augmenation(self, mode="train"):
        if mode == "train" and not self.refinement_step:
            aug = rtr.Compose(
                [
                    rtr.OneOf(
                        [
                            rtr.DoNothing(),
                            rtr.GaussianNoise(mean=0, std=1, keys=["data"]),
                            rtr.RandomScaleValue(
                                UniformParameter(0.8, 1.2),
                                per_channel=True,
                                keys=["data"],
                            ),
                            rtr.RandomAddValue(
                                UniformParameter(0.0, 0.2),
                                per_channel=True,
                                keys=["data"],
                            ),
                        ],
                        weights=[0.5, 0.2, 0.2, 0.1],
                    ),
                    rtr.NormZeroMeanUnitStd(keys=["data"]),
                ]
            )
        else:
            aug = rtr.Compose([rtr.NormZeroMeanUnitStd(keys=["data"])])
        return aug

    def train_dataloader(self):
        if self.data_config.dataset_name == "normal_dataset":
            transform = self._3d_augmenation("train")
        else:
            transform = None
        if self.use_ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, shuffle=True
            )
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.train_config.batch_size,
                num_workers=self.cpu_count,
                pin_memory=True,
                sampler=sampler,
                drop_last=True,
                gpu_transforms=transform,
            )
        else:
            sampler = torch.utils.data.sampler.RandomSampler(self.train_dataset)
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.train_config.batch_size,
                pin_memory=True,
                sampler=sampler,
                drop_last=True,
                num_workers=self.cpu_count,
                gpu_transforms=transform,
            )
        return train_loader

    def val_dataloader(self):
        if self.data_config.dataset_name == "normal_dataset":
            transform = self._3d_augmenation("valid")
        else:
            transform = None
        if self.use_ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.valid_dataset, shuffle=False
            )
            valid_loader = DataLoader(
                self.valid_dataset,
                batch_size=self.test_config.batch_size,
                num_workers=self.cpu_count,
                pin_memory=True,
                sampler=sampler,
                gpu_transforms=transform,
            )
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(self.valid_dataset)
            valid_loader = DataLoader(
                self.valid_dataset,
                batch_size=self.test_config.batch_size,
                pin_memory=True,
                sampler=sampler,
                num_workers=self.cpu_count,
                gpu_transforms=transform,
            )

        return valid_loader

    def test_dataloader(self):
        if self.data_config.dataset_name == "normal_dataset":
            transform = self._3d_augmenation("test")
        else:
            transform = None
        sampler = torch.utils.data.sampler.SequentialSampler(self.test_dataset)
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.test_config.batch_size,
            pin_memory=True,
            sampler=sampler,
            num_workers=self.cpu_count,
            gpu_transforms=transform,
        )
        return test_loader

    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
        optimizer.step()
        optimizer.zero_grad()
        # self.scheduler.step()

    def initialize_variables(self):
        self.step = 0
        self.best_score = 100
        if self.train_config.warm_start:
            with open(
                os.path.join(self.store_config.log_path, "best_score.yaml"), "r"
            ) as f:
                res = yaml.safe_load(f)
            if "best_score" in res.keys():
                self.best_score = res["best_score"]
            self.step = res["step"]


@hydra.main(config_path="yamls/nn.yaml")
def main(config):
    set_up(config)
    # Preparing for trainer
    monitor_metric = "avg_val_score"
    if config.data.n_fold != "all":
        save_top_k = 1
        val_check_interval = 1.0
    else:
        save_top_k = 3
        val_check_interval = 0.2
    check_val_every_n_epoch = 1
    # GCS upload function implemented
    checkpoint_callback = MyCallback(
        store_config=config.store,
        save_top_k=save_top_k,
        verbose=1,
        monitor=monitor_metric,
        mode="min",
    )
    hparams = {}
    for key, value in config.items():
        hparams.update(value)
    if config.store.wandb_project is not None:
        logger = WandbLogger(
            name=config.store.model_name + f"_fold{config.data.n_fold}",
            save_dir=config.store.log_path,
            project=config.store.wandb_project,
            version=hashlib.sha224(bytes(str(hparams), "utf8")).hexdigest()[:3],
            anonymous=True,
            config=hparams,
        )
    else:
        logger = None

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric, patience=10, verbose=False, mode="min"
    )

    backend = "dp" if len(config.base.gpu_id) > 1 else None
    if config.train.warm_start:
        checkpoint_path = sorted(
            glob(config.store.model_path + "/*"), key=lambda x: re.split("[=.]", x)[-2]
        )[-1]
    else:
        checkpoint_path = None

    model = Model(config)
    if config.data.is_train:
        trainer = Trainer(
            logger=logger,
            default_save_path=config.store.save_path,
            early_stop_callback=early_stop_callback,
            max_epochs=config.train.epoch,
            checkpoint_callback=checkpoint_callback,
            accumulate_grad_batches=config.train.accumulation_steps,
            use_amp=True if config.model.model_name != "gin" else False,
            amp_level="O1",
            gpus=len(config.base.gpu_id),
            distributed_backend=backend,
            show_progress_bar=True,
            train_percent_check=1.0,
            check_val_every_n_epoch=check_val_every_n_epoch,
            val_check_interval=val_check_interval,
            val_percent_check=1.0,
            test_percent_check=0.0,
            num_sanity_val_steps=4,
            nb_gpu_nodes=1,
            print_nan_grads=False,
            track_grad_norm=-1,
            gradient_clip_val=0.5,
            row_log_interval=10,
            log_save_interval=10,
            profiler=True,
            benchmark=True,
            resume_from_checkpoint=checkpoint_path,
            reload_dataloaders_every_epoch=True,
        )

        trainer.fit(model)
    else:
        trainer = Trainer(
            logger=logger,
            default_save_path=config.store.save_path,
            early_stop_callback=None,
            max_epochs=config.train.epoch,
            checkpoint_callback=checkpoint_callback,
            accumulate_grad_batches=config.train.accumulation_steps,
            use_amp=True if config.model.model_name != "gin" else False,
            amp_level="O1",
            gpus=len(config.base.gpu_id),
            distributed_backend=backend,
            show_progress_bar=True,
            train_percent_check=0.0,
            check_val_every_n_epoch=check_val_every_n_epoch,
            val_check_interval=val_check_interval,
            val_percent_check=0.0,
            test_percent_check=1.0,
            num_sanity_val_steps=4,
            nb_gpu_nodes=1,
            print_nan_grads=False,
            track_grad_norm=-1,
            gradient_clip_val=0.5,
            row_log_interval=10,
            log_save_interval=10,
            profiler=True,
            benchmark=True,
            resume_from_checkpoint=checkpoint_path,
            reload_dataloaders_every_epoch=True,
        )
        trainer.test(model, model.test_dataloader())


if __name__ == "__main__":
    main()
