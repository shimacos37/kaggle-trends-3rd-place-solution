import os
from glob import glob
import csv
from typing import Optional, List, Dict
import wandb
from wandb.wandb_run import Run
from google.cloud import storage
from pytorch_lightning.logging import TensorBoardLogger, LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter


class MyLogger(TensorBoardLogger):
    __test__ = False

    def __init__(self, save_dir, debug=False, version=None, create_git_tag=False):
        super().__init__(save_dir, name=None, version=version)
        self.save_dir = save_dir
        self._name = None
        self.debug = debug
        self._version = version
        self.create_git_tag = create_git_tag

    @property
    def root_dir(self):
        """
        Parent directory for all tensorboard checkpoint subdirectories.
        If the experiment name parameter is None or the empty string, no experiment subdirectory is used
        and checkpoint will be saved in save_dir/version_dir
        """
        return self.save_dir

    @property
    def log_dir(self):
        """
        The directory for this run's tensorboard checkpoint.  By default, it is named 'version_${self.version}'
        but it can be overridden by passing a string value for the constructor's version parameter
        instead of None or an int
        """
        return self.root_dir

    @property
    def experiment(self):
        r"""
         Actual tensorboard object. To use tensorboard features do the following.
         Example::
             self.logger.experiment.some_tensorboard_function()
         """
        if self._experiment is not None:
            return self._experiment

        root_dir = self.save_dir
        os.makedirs(root_dir, exist_ok=True)
        log_dir = root_dir
        self._experiment = SummaryWriter(log_dir=log_dir, **self.kwargs)
        return self._experiment

    @property
    def version(self):
        return self._version

    @rank_zero_only
    def save(self):
        try:
            self.experiment.flush()
        except AttributeError:
            # you are using PT version (<v1.2) which does not have implemented flush
            self.experiment._get_file_writer().flush()

        # create a preudo standard path ala test-tube
        dir_path = self.save_dir

        # prepare the file path
        meta_tags_path = os.path.join(dir_path, self.NAME_CSV_TAGS)

        # save the metatags file
        with open(meta_tags_path, "w", newline="") as csvfile:
            fieldnames = ["key", "value"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({"key": "key", "value": "value"})
            for k, v in self.tags.items():
                writer.writerow({"key": k, "value": v})


class MyCallback(ModelCheckpoint):
    def __init__(
        self,
        store_config,
        monitor="val_loss",
        verbose=0,
        save_top_k=1,
        save_weights_only=False,
        mode="auto",
        period=1,
    ):
        super(MyCallback, self).__init__(
            store_config.model_path,
            monitor,
            verbose,
            save_top_k,
            save_weights_only,
            mode,
            period,
            store_config.model_name,
        )
        self.store_config = store_config

    def _save_model(self, filepath):
        dirpath = os.path.dirname(filepath)
        # make paths
        os.makedirs(dirpath, exist_ok=True)

        # delegate the saving to the model
        self.save_function(filepath)
        if self.store_config.gcs_project is not None:
            self.upload_directory()

    def upload_directory(self):
        storage_client = storage.Client(self.store_config.gcs_project)
        bucket = storage_client.get_bucket(self.store_config.bucket_name)
        filenames = glob(
            os.path.join(self.store_config.save_path, "**"), recursive=True
        )
        for filename in filenames:
            if os.path.isdir(filename):
                continue
            destination_blob_name = os.path.join(
                self.store_config.gcs_path,
                filename.split(self.store_config.save_path)[-1][1:],
            )
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(filename)


class WandbLogger(LightningLoggerBase):
    """
    Logger for `W&B <https://www.wandb.com/>`_.
    Args:
        name (str): display name for the run.
        save_dir (str): path where data is saved.
        offline (bool): run offline (data can be streamed later to wandb servers).
        id or version (str): sets the version, mainly used to resume a previous run.
        anonymous (bool): enables or explicitly disables anonymous logging.
        project (str): the name of the project to which this run will belong.
        tags (list of str): tags associated with this run.
    Example
    --------
    .. code-block:: python
        from pytorch_lightning.loggers import WandbLogger
        from pytorch_lightning import Trainer
        wandb_logger = WandbLogger()
        trainer = Trainer(logger=wandb_logger)
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: bool = False,
        id: Optional[str] = None,
        anonymous: bool = False,
        version: Optional[str] = None,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        experiment=None,
        entity=None,
        config=None,
    ):
        super().__init__()
        self._name = name
        self._save_dir = save_dir
        self._anonymous = "allow" if anonymous else None
        self._id = version or id
        self._tags = tags
        self._project = project
        self._experiment = experiment
        self._offline = offline
        self._entity = entity
        self._config = config

    def __getstate__(self):
        state = self.__dict__.copy()
        # args needed to reload correct experiment
        if self._experiment is not None:
            state["_id"] = self._experiment.id
        else:
            state["_id"] = None

        # cannot be pickled
        state["_experiment"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        self.set_experiment()

    @rank_zero_only
    def set_experiment(self):
        self._experiment = wandb.init(
            name=self._name,
            dir=self._save_dir,
            project=self._project,
            anonymous=self._anonymous,
            id=self._id,
            resume="allow",
            tags=self._tags,
            entity=self._entity,
            config=self._config,
        )

    @property
    def experiment(self) -> Run:
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"
            self._experiment = wandb.init(
                name=self._name,
                dir=self._save_dir,
                project=self._project,
                anonymous=self._anonymous,
                reinit=True,
                id=self._id,
                resume="allow",
                tags=self._tags,
                entity=self._entity,
                config=self._config,
            )
        return self._experiment

    def watch(self, model, log="gradients", log_freq=100):
        wandb.watch(model, log, log_freq)

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        metrics["global_step"] = step
        self.experiment.log(metrics, step=step)

    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status: str = "success"):
        try:
            exit_code = 0 if status == "success" else 1
            wandb.join(exit_code)
        except TypeError:
            wandb.join()

    @property
    def name(self) -> str:
        if self._experiment:
            return self._experiment.project_name()
        else:
            return self._name

    @property
    def version(self) -> str:
        if self._experiment:
            return self._experiment.id
        return None
