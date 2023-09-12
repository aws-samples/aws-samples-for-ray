import argparse
import json
import os
from pathlib import Path
from typing import Union
import boto3
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_tabnet import tab_network
from ray_lightning import RayStrategy
from sagemaker_ray_helper import RayHelper
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy


class LitTabNet(pl.LightningModule):
    def __init__(
        self, num_features, lr=2e-1, lambda_sparse=1e-3, cat_emb_dim=1, cat_idxs=[], cat_dims=[],
    ):
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.lambda_sparse = lambda_sparse
        self.model = tab_network.TabNet(
            num_features,
            2,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            n_d=10,
            n_a=10,
            n_steps=5,
        )

        # self.example_input_array = torch.randn((1, num_features))

    def forward(self, x):
        output, M_loss = self.model(x)
        return output, M_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        output, M_loss = self(x)
        loss = F.cross_entropy(output, y)
        loss = loss - self.lambda_sparse * M_loss
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        output, M_loss = self(x)
        loss = F.cross_entropy(output, y)
        loss = loss - self.lambda_sparse * M_loss

        preds = torch.argmax(output, dim=1)
        acc = accuracy(preds, y, task="binary")

        if stage:
            self.log(
                f"{stage}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True
            )
            self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.9
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


class TabnetDataModule(pl.LightningDataModule):
    def __init__(self, train_pipe_shards, val_pipe_shards, batch_size: int = 32):
        super().__init__()
        self.prepare_data_per_node = True
        self.batch_size = batch_size
        self.train_pipe_shards = train_pipe_shards
        self.val_pipe_shards = val_pipe_shards

    def setup(self, stage=None):

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.assigned_train_shard = self.train_pipe_shards[self.rank]
        self.assigned_val_shard = self.val_pipe_shards[self.rank]

        self.train_epoch_iterator = self.assigned_train_shard.iter_epochs()
        self.val_epoch_iterator = self.assigned_val_shard.iter_epochs()

    def train_dataloader(self):

        train_epoch_pipe = next(self.train_epoch_iterator)

        return train_epoch_pipe.to_torch(
            label_column="label",
            label_column_dtype=torch.int64,
            feature_column_dtypes=torch.float32,
            batch_size=self.batch_size,
            unsqueeze_label_tensor=False,
            drop_last=True,
        )

    def val_dataloader(self):

        val_epoch_pipe = next(self.val_epoch_iterator)

        return val_epoch_pipe.to_torch(
            label_column="label",
            label_column_dtype=torch.int64,
            feature_column_dtypes=torch.float32,
            batch_size=self.batch_size * 4,
            unsqueeze_label_tensor=False,
            drop_last=True,
        )


def prep_data(df: pd.DataFrame, cat_encoders, drop_cols) -> pd.DataFrame:
    for col in cat_encoders:
        df[col] = df[col].astype(str).map(cat_encoders[col])
    df["label"] = 1 * (df["ArrDelay"] > 0)
    df = df.drop(drop_cols + ["ArrDelay"], axis=1)

    return df


def train(
    num_gpus,
    cpus_per_worker,
    col_meta_path,
    train_data_path,
    val_data_path,
    schema_data_path,
    tb_logging_path,
    epochs=5,
    batch_size=25_000,
    lr=2e-2
):

    col_meta_path = Path(col_meta_path)
    cat_encoders = json.load((col_meta_path / "encoders.json").open("r"))
    cat_embed_size = json.load((col_meta_path / "embed_size.json").open("r"))
    cat_num_unique = json.load((col_meta_path / "num_unique.json").open("r"))
    
    train_bucket, *train_key = train_data_path[5:].split("/")
    train_key = "/".join(train_key)
    val_bucket, *val_key = val_data_path[5:].split("/")
    val_key = "/".join(val_key)
    
    # the number of input files should be a multiple of the number of GPUs. Otherwise repartitioning is requried
    s3_client = boto3.client("s3")
    train_files = s3_client.list_objects(Bucket=train_bucket, Prefix=train_key)["Contents"]
    val_files = s3_client.list_objects(Bucket=val_bucket, Prefix=val_key)["Contents"]
    
    num_train_files = (len(train_files) // num_gpus) * num_gpus
    num_val_files = (len(val_files) // num_gpus) * num_gpus
    s3_train_paths = [os.path.join(f"s3://{train_bucket}", obj["Key"]) for obj in random.sample(train_files, num_train_files)]
    s3_val_paths = [os.path.join(f"s3://{val_bucket}", obj["Key"]) for obj in random.sample(val_files, num_val_files)]
    

    cat_cols = list(cat_encoders.keys())
    drop_cols = ["Year", "FlightNum"]

    print("creating data module")

    train_pipe = (
        ray.data.read_parquet(
            s3_train_paths
        )
        # .repartition(num_gpus)
        .window(blocks_per_window=num_gpus)
        .map_batches(
            prep_data, 
            fn_kwargs={"cat_encoders": cat_encoders, "drop_cols": drop_cols}, 
            batch_format="pandas"
        )
        .random_shuffle_each_window()
        .repeat()
    )

    print(f"s3_val_paths: {s3_val_paths}")
    print(f"num of gpus: {num_gpus}")
    val_pipe = (
        ray.data.read_parquet(
            s3_val_paths
        )
        # .repartition(num_gpus)
        .window(blocks_per_window=num_gpus)
        .map_batches(
            prep_data, 
            fn_kwargs={"cat_encoders": cat_encoders, "drop_cols": drop_cols}, 
            batch_format="pandas"
        )
        .repeat()
    )

    train_pipe_shards = train_pipe.split(n=num_gpus)
    val_pipe_shards = val_pipe.split(n=num_gpus)

    schema = (
        ray.data.read_parquet(
            schema_data_path
        )
        .window(blocks_per_window=1)
        .map_batches(
            prep_data, 
            fn_kwargs={"cat_encoders": cat_encoders, "drop_cols": drop_cols},
            batch_format="pandas"
        )
        .schema()
    )

    tab_dm = TabnetDataModule(train_pipe_shards, val_pipe_shards, batch_size)

    cat_idxs = [schema.names.index(col) for col in cat_cols]
    cat_dims = [ds for ds in cat_num_unique.values()]
    cat_emb_dim = [es for es in cat_embed_size.values()]
    num_features = len(schema.names)-1

    pmod = LitTabNet(cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, lr=lr, num_features=num_features)

    logger = TensorBoardLogger(
        tb_logging_path,
        name="pytorch_tabnet",
        log_graph=True,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        strategy=RayStrategy(
            num_workers=num_gpus, use_gpu=True, num_cpus_per_worker=cpus_per_worker
        ),
        enable_checkpointing=True,
        enable_progress_bar=True,
        reload_dataloaders_every_n_epochs=1,
        logger=logger,
        num_sanity_val_steps=0,
    )

    trainer.fit(pmod, tab_dm)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--batch_size", type=int, default=25_000)
    parser.add_argument("--s3_train_data", type=str)
    parser.add_argument("--s3_test_data", type=str)
    parser.add_argument("--s3_schema_file", type=str)
    parser.add_argument("--tb_logging_path", type=str)
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--meta_data_path", type=str, default=os.environ.get("SM_CHANNEL_META")
    )

    args, _ = parser.parse_known_args()

    ray_helper = RayHelper()
    ray_helper.start_ray()
    cluster_resources = ray.cluster_resources()
    num_gpus = int(cluster_resources["GPU"])
    num_cpus = int(cluster_resources["CPU"])

    cpus_per_worker = int((num_cpus - num_gpus) // num_gpus)
    train(num_gpus=num_gpus, 
          cpus_per_worker=cpus_per_worker, 
          col_meta_path=args.meta_data_path,     
          train_data_path=args.s3_train_data,
          val_data_path=args.s3_test_data,
          schema_data_path=args.s3_schema_file,
          tb_logging_path=args.tb_logging_path,
          epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr)
