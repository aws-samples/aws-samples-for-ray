import argparse
import json
import os
from functools import partial
from itertools import cycle
from pathlib import Path

import pandas as pd
import ray
import tabnet
import tensorflow as tf
from ray import train
from ray.air import session
from ray.air.integrations.keras import ReportCheckpointCallback as KerasCallback
from ray.air.config import RunConfig, ScalingConfig
from ray.train.tensorflow import (TensorflowCheckpoint, TensorflowTrainer,
                                  prepare_dataset_shard)
from sagemaker_ray_helper import RayHelper


def prep_data(df: pd.DataFrame, cat_encoders) -> pd.DataFrame:
    for col in cat_encoders:
        df[col] = df[col].astype(str).map(cat_encoders[col])
    df["label"] = 1 * (df["ArrDelay"] > 0)
    df = df.drop(drop_cols + ["ArrDelay"], axis=1)

    return df


class TabNet(tf.keras.Model):
    def __init__(self, feature_columns, projection_dim=64, **kwargs):
        super(TabNet, self).__init__(**kwargs)

        self.embed_layer = tf.keras.layers.DenseFeatures(feature_columns)
        self.projection = tf.keras.layers.Dense(
            projection_dim, activation="linear", use_bias=False
        )

        self.tabnet_model = tabnet.TabNetClassifier(
            None,
            num_features=projection_dim,
            num_classes=2,
            feature_dim=12,
            output_dim=8,
            num_decision_steps=3,
            relaxation_factor=0.5,
            sparsity_coefficient=1e-3,
            batch_momentum=0.02,
            virtual_batch_size=1000,
        )

    def call(self, inputs, training=None):
        embed = self.embed_layer(inputs)
        proj = self.projection(embed)
        out = self.tabnet_model(proj, training=training)
        return out


def train_loop_per_worker(config):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["num_epochs"]
    schema = config["schema"]
    cat_encoders = config["cat_encoders"]
    cat_embed_size = config["cat_embed_size"]
    model_dir = config["model_dir"]

    # Get the Ray Dataset shard for this data parallel worker,
    # and convert it to a Tensorflow Dataset.

    train_data = session.get_dataset_shard("train")

    feature_columns = []
    for col_name in schema.names:
        if col_name == "label":
            pass
        elif col_name in cat_cols:
            feat_col = tf.feature_column.categorical_column_with_vocabulary_list(
                col_name, list(cat_encoders[col_name].values())
            )
            if cat_embed_size[col_name] > 4:
                feat_col = tf.feature_column.embedding_column(
                    feat_col, cat_embed_size[col_name]
                )
            else:
                feat_col = tf.feature_column.indicator_column(feat_col)
            feature_columns.append(feat_col)
        else:
            feature_columns.append(tf.feature_column.numeric_column(col_name))

    output_types = dict(zip(schema.names, cycle([tf.float32])))
    for col in output_types:
        if col in cat_cols:
            output_types[col] = tf.int64

    output_types["label"] = tf.int64
    output_shapes = dict(zip(schema.names, cycle((batch_size,))))

    def create_iterator(ds, batch_size):
        for batch in ds.iter_tf_batches(batch_size=batch_size, drop_last=True):
            yield batch

    def to_tf_dataset(dataset, batch_size, output_types, output_shape):

        tfds = tf.data.Dataset.from_generator(
            partial(create_iterator, dataset, batch_size),
            output_types=output_types,
            output_shapes=output_shapes,
        )
        return prepare_dataset_shard(tfds)

    def split_target(record):
        target = record.pop("label")
        return record, target

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.

        model = TabNet(feature_columns)
        lr = tf.keras.optimizers.schedules.InverseTimeDecay(lr, 20, 0.75)
        optimizer = tf.keras.optimizers.Adam(lr)
        model.compile(
            optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    results = []

    for _ in range(epochs):
        tf_train_dataset = to_tf_dataset(
            dataset=train_data,
            batch_size=batch_size,
            output_types=output_types,
            output_shape=output_shapes,
        )

        tf_train_dataset = tf_train_dataset.map(split_target).shuffle(batch_size)
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="/opt/ml/checkpoints", 
                                                      write_images=True,
                                                      histogram_freq=1,
                                                      update_freq="epoch")

        history = model.fit(
            tf_train_dataset,
            callbacks=[KerasCallback(), tensorboard_callback],
            verbose=0,
        )
        results.append(history)

#     model.save(model_dir)

    #     test_loss, test_acc = model.evaluate(tf_test_dataset)
    #     session.report({"val_acc": test_acc})

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--per_device_batch_size", type=int, default=1024)
    parser.add_argument("--s3_train_data", type=str)
    parser.add_argument("--s3_test_data", type=str)
    parser.add_argument("--s3_schema_file", type=str)
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--meta_data_path", type=str, default=os.environ.get("SM_CHANNEL_META"))

    args, _ = parser.parse_known_args()

    ray_helper = RayHelper()
    ray_helper.start_ray()

    cluster_resources = ray.cluster_resources()
    num_gpus = int(cluster_resources["GPU"])
    num_cpus = int(cluster_resources["CPU"])

    col_meta_path = Path(args.meta_data_path)
    cat_encoders = json.load((col_meta_path / "encoders.json").open("r"))
    cat_embed_size = json.load((col_meta_path / "embed_size.json").open("r"))
    cat_num_unique = json.load((col_meta_path / "num_unique.json").open("r"))

    cat_cols = list(cat_encoders.keys())
    drop_cols = [
        "Year",
        "FlightNum",
    ]

    schema = (
        ray.data.read_parquet(args.s3_schema_file)
        .map_batches(prep_data, fn_kwargs={"cat_encoders": cat_encoders}, batch_format="pandas")
        .schema()
    )

    train_ds = (
        ray.data.read_parquet(args.s3_train_data)
        .randomize_block_order()
        .map_batches(prep_data, fn_kwargs={"cat_encoders": cat_encoders}, batch_format="pandas")
    )

    #     test_ds = (ray.data
    #                   .read_parquet("s3://sagemaker-us-east-1-152804913371/pt_lightning_tabnet_test/airlines/test")
    #                   .map_batches(prep_data, fn_kwargs={"cat_encoders":cat_encoders})
    #                  )

    batch_size = args.per_device_batch_size * num_gpus

    trainer = TensorflowTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "batch_size": batch_size,
            "num_epochs": args.epochs,
            "lr": args.lr,
            "schema": schema,
            "cat_encoders": cat_encoders,
            "cat_embed_size": cat_embed_size,
            "model_dir": args.model_dir,
        },
        scaling_config=ScalingConfig(
            num_workers=num_gpus,  # Number of data parallel training workers
            use_gpu=True,
            trainer_resources={"CPU": num_cpus - num_gpus},
        ),
        run_config=RunConfig(local_dir="/opt/ml/checkpoints"),
        datasets={"train": train_ds},
    )

    result = trainer.fit()

    print(result.metrics)
