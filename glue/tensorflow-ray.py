%region us-east-1
%glue_ray
#%endpoint https://glue-gamma.us-east-2.amazonaws.com/
%worker_type Z.2X
%session_id_prefix cfregly-test
%additional_python_modules ray[ml],tensorflow-io,tensorflow


import tensorflow as tf

import ray
from ray.air import session
from ray.air.callbacks.keras import Callback
from ray.train.tensorflow import prepare_dataset_shard
from ray.train.tensorflow import TensorflowTrainer
from ray.air.config import ScalingConfig
from ray.air.config import RunConfig

print(tf.__version__)
print(ray.__version__)

train_dataset = ray.data.from_items([{"x": x, "y": 2 * x + 1} for x in range(200)])
print(train_dataset)


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=()),
            # Add feature dimension, expanding (batch_size,) to (batch_size, 1).
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Dense(1),
        ]
    )
    return model


def train_func(config: dict):
    batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 1)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = build_model()
        multi_worker_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=config.get("lr", 1e-3)),
            loss=tf.keras.losses.mean_squared_error,
            metrics=[tf.keras.metrics.mean_squared_error],
        )

    dataset = session.get_dataset_shard("train")

    def to_tf_dataset(dataset, batch_size):
        def to_tensor_iterator():
            for batch in dataset.iter_tf_batches(
                batch_size=batch_size, dtypes=tf.float32
            ):
                yield batch["x"], batch["y"]

        output_signature = (
            tf.TensorSpec(shape=(None), dtype=tf.float32),
            tf.TensorSpec(shape=(None), dtype=tf.float32),
        )
        tf_dataset = tf.data.Dataset.from_generator(
            to_tensor_iterator, output_signature=output_signature
        )
        return prepare_dataset_shard(tf_dataset)

    results = []
    for _ in range(epochs):
        tf_dataset = to_tf_dataset(dataset=dataset, batch_size=batch_size)
        history = multi_worker_model.fit(tf_dataset, callbacks=[Callback()])
        results.append(history.history)
    return results


num_workers = 2
use_gpu = False

config = {"lr": 1e-3, "batch_size": 32, "epochs": 1}

print(config)
trainer = TensorflowTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    run_config=RunConfig(local_dir="/tmp/ray_results"),
    datasets={"train": train_dataset},
)
result = trainer.fit()
print(result.metrics)