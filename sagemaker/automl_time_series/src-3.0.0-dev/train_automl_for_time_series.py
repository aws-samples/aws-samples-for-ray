import os
import logging
import sys
import time
import itertools
import pandas as pd
import numpy as np
from collections import defaultdict
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA, _TS
from pyarrow import parquet as pq
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys
import ray
from ray import air, tune
from ray.air import Checkpoint, session
from sagemaker_ray_helper import RayHelper



FILENAME = os.path.join(os.environ.get("SM_CHANNEL_TRAIN"), "target.parquet")
MODEL_DIR = os.environ["SM_MODEL_DIR"]
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# %% ../nbs/models.ipynb 8
class _TS:
    def new(self):
        b = type(self).__new__(type(self))
        b.__dict__.update(self.__dict__)
        return b
    

def cross_validation(config, Y_train_df=None):
    assert Y_train_df is not None, "Must pass in the dataset"

    # Get the model class
    model_cls, model_params = config.get("model_cls_and_params")
    freq = config.get("freq")
    metrics = config.get("metrics", {"mse": mean_squared_error})

    # CV params
    test_size = config.get("test_size", None)
    n_splits = config.get("n_splits", 5)

    model = model_cls(**model_params)

    # Default the parallelism to the # of cross-validation splits
    parallelism_kwargs = {"n_jobs": n_splits}

    # Initialize statsforecast with the model
    statsforecast = StatsForecast(
        df=Y_train_df,
        models=[model],
        freq=freq,
        **parallelism_kwargs,
    )

    # Perform temporal cross-validation (see `sklearn.TimeSeriesSplit`)
    test_size = test_size or len(Y_train_df) // (n_splits + 1)

    start_time = time.time()
    forecasts_cv = statsforecast.cross_validation(
        h=test_size,
        n_windows=n_splits,
        step_size=test_size,
    )
    cv_time = time.time() - start_time

    # Compute metrics (according to `metrics`)
    cv_results = compute_metrics_and_aggregate(forecasts_cv, model, metrics)

    # Report metrics and save cross-validation output DataFrame
    results = {
        **cv_results,
        "cv_time": cv_time,
    }
    checkpoint_dict = {
        "cross_validation_df": forecasts_cv,
    }
    checkpoint = Checkpoint.from_dict(checkpoint_dict)
    session.report(results, checkpoint=checkpoint)
    
    
    
def compute_metrics_and_aggregate(
    forecasts_df: pd.DataFrame, model: _TS, metrics: dict
):
    unique_ids = forecasts_df.index.unique()
    assert len(unique_ids) == 1, "This example only expects a single time series."

    cutoff_values = forecasts_df["cutoff"].unique()

    # Calculate metrics of the predictions of the models fit on
    # each training split
    cv_metrics = defaultdict(list)
    for ct in cutoff_values:
        # Get CV metrics for a specific training split
        # All forecasts made with the same `cutoff` date
        split_df = forecasts_df[forecasts_df["cutoff"] == ct]
        for metric_name, metric_fn in metrics.items():
            cv_metrics[metric_name].append(
                metric_fn(
                    split_df["y"], split_df[model.__class__.__name__]
                )
            )

    # Calculate aggregated metrics (mean, std) across training splits
    cv_aggregates = {}
    for metric_name, metric_vals in cv_metrics.items():
        try:
            cv_aggregates[f"{metric_name}_mean"] = np.nanmean(
                metric_vals
            )
            cv_aggregates[f"{metric_name}_std"] = np.nanstd(
                metric_vals
            )
        except Exception as e:
            cv_aggregates[f"{metric_name}_mean"] = np.nan
            cv_aggregates[f"{metric_name}_std"] = np.nan

    return {
        "unique_ids": list(unique_ids),
        **cv_aggregates,
        "cutoff_values": cutoff_values,
    }


def generate_configurations(search_space):
    for model, params in search_space.items():
        if not params:
            yield model, {}
        else:
            configurations = itertools.product(*params.values())
            for config in configurations:
                config_dict = {k: v for k, v in zip(params.keys(), config)}
                yield model, config_dict


def get_m5_partition(unique_id: str) -> pd.DataFrame:
    ds1 = pq.read_table(
        FILENAME,
        filters=[("item_id", "=", unique_id)],
    )
    Y_df = ds1.to_pandas()
    # StatsForecasts expects specific column names!
    Y_df = Y_df.rename(
        columns={"item_id": "unique_id", "timestamp": "ds", "demand": "y"}
    )
    Y_df["unique_id"] = Y_df["unique_id"].astype(str)
    Y_df["ds"] = pd.to_datetime(Y_df["ds"])
    Y_df = Y_df.dropna()
    constant = 10
    Y_df["y"] += constant
    return Y_df[Y_df.unique_id == unique_id]

    
if __name__ == "__main__":
    ray_helper = RayHelper()
    ray_helper.start_ray()
    # if ray.is_initialized():
    #     ray.shutdown()
    # ray.init(runtime_env={"pip": ["statsforecast"]})
    start = time.time()
    df = get_m5_partition("FOODS_1_001_CA_1")
    search_space = {
        AutoARIMA: {},
        AutoETS: {
            "season_length": [6, 7],
            "model": ["ZNA", "ZZZ"],
        }
    }
    
    configs = list(generate_configurations(search_space))
    tuner = tune.Tuner(
        tune.with_parameters(cross_validation, Y_train_df=df),
        param_space={
            "model_cls_and_params": tune.grid_search(configs),
            "n_splits": 5,
            "test_size": 1,
            "freq": "D",
            "metrics":  {"mse": mean_squared_error, "mae": mean_absolute_error},
        },
        tune_config=tune.TuneConfig(
            metric="mse_mean",
            mode="min",
        ),
    )

    result_grid = tuner.fit()
    
    best_result = result_grid.get_best_result()
    best_result.config
    
    best_model_cls, best_model_params = best_result.config["model_cls_and_params"]
    print("Best model type:", best_model_cls)
    print("Best model params:", best_model_params)
    
    print("Best mse_mean:", best_result.metrics["mse_mean"])
    print("Best mae_mean:", best_result.metrics["mae_mean"])
    
    
    taken = time.time() - start
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")
