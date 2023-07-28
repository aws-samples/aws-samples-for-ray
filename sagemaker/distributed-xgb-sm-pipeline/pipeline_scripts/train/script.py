import subprocess
import sys
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas==1.5.2', 'sagemaker','ray[all]==2.4.0', 'modin[ray]==0.18.0', 'xgboost_ray', 'pyarrow >= 6.0.1','pydantic==1.10.10', 'gpustat==1.0.0'])

import os
import time
from glob import glob
import argparse
import json
import logging
import boto3
import sagemaker
import numpy as np
import modin.pandas as pd

# Experiments
from sagemaker.session import Session
from sagemaker.experiments.run import load_run

import ray
from xgboost_ray import RayDMatrix, RayParams, train

from ray.air.config import ScalingConfig
from ray.data import Dataset
from ray.air.result import Result
from ray.air.checkpoint import Checkpoint
from sagemaker_ray_helper import RayHelper 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def read_parameters():
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.
    parser.add_argument('--max_depth', type=int, default=os.environ.get('SM_HP_MAX_DEPTH'))
    parser.add_argument('--eta', type=float, default=os.environ.get('SM_HP_ETA'))
    parser.add_argument('--min_child_weight', type=int, default=os.environ.get('SM_HP_MIN_CHILD_WEIGHT'))
    parser.add_argument('--subsample', type=float, default=os.environ.get('SM_HP_SUBSAMPLE'))
    parser.add_argument('--verbosity', type=int)
    parser.add_argument('--num_round', type=int)
    parser.add_argument('--tree_method', type=str, default="auto")
    parser.add_argument('--predictor', type=str, default="auto")

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--sm_hosts', type=str, default=os.environ.get('SM_HOSTS'))
    parser.add_argument('--sm_current_host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    
    parser.add_argument('--num_ray_workers', type=int,default=6)
    parser.add_argument('--use_gpu', type=bool, default=False)
    # parse region
    parser.add_argument('--region', type=str, default='us-east-1')
    
    parser.add_argument('--target_col', type=str, default='price')
    
    try:
        from sagemaker_training import environment
        env = environment.Environment()
        parser.add_argument('--n_jobs', type=int, default=env.num_cpus)
    except:
        parser.add_argument('--n_jobs', type=int, default=4)

    args, _ = parser.parse_known_args()
    return args

def load_dataset(path, num_workers, target_col="price"):
    """
    Loads the data as a ray dataset from the offline featurestore S3 location
    Args:
        feature_group_name (str): name of the feature group
        target_col (str): the target columns (will be used only for the test set).
    Returns:
        ds (ray.data.dataset): Ray dataset the contains the requested dat from the feature store
    """
    """
    cols_to_drop=[]
    # A simple check is this is test data
    # If True add the target column to the columns list to be dropped
    if '/test/' in path:
        cols_to_drop.append(target_col)
    """
    csv_files = glob(os.path.join(path, "*.csv"))
    print(f"found {len(csv_files)} files at {path}")
    ds = ray.data.read_csv(path)
    # ds = ds.drop_columns(cols_to_drop)
    print(f"{path} count is {ds.count()}")

    return ds.repartition(num_workers)

def train_xgboost(ds_train, ds_val, params, num_workers, target_col = "price") -> Result:
    """
    Creates a XGBoost trainer, train it, and return the result.        
    Args:
        ds_train (ray.data.dataset): Training dataset
        ds_val (ray.data.dataset): Validation dataset
        params (dict): Hyperparameters
        num_workers (int): number of workers to distribute the training across
        target_col (str): target column
    Returns:
        result (ray.air.result.Result): Result of the training job
    """
    
    train_set = RayDMatrix(ds_train, 'PRICE')
    val_set = RayDMatrix(ds_val, 'PRICE')
    
    evals_result = {}
    
    trainer = train(
        params=params,
        dtrain=train_set,
        evals_result=evals_result,
        evals=[(val_set, "validation")],
        verbose_eval=False,
        num_boost_round=100,
        ray_params=RayParams(num_actors=num_workers, cpus_per_actor=1),
    )
    
    output_path=os.path.join(args.model_dir, 'model.xgb')
    
    trainer.save_model(output_path)
    
    valMAE = evals_result["validation"]["mae"][-1]
    valRMSE = evals_result["validation"]["rmse"][-1]
 
    print('[3] #011validation-mae:{}'.format(valMAE))
    print('[4] #011validation-rmse:{}'.format(valRMSE))
    
    local_testing = False
    try:
        load_run(sagemaker_session=sess)
    except:
        local_testing = True
    if not local_testing: # Track experiment if using SageMaker Training
        with load_run(sagemaker_session=sess) as run:
            run.log_metric('validation-mae', valMAE)
            run.log_metric('validation-rmse', valRMSE)

def main():
    # Get SageMaker host information from runtime environment variables
    sm_hosts = json.loads(args.sm_hosts)
    sm_current_host = args.sm_current_host
    
    hyperparams = {
        'max_depth': args.max_depth,
        'min_child_weight': args.min_child_weight,
        'eta': args.eta,
        'subsample': args.subsample,
        "tree_method": "approx",
        "objective": "reg:squarederror",
        "eval_metric": ["mae", "rmse"],
        "num_round": 100,
        "seed": 47
    }

    ds_train = load_dataset(args.train, args.num_ray_workers, args.target_col)
    ds_validation = load_dataset(args.validation, args.num_ray_workers, args.target_col)
    
    trainer = train_xgboost(ds_train, ds_validation, hyperparams, args.num_ray_workers, args.target_col)

    
if __name__ == '__main__':
    ray_helper = RayHelper()
    
    ray_helper.start_ray()
    args = read_parameters()
    sess = sagemaker.Session(boto3.Session(region_name=args.region))

    start = time.time()
    main()
    taken = time.time() - start
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")
    
    
    
