import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker','ray', 'xgboost_ray', 'pyarrow >= 6.0.1'])
import os
import time
import ray.cloudpickle as cloudpickle
import argparse
import json
import logging
import boto3
import sagemaker
# Experiments
from sagemaker.session import Session
from sagemaker.experiments.run import load_run

import ray
from ray.train.xgboost import XGBoostTrainer
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
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--min_child_weight', type=int)
    parser.add_argument('--subsample', type=float)
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
    
    parser.add_argument('--num_ray_workers', type=int,default=3)
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

def load_dataset(fs_data_loc, target_col="price"):
    """
    Loads the data as a ray dataset from the offline featurestore S3 location
    Args:
        feature_group_name (str): name of the feature group
        target_col (str): the target columns (will be used only for the test set).
    Returns:
        ds (ray.data.dataset): Ray dataset the contains the requested dat from the feature store
    """
    # Drop columns added by the feature store
    cols_to_drop = ["record_id", "event_time","write_time", 
                    "api_invocation_time", "is_deleted", 
                    "year", "month", "day", "hour"]
                    
    
    # A simple check is this is test data
    # If True add the target column to the columns list to be dropped
    if '/test/' in fs_data_loc:
        cols_to_drop.append(target_col)

    ds = ray.data.read_parquet(fs_data_loc)
    ds = ds.drop_columns(cols_to_drop)
    print(f"{fs_data_loc} count is {ds.count()}")

    return ds

def train_xgboost(ds_train, ds_val, params, num_workers, use_gpu = False, target_col = "price") -> Result:
    """
    Creates a XGBoost trainer, train it, and return the result.        
    Args:
        ds_train (ray.data.dataset): Training dataset
        ds_val (ray.data.dataset): Validation dataset
        params (dict): Hyperparameters
        num_workers (int): number of workers to distribute the training across
        use_gpu (bool): Should the taining job use GPUs
        target_col (str): target column
    Returns:
        result (ray.air.result.Result): Result of the training job
    """
    trainer = XGBoostTrainer(
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        label_column="PRICE",
        params=params,
        datasets={"train": ds_train, "valid": ds_val},
        num_boost_round=100,
    )
    result = trainer.fit()
    print("<==== Start Training Metrics ====>")
    print(result.metrics)
    print("<==== END Training Metrics ====>")

    return result

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

    ds_train = load_dataset(args.train, args.target_col)
    ds_validation = load_dataset(args.validation, args.target_col)
    
    result = train_xgboost(ds_train, ds_validation, hyperparams, args.num_ray_workers, args.use_gpu, args.target_col)
    metrics = result.metrics
    # checkpoint = result.checkpoint.to_directory(path=os.path.join(args.model_dir, f'model.xgb'))
    
    output_path=os.path.join(args.model_dir, f'model.pkl')
    # Serialize the trained model using ray.cloudpickle
    serialized_model = cloudpickle.dumps(result)

    # Save the serialized model to a file
    with open(output_path, 'wb') as f:
        f.write(serialized_model)
    
    trainMAE = metrics['train-mae']
    trainRMSE = metrics['train-rmse']
    valMAE = metrics['valid-mae']
    valRMSE = metrics['valid-rmse']
    print('[1] #011train-mae:{}'.format(trainMAE))
    print('[2] #011train-rmse:{}'.format(trainRMSE))
    print('[3] #011validation-mae:{}'.format(valMAE))
    print('[4] #011validation-rmse:{}'.format(valRMSE))
    
    local_testing = False
    try:
        load_run(sagemaker_session=sess)
    except:
        local_testing = True
    if not local_testing: # Track experiment if using SageMaker Training
        with load_run(sagemaker_session=sess) as run:
            run.log_metric('train-mae', trainMAE)
            run.log_metric('train-rmse', trainRMSE)
            run.log_metric('validation-mae', valMAE)
            run.log_metric('validation-rmse', valRMSE)
    
if __name__ == '__main__':
    ray_helper = RayHelper()
    
    ray_helper.start_ray()
    args = read_parameters()
    sess = sagemaker.Session(boto3.Session(region_name=args.region))

    start = time.time()
    main()
    taken = time.time() - start
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")
    
    
    
