import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker', 'ray', 'modin[ray]', 'pydantic==1.10.10', 'xgboost_ray'])
import os
import time
import tarfile
import argparse
import json
import logging
import boto3
import sagemaker
import glob

import pathlib
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sagemaker specific imports
from sagemaker.session import Session
import pandas as pd
import xgboost as xgb

# Ray specific imports
# import ray
# from ray.air.checkpoint import Checkpoint
# from ray.train.xgboost import XGBoostCheckpoint, XGBoostPredictor
# import ray.cloudpickle as cloudpickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

if __name__ == "__main__":
    logger.debug('Starting evaluation.')
    
    model_dir = '/opt/ml/processing/model'
    for file in os.listdir(model_dir):
        logger.info(file)
        
    model_path = os.path.join(model_dir, 'model.tar.gz')
    with tarfile.open(model_path) as tar:
        tar.extractall(path=model_dir)
    
    for file in os.listdir(model_dir):
        logger.info(file)
        
    logger.debug('Loading sklearn model.')
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, 'model.xgb'))

    logger.debug('Reading test data.')

    test_path = "/opt/ml/processing/test/"
    # Get list of all csv files in folder
    csv_files = glob.glob(f'{test_path}*.csv')
    # Create empty dataframe
    df = pd.DataFrame()
    
    # Loop through csv files and read into df 
    for f in csv_files:
        filename = os.path.basename(f)
        tmp_df = pd.read_csv(f, header=None)
        df = pd.concat([df, tmp_df], ignore_index=True)

    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = df.to_numpy()
    
    logger.info('Performing predictions against test data.')
    predictions = model.predict(X_test)

    # See the regression metrics
    # see: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    logger.debug('Calculating metrics.')
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = sqrt(mse)
    report_dict = {
        'regression_metrics': {
            'mae': {
                'value': mae,
            },
            'rmse': {
                'value': rmse,
            },
        },
    }

    output_dir = '/opt/ml/processing/evaluation'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info('Writing out evaluation report with rmse: %f', rmse)
    evaluation_path = f'{output_dir}/evaluation.json'
    with open(evaluation_path, 'w') as f:
        f.write(json.dumps(report_dict))
