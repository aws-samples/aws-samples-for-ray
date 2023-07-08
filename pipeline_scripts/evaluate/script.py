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
from sagemaker.experiments.run import load_run
import modin.pandas as pd
# Ray specific imports
import ray
from ray.air.checkpoint import Checkpoint
from ray.train.xgboost import XGBoostCheckpoint, XGBoostPredictor


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

if __name__ == "__main__":
    logger.debug('Starting evaluation.')
    
    model_dir = '/opt/ml/processing/model'
    for file in os.listdir(model_dir):
        logger.info(file)
        
    model_path = os.path.join(model_dir, 'model.tar.gz')
    # Open the .tar.gz file
    with tarfile.open(model_path, 'r:gz') as tar:
        # Extract all files to the model directory
        tar.extractall(path=model_dir)

    for file in os.listdir(model_dir):
        logger.debug(file)
        
    logger.debug('Loading model.')
    checkpoint = XGBoostCheckpoint.from_directory(f'{model_dir}/model.xgb')
    predictor = XGBoostPredictor.from_checkpoint(checkpoint)

    logger.debug('Reading test data.')
    test_path = "/opt/ml/processing/test/"
    all_files = glob.glob(os.path.join(test_path , "*.csv"))
    frames = []
    for filename in all_files:
        frame = pd.read_csv(filename, index_col=None, header=0)
        frames.append(frame)
    df = pd.concat(frames, axis=0, ignore_index=True)
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = df.to_numpy()

    
    logger.info('Performing predictions against test data.')
    predictions = predictor.predict(X_test)

    # See the regression metrics
    # see: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    logger.debug('Calculating metrics.')
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, predictions)
    std = np.std(y_test - predictions)
    report_dict = {
        'regression_metrics': {
            'mae': {
                'value': mae,
                'standard_deviation': std,
            },
            'mse': {
                'value': mse,
                'standard_deviation': std,
            },
            'rmse': {
                'value': rmse,
                'standard_deviation': std,
            },
            'r2': {
                'value': r2,
                'standard_deviation': std,
            },
        },
    }

    output_dir = '/opt/ml/processing/evaluation'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info('Writing out evaluation report with mse: %f', mse)
    evaluation_path = f'{output_dir}/evaluation.json'
    with open(evaluation_path, 'w') as f:
        f.write(json.dumps(report_dict))
