import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker','ray', 'pyarrow >= 6.0.1'])

import argparse
import os

import sagemaker
# Experiments
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup

import boto3
import ray
from ray.air.config import ScalingConfig
from ray.data import Dataset
from ray.data.preprocessors import StandardScaler

def read_parameters():
    """
    Read job parameters
    Returns:
        (Namespace): read parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_group_name', type=str, default='fs-synthetic-house-price')
    parser.add_argument('--train_size', type=float, default=0.6)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--target_col', type=str, default='PRICE')
    parser.add_argument('--region', type=str, default='us-east-1')
    params, _ = parser.parse_known_args()
    return params

def split_dataset(dataset, train_size, val_size, test_size, random_state=None):
    """
    Split dataset into train, validation and test samples
    Args:
        dataset (ray.data.Dataset): input data
        train_size (float): ratio of data to use as training dataset
        val_size (float): ratio of data to use as validation dataset
        test_size (float): ratio of data to use as test dataset
        random_state (int): Pass an int for reproducible output across multiple function calls.
    Returns:
        train_set (ray.data.Dataset): train dataset
        val_set (ray.data.Dataset): validation dataset
        test_set (ray.data.Dataset): test dataset
    """
    if (train_size + val_size + test_size) != 1.0:
        raise ValueError("train_size, val_size and test_size must sum up to 1.0")
    
    # Shuffle this dataset with a fixed random seed.
    shuffled_ds = dataset.random_shuffle(seed=random_state)
    # Split the data into train, validation and test datasets
    train_set, val_set, test_set = shuffled_ds.split_proportionately([train_size, val_size])
    
    # Sanity check
    # IMPORTANT!!! Do not include this for large datasets as this can be an expensive operation
    train_perc = int((train_set.count()/shuffled_ds.count()) * 100)
    print(f"Training size: {train_set.count()} - {train_perc}% of total")
    val_perc = int((val_set.count()/shuffled_ds.count()) * 100)
    print(f"Val size: {val_set.count()} - {val_perc}% of total")
    test_perc = int((test_set.count()/shuffled_ds.count()) * 100)
    print(f"Test size: {test_set.count()} - {test_perc}% of total")
    return train_set, val_set, test_set

def scale_dataset(train_set, val_set, test_set, target_col):
    """
    Fit StandardScaler to train_set and apply it to val_set and test_set
    Args:
        train_set (ray.data.Dataset): train dataset
        val_set (ray.data.Dataset): validation dataset
        test_set (ray.data.Dataset): test dataset
        target_col (str): target col
    Returns:
        train_transformed (ray.data.Dataset): train data scaled
        val_transformed (ray.data.Dataset): val data scaled
        test_transformed (ray.data.Dataset): test data scaled
    """
    
    tranform_cols = dataset.columns()
    # Remove the target columns from being scaled
    tranform_cols.remove(target_col)
    # set up a standard scaler
    standard_scaler = StandardScaler(tranform_cols)
    # fit scaler to training dataset
    print("Fitting scaling to training data and transforming dataset...")
    train_set_transformed = standard_scaler.fit_transform(train_set)
    # apply scaler to validation and test datasets
    print("Transforming validation and test datasets...")
    val_set_transformed = standard_scaler.transform(val_set)
    test_set_transformed = standard_scaler.transform(test_set)
    return train_set_transformed, val_set_transformed, test_set_transformed

def load_dataset(feature_group_name, region):
    """
    Loads the data as a ray dataset from the offline featurestore S3 location
    Args:
        feature_group_name (str): name of the feature group
    Returns:
        ds (ray.data.dataset): Ray dataset the contains the requested dat from the feature store
    """
    session = sagemaker.Session(boto3.Session(region_name=region))
    fs_group = FeatureGroup(
        name=feature_group_name, 
        sagemaker_session=session
    )

    fs_data_loc = fs_group.describe().get("OfflineStoreConfig").get("S3StorageConfig").get("ResolvedOutputS3Uri")
    
    # Drop columns added by the feature store
    # Since these are not related to the ML problem at hand
    cols_to_drop = ["record_id", "event_time","write_time", 
                    "api_invocation_time", "is_deleted", 
                    "year", "month", "day", "hour"]           

    ds = ray.data.read_parquet(fs_data_loc)
    ds = ds.drop_columns(cols_to_drop)
    print(f"{fs_data_loc} count is {ds.count()}")
    return ds

print(f"===========================================================")
print(f"Starting pre-processing")
print(f"Reading parameters")

# reading job parameters
args = read_parameters()
print(f"Parameters read: {args}")

# set output paths
train_data_path = "/opt/ml/processing/output/train"
val_data_path = "/opt/ml/processing/output/validation"
test_data_path = "/opt/ml/processing/output/test"

try:
    os.makedirs(train_data_path)
    os.makedirs(val_data_path)
    os.makedirs(test_data_path)
except:
    pass

# read data input
dataset = load_dataset(args.feature_group_name, args.region)

# split dataset into train, validation and test
train_set, val_set, test_set = split_dataset(
    dataset,
    train_size=args.train_size,
    val_size=args.val_size,
    test_size=args.test_size,
    random_state=args.random_state
)


# scale datasets
train_transformed, val_transformed, test_transformed = scale_dataset(
    train_set, 
    val_set, 
    test_set,
    args.target_col
)

print("Saving data")
train_transformed.write_csv(train_data_path)
val_transformed.write_csv(val_data_path)
test_transformed.write_csv(test_data_path)
print(f"Ending pre-processing")
print(f"===========================================================")
