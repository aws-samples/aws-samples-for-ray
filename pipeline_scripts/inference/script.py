
import json
import os

import numpy as np
import tarfile
import xgboost as xgb
import sagemaker_xgboost_container.encoder as xgb_encoder


def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, 'model.xgb'))
    return booster


def input_fn(request_body, request_content_type):
    """
    The SageMaker XGBoost model server receives the request data body and the content type,
    and invokes the `input_fn`.

    Return a DMatrix (an object that can be passed to predict_fn).
    """
    print(f'Incoming format type is {request_content_type}')
    if request_content_type == "text/csv":
        decoded_payload = request_body.strip()
        return xgb_encoder.csv_to_dmatrix(decoded_payload, dtype=np.float)
    if request_content_type == "text/libsvm":
        return xgb_encoder.libsvm_to_dmatrix(request_body)
    else:
        raise ValueError(
            "Content type {} is not supported.".format(request_content_type)
        )


def predict_fn(input_data, model):
    """
    SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.

    Return a two-dimensional NumPy array where the first columns are predictions
    and the remaining columns are the feature contributions (SHAP values) for that prediction.
    """
    prediction = model.predict(input_data)
    feature_contribs = model.predict(input_data, pred_contribs=True, validate_features=False)
    output = np.hstack((prediction[:, np.newaxis], feature_contribs))
    return output


def output_fn(predictions, content_type):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    print(f'outgoing format type is {content_type}')
    print (predictions)
    if content_type == "text/csv":
        return ','.join(str(x[0]) for x in predictions)
    else:
        raise ValueError("Content type {} is not supported.".format(content_type))
