# inference.py
import joblib
import os
import json
import numpy as np
import logging
import tarfile

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    logger.info("Loading model from %s", model_dir)
    model = joblib.load('/opt/ml/model/lr.joblib')
    return model

def input_fn(request_body, request_content_type):
    logger.info("Deserializing the input data.")
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        logging.debug("Input data: %s", input_data)
        return np.array(input_data)
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    logger.info("Making prediction.")
    prediction = model.predict(input_data)
    logging.debug("Prediction: %s", prediction)
    return prediction

def output_fn(prediction, response_content_type):
    logger.info("Serializing the output data.")
    return json.dumps(prediction.tolist())