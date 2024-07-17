import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.common import read_yaml
from src.constants import PARAMS_FILE_PATH

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.preprocessor = joblib.load(Path('artifacts/data_transformation/preprocessor.joblib'))
        params = read_yaml(PARAMS_FILE_PATH)
        self.optimal_threshold = params['optimal_threshold']
    
    def predict(self, data):
        preprocessed_data = self.preprocessor.transform(data)
        prediction_proba = self.model.predict_proba(preprocessed_data)[:, 1]
        prediction = (prediction_proba >= self.optimal_threshold).astype(int)

        return prediction
