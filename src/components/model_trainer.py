import pandas as pd
import os
from src import logger
from src.entity.config_entity import ModelTrainerConfig
import xgboost as xgb
# from sklearn.preprocessing import OrdinalEncoder
import joblib

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        ## I need to make changes here. 
        xgb_clf = xgb.XGBClassifier(learning_rate=self.config.learning_rate,
                                    n_estimators=self.config.n_estimators,
                                    max_depth=self.config.max_depth)
        xgb_clf.fit(train_x, train_y)

        joblib.dump(xgb_clf, os.path.join(self.config.root_dir, self.config.model_name))