import os
from src import logger
from src.entity.config_entity import DataTransformationConfig

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from joblib import dump


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        data = data.drop('Unnamed: 0',axis = 1)

        data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
        # Extract additional time features
        data['trans_hour'] = data['trans_date_trans_time'].dt.hour
        data['trans_day'] = data['trans_date_trans_time'].dt.dayofweek
        data['trans_month'] = data['trans_date_trans_time'].dt.month
        data['trans_year'] = data['trans_date_trans_time'].dt.year

        columns_to_drop = [
            'trans_date_trans_time', 
            'cc_num', 
            'first', 
            'last', 
            'gender', 
            'street', 
            'city', 
            'state', 
            'zip', 
            'trans_num', 
            'dob',
            'trans_year'
        ]
        data = data.drop(columns=columns_to_drop)

        train,test = train_test_split(data,test_size=self.config.test_size)

        features = train.columns.tolist()
        features.remove('is_fraud')

        num_features = train.select_dtypes(exclude="object").columns
        cat_features = train.select_dtypes(include="object").columns
        
        class CustomLabelEncoder(BaseEstimator, TransformerMixin):
            def __init__(self):
                self.label_encoders = {}
                
            def fit(self, X, y=None):
                for column in X.columns:
                    le = LabelEncoder()
                    le.fit(X[column])
                    self.label_encoders[column] = le
                return self
            
            def transform(self, X):
                X_transformed = X.copy()
                for column, le in self.label_encoders.items():
                    X_transformed[column] = le.transform(X[column])
                return X_transformed
            
        numeric_transformer = StandardScaler()
        oh_transformer = CustomLabelEncoder()

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', oh_transformer, cat_features),
                ('num', numeric_transformer, num_features)
            ],
            remainder='passthrough'
        )

        # Fit preprocessor on training data and transform both train and test data
        train_ = preprocessor.fit_transform(train.drop(self.config.target_column,axis=1))
        test_ = preprocessor.transform(test.drop(self.config.target_column,axis=1))

        # Convert transformed arrays back to DataFrame
        train = pd.DataFrame(train_, columns=features)
        test = pd.DataFrame(test_, columns=features)

        # Concatenate the target column to the DataFrame
        train[self.config.target_column] = data.loc[train.index, self.config.target_column]
        test[self.config.target_column] = data.loc[test.index, self.config.target_column]

        # Save preprocessed data to CSV files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        dump(preprocessor, os.path.join(self.config.root_dir, "preprocessor.joblib"))

        logger.info(f"Data split into training and test sets (test_size: {self.config.test_size})")
        logger.info(f"Training features shape: {train.shape}")
        logger.info(f"Test features shape: {test.shape}")

        print(f"Data split into training and test sets (test_size: {self.config.test_size})")
        print(f"Training features shape: {train.shape}")
        print(f"Test features shape: {test.shape}")