import os
from src import logger
from src.entity.config_entity import DataTransformationConfig

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from joblib import dump

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


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        data = data.drop(data.columns[0], axis=1)

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
        logger.info("Extracted time features and dropped unnecessary columns")

        train, test = train_test_split(data, test_size=self.config.test_size, stratify=data[self.config.target_column])
        logger.info(f"Data split into training and test sets (test_size: {self.config.test_size})")

        # Log initial column names in train
        logger.info(f"Initial columns in train: {train.columns.tolist()}")

        features = train.columns.tolist()
        if 'is_fraud' in features:
            features.remove('is_fraud')
            logger.info("Removed 'is_fraud' from features list")
        else:
            logger.warning("'is_fraud' column not found in features list")

        # Separate numeric and categorical features
        num_features = list(train.select_dtypes(include=[np.number]).columns)
        cat_features = list(train.select_dtypes(include=['object']).columns)
        if 'is_fraud' in num_features:
            num_features.remove('is_fraud')

        logger.info("Numeric features: {}".format(num_features))
        logger.info("Categorical features: {}".format(cat_features))

        # Ensure the target column is correctly handled
        if 'is_fraud' not in train.columns:
            logger.error("'is_fraud' column not found in train DataFrame")
            return  # or handle the error appropriately

        numeric_transformer = StandardScaler()
        oh_transformer = CustomLabelEncoder()

        preprocessor = ColumnTransformer(
            transformers=[
                ('Categorical', oh_transformer, cat_features),
                ('StandardScaler', numeric_transformer, num_features)
            ],
            remainder='passthrough'
        )
        logger.info("preprocessor was successful too!")

        logger.info(f"Columns just before fit_transform: {train.columns.tolist()}")

        # Fit preprocessor on training data and transform both train and test data
        try:
            train_ = preprocessor.fit_transform(train.drop(self.config.target_column, axis=1))
            test_ = preprocessor.transform(test.drop(self.config.target_column, axis=1))
        except Exception as e:
            logger.error(f"Error during fit_transform: {e}")
            return

        logger.info("Preprocessing completed")

        # Convert transformed arrays back to DataFrame
        train__ = pd.DataFrame(train_, columns=features)
        test__ = pd.DataFrame(test_, columns=features)

        # Concatenate the target column to the DataFrame
        train__[self.config.target_column] = train[self.config.target_column].values
        test__[self.config.target_column] = test[self.config.target_column].values

        # Save preprocessed data to CSV files
        train__.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test__.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        dump(preprocessor, os.path.join(self.config.root_dir, "preprocessor.joblib"))

        logger.info(f"Saved preprocessed data to {os.path.join(self.config.root_dir, 'train.csv')} and {os.path.join(self.config.root_dir, 'test.csv')}")
        logger.info(f"Data split into training and test sets (test_size: {self.config.test_size})")
        logger.info(f"Training features shape: {train.shape}")
        logger.info(f"Test features shape: {test.shape}")

        print(f"Data split into training and test sets (test_size: {self.config.test_size})")
        print(f"Training features shape: {train.shape}")
        print(f"Test features shape: {test.shape}")
