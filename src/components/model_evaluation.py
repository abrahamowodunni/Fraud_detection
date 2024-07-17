import os
import pandas as pd
from src import logger
from src.entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  roc_curve, auc, classification_report
import numpy as np
import joblib
from src.utils.common import save_json
from pathlib import Path
import matplotlib.pyplot as plt

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,true, predicted):
        accuracy = accuracy_score(true, predicted)
        precision = precision_score(true, predicted, average='weighted')  # Update here
        recall = recall_score(true, predicted, average='weighted')  # Update here
        f1 = f1_score(true, predicted, average='weighted')  # Update here
        return accuracy, precision, recall, f1
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    
    def save_results(self):

        test_data = pd.read_csv(self.config.test_data_path)
        test_data[self.config.target_column] = test_data[self.config.target_column]
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]
        
        predicted_qualities = model.predict(test_x)
        predicted_proba = model.predict_proba(test_x)[:, 1]

        (accuracy, precision, recall, f1) = self.eval_metrics(test_y, predicted_qualities)
        
        # Saving metrics as local
        scores = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        save_json(path=Path(self.config.metric_file_name), data=scores)
        
        # Plot ROC curve and save image
        roc_curve_path = Path(self.config.roc_curve_file_name)
        self.plot_roc_curve(test_y, predicted_proba, save_path=roc_curve_path)

        # Evaluate with optimal threshold
        predicted_optimal = (predicted_proba >= self.config.optimal_threshold).astype(int)
        (accuracy_opt, precision_opt, recall_opt, f1_opt) = self.eval_metrics(test_y, predicted_optimal)

        # Print classification report
        report_default = classification_report(test_y, predicted_qualities, output_dict=True)
        report_optimal = classification_report(test_y, predicted_optimal, output_dict=True)

        print("Classification Report with Default Threshold:\n", classification_report(test_y, predicted_qualities))
        print("Classification Report with Optimal Threshold:\n", classification_report(test_y, predicted_optimal))
        
        # Save metrics with optimal threshold
        scores_opt = {'accuracy': accuracy_opt, 'precision': precision_opt, 'recall': recall_opt, 'f1': f1_opt}
        save_json(path=Path(self.config.optimal_metric_file_name), data=scores_opt)

        # Extract and save precision and recall for fraud class
        fraud_metrics = {
            'precision_fraud': report_optimal['1']['precision'],
            'recall_fraud': report_optimal['1']['recall'],
            'f1_fraud': report_optimal['1']['f1-score']
        }
        save_json(path=Path(self.config.fraud_metric_file_name), data=fraud_metrics)