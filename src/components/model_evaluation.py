import os
import pandas as pd
from src.entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import classification_report, roc_curve, auc
import joblib
from src.utils.common import save_json
from pathlib import Path
import matplotlib.pyplot as plt

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True)
        fraud_metrics = {
            'precision_fraud': round(report['1']['precision'], 2),
            'recall_fraud': round(report['1']['recall'], 2),
            'f1_fraud': round(report['1']['f1-score'], 2)
        }
        return fraud_metrics

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
        
        plt.savefig(self.config.roc_curve_file_name)
        #plt.show()
    
    def save_results(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]
        
        predicted_proba = model.predict_proba(test_x)[:, 1]
        y_pred_optimal = (predicted_proba >= self.config.optimal_threshold).astype(int)
        
        fraud_metrics = self.eval_metrics(test_y, y_pred_optimal)
        save_json(path=Path(self.config.fraud_metric_file_name), data=fraud_metrics)

        self.plot_roc_curve(test_y, predicted_proba)
