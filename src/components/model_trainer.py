import os 
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from src.utils.main_utils import MainUtils 

@dataclass
class ModelTrainerConfig:
    artifact_folder=os.path.join('artifacts')
    trained_model_path=os.path.join(artifact_folder,'model.pkl')
    expected_accuracy=0.45
    model_config_path: str =os.path.join('config','model.yaml')


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.utils = MainUtils()
        self.model = {
            'XGBClassifier': XGBClassifier(n_jobs=-1, verbosity=1),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'RandomForestClassifier': RandomForestClassifier()
        }
        self.model_param_grid = self.utils.read_yaml_file(self.config.model_config_path)['model_selection']['model']

    def evaluate_model(self, X_train, y_train, X_test, y_test):
        logging.info(f"Evaluating Model.......")
        report = {}
        for name, model in self.model.items():
            logging.info(f"Training base model {name}......")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            report[name] = score
            logging.info(f"{name} accuracy: {score:.4f}")
            print(f"{name} accuracy: {score:.4f}")
        logging.info(f"Model Evaluation report: {report}")
        print(f"Model Evaluation report: {report}")
        return report

    def finetune_best_model(self, model_name, model, X_train, y_train):
        print(f"Starting GridSearch for {model_name} .... ")
        logging.info(f"Starting GridSearch for {model_name} .... ")
        param_grid = self.model_param_grid[model_name]['search_param_grid']
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"Best params for {model_name}: {best_params}")
        logging.info(f"Best params for {model_name}: {best_params}")
        model.set_params(**best_params)
        return model

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        logging.info(f"Initiating model training.....")
        try:
            report = self.evaluate_model(X_train, y_train, X_test, y_test)
            best_model_name = max(report, key=report.get)
            logging.info(f"Best model selected: {best_model_name} with accuracy {report}")

            if report[best_model_name] < self.config.expected_accuracy:
                raise CustomException(f"Model accuracy {report[best_model_name]} is low accuracy")

            best_model = self.finetune_best_model(best_model_name, self.model[best_model_name], X_train, y_train)

            # Save the model
            self.utils.save_object(self.config.trained_model_path, best_model)
            logging.info(f"Trained Model Saved at: {self.config.trained_model_path}")
            return self.config.trained_model_path

        except Exception as e:
            raise Exception(e)
