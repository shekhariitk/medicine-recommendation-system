import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj
from src.utils import evaluate_model

import os, sys
from dataclasses import dataclass

# Model 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC




## Model Trainning Configuration

@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

## Model Training Class
class ModelTrainerClass:
    def __init__(self):
        self.model_trainer_config  = ModelTrainerconfig()

    def  initiate_model_training(self,train_arr, test_arr):
        try:
            logging.info("Splitting independent and Dependent Variable")
            X_train,y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]

            )

            ## Train multiple models

            models = {

                "LogisticRegression":LogisticRegression(),
                "DecisionTreeClassifier":DecisionTreeClassifier(),
                "RandomForestClassifier":RandomForestClassifier(),
                "Extratressclassifier":ExtraTreesClassifier(),
                "KNeighborsClassifier":KNeighborsClassifier(n_neighbors=5),
                "AdaBoostClassifier":AdaBoostClassifier(),
                "SVC":SVC()
                }
            # Define the parameter grid
            param_grid = {
                "LogisticRegression": {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [100, 200, 500]
                    },
    
                "DecisionTreeClassifier": {

                   'max_depth': [None, 10, 20, 30],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4],
                   'criterion': ['gini', 'entropy']
                   },
    
                "RandomForestClassifier": {
                   'n_estimators': [50, 100, 200],
                   'max_depth': [None, 10, 20, 30],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4],
                   #'max_features': [None, 'sqrt', 'log2'],
                   #'bootstrap': [True, False]
                   },
    
                "Extratressclassifier": {
                   'n_estimators': [50, 100, 200],
                   'max_depth': [None, 10, 20, 30],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4],
                   #'max_features': ['auto', 'sqrt', 'log2'],
                   #'bootstrap': [True, False]
                   },
    
                "KNeighborsClassifier": {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                    },
    
                "AdaBoostClassifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1, 10]
                    },

                "SVC": {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto'],
                    #'degree': [3, 5],
                    #'coef0': [0.0, 0.1, 0.5]
                    }
}
            model_report : dict = evaluate_model(X_train,y_train, X_test, y_test, models,param=param_grid)
            print(model_report)
            print("\n ==================================================================================")
            logging.info(f"Model report info : {model_report}")

            ## To get best model from model dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name  = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f"Best model found , Best model name is {best_model_name} and that R2 Score: {best_model_score}")
            print("\n=================================================================")
            logging.info(f"Best model found , Best model name is {best_model_name} and that R2 Score: {best_model_score}")


            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except   Exception as e:
            logging.info("Error occured in model trainer path") 
            raise CustomException(e, sys)
        
        