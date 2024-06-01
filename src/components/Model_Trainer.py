
import os
import sys
from dataclasses import dataclass
from logger import logging
from src.utils import save_object,evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from src.components.exception import CustomException
from sklearn.model_selection import GridSearchCV 

# Model
@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts/model_trainer", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()    

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting our data into dependent and independent features")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features
                train_array[:, -1],   # Target
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier()        
            }
            params = {
                "Logistic Regression": {
                    "class_weight": ["balanced"],
                    "penalty": ['l1', 'l2'],
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "solver": ["liblinear", "saga"],
                
                },
                "Decision Tree": {
                    "class_weight": ["balanced"],
                    "criterion": ["gini", "entropy", "log_loss"],
                    "splitter": ["best", "random"],
                    "max_depth": [3, 4, 5, 6],
                    "min_samples_split": [2, 3, 4, 5],
                    "min_samples_leaf": [1,2,3],
                    "max_features": ["sqrt", "log2",None],
                },
                "Random Forest": {
                    "class_weight": ["balanced"],
                    "n_estimators": [20, 50, 30],
                    "max_depth": [10, 8, 5],
                    "min_samples_split": [2, 5, 10]
                }
            }
            ## Evaluate function 
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)
            
            ## to get best model from our report dict
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f"Best Model Found, model Name is :{best_model_name},Accuracy_Score:{best_model_score}")
            print("/n**********************************************************************************/n")
            logging.info(f"Best model found: {best_model_name} with accuracy score: {best_model_score}")

            # Save the best model
            save_object(file_path=self.model_trainer_config.train_model_file_path, obj=best_model)

            return best_model_score, best_model_name

        except Exception as e:
            raise CustomException(e, sys)
