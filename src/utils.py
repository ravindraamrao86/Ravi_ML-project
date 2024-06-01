# src/utils.py
import pickle
import os,sys
from src.components.exception import CustomException
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve ,precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score

def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
### Load model    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_objt:
            return pickle.load(file_objt)

        pass
    except Exception as e:
        raise CustomException(e, sys)




## model Evaluation & GridSearch CV

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name in models:
            model = models[model_name]
            param = params[model_name]
            
            GS = GridSearchCV(model, param, cv=5)
            GS.fit(X_train, y_train)
            model.set_params(**GS.best_params_)
            model.fit(X_train, y_train)

            ## Make prediction 
            y_pred = model.predict(X_test)
            test_model_accuracy = accuracy_score(y_test, y_pred)

            report[model_name] = test_model_accuracy
        
        return report
        
    except Exception as e:
        raise CustomException(e, sys)
