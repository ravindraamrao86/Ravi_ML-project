## Features Enginnering  
## handel outlier
## handel imbalance data 
##convert categorical columns into numerical data 

# src/components/data_transformation.py

import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.components.exception import CustomException
from src.pipeline.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Data transformation started")

            numerical_features = ['age', 'workclass', 'educational-num', 'marital-status', 'occupation',
                                  'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
                                  'hours-per-week', 'native-country']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def remove_outlier_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            iqr = Q3 - Q1

            upper_limit = Q3 + 1.5 * iqr
            lower_limit = Q1 - 1.5 * iqr

            df.loc[df[col] > upper_limit, col] = upper_limit
            df.loc[df[col] < lower_limit, col] = lower_limit

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            numerical_features = ['age', 'workclass', 'educational-num', 'marital-status', 'occupation',
                                  'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
                                  'hours-per-week', 'native-country']
            
            for col in numerical_features:
                self.remove_outlier_IQR(col=col, df=train_data)
            logging.info("Outliers capped in train data")

            for col in numerical_features:
                self.remove_outlier_IQR(col=col, df=test_data)
            logging.info("Outliers capped in test data")
                
            preprocess_obj = self.get_data_transformation_obj()
            target_column = "income"
            drop_columns = [target_column]

            # Splitting data into train, test
            logging.info("Splitting data into dependent and independent features")
            input_feature_train_data = train_data.drop(drop_columns, axis=1)
            target_feature_train_data = train_data[target_column]

            logging.info("Splitting test data into dependent and independent features")
            input_feature_test_data = test_data.drop(drop_columns, axis=1)
            target_feature_test_data = test_data[target_column]

            # Apply Transformation on train and test data 
            input_train_arr = preprocess_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocess_obj.transform(input_feature_test_data)

            # Apply preprocessor object on train, test datasets
            train_array = np.c_[input_train_arr, np.array(target_feature_train_data)]
            test_array = np.c_[input_test_arr, np.array(target_feature_test_data)]

            # Save preprocessor object to artifacts folder
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocess_obj
            )

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        data_transformation = DataTransformation()
        train_data_path = "path/to/train_data.csv"
        test_data_path = "path/to/test_data.csv"
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path,
                                                                                                   test_data_path)
        logging.info("Data transformation completed successfully")
        
    except Exception as e:
        logging.error("Error occurred in main execution")
        raise CustomException(e,sys)
