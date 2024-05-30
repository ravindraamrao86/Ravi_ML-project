# src/components/data_ingestion.py

import os
import sys

# Ensure the project root directory is in the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts/data_ingestion", "train.csv")
    test_data_path: str = os.path.join("artifacts/data_ingestion", "test.csv")
    raw_data_path: str = os.path.join("artifacts/data_ingestion", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            logging.info("Reading data using pandas library from local system")

            # Read dataset
            data = pd.read_csv(os.path.join("Notebook", "cleandata.csv"))
            logging.info("Data read completed")

            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Save raw data
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            # Split data into training and testing sets
            train_set, test_set = train_test_split(data, test_size=0.30, random_state=42)

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data split into train and test sets")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error("Error occurred in data ingestion stage")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data transformation completed successfully")

    except Exception as e:
        logging.error("Error occurred in main execution")
        raise CustomException(e,sys)
    
    # This code will be use data transformation execution
if __name__ =="__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
