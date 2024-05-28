import os,sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass  ### Create Decoreter

class DataIngestionConfig:   ### Create class
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")
    raw_data_path = os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def inititate_data_ingestion(self):
        logging.info("data ingestion started")
        try:
            logging.info("Data Reading useing pandas library from local to system")# for Error findout

            data=pd.read_csv(os.path.join("Notebook\Data\income.csv"))  ## read data set
            logging.info("Data split complited")# for Error findout
            

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True) ##Create Artifacts folder
            
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Data splited into train test split ") # for Error findout

            train_set ,test_set= train_test_split(data,test_size=.30,random_state=42) # Spliting data into training and test

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) # Save Data
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) # Save Data

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
            )
        
        except Exception as e:
            logging.info("Error Accored in data ingestion stage")
            raise CustomException(e,sys)     
        
if __name__=="__main__":
    obj = DataIngestion()
    obj.inititate_data_ingestion()  # type: ignore