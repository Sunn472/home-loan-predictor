import os
import sys
import pandas as pd
from  src.constant import  APPLICATION_TRAIN_PATH
from src.exception import CustomException
from dataclasses import dataclass
from src.logger import logging


@dataclass
class  DataIngestionConfig:
    artifacts_folder: str="artifacts"
    train_file_name: str="application_train.csv"
    test_file_name: str="application_test.csv"


class DataIngestion:
    def __init__(self):
        self.config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Start")
        try:
            os.makedirs(self.config.artifacts_folder,exist_ok=True)
            logging.info(f"Artifacts folder created at: {self.config.artifacts_folder}")
            dst_path=os.path.join(self.config.artifacts_folder,self.config.train_file_name)
            df=pd.read_csv(APPLICATION_TRAIN_PATH)
            df.to_csv(dst_path,index=False)
            logging.info(f"Data saved to {dst_path}")
            logging.info("Data Ingestion completed succesfully")
        except Exception as e:
            logging.info(f"Error occured during data ingestion: {str(e)}")
            raise CustomException(e,sys)

