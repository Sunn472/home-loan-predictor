import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
                
from src.components.data_transformation import DataTransformation

if __name__=="__main__":
    transformer=DataTransformation()
    train_path,test_path,preprocessor_path=transformer.initiate_data_transformation()
    print(f"Transformed train data saved at: {train_path}")
    print(f"Transformed test data saved at: {test_path}")
    print(f"Preprocessor object saved at: {preprocessor_path}")
    print('Data Transformation process finished')