import os
import sys
import logging
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
                
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
    #load data as dict with X and Y
    train_data=np.load('artifacts/train.npy',allow_pickle=True).item()
    test_data=np.load('artifacts/train.npy',allow_pickle=True).item()
    X_train,y_train=train_data['X'],train_data['y']
    X_test,y_test=test_data['X'],test_data['y']
    X_train,y_train=X_train[:1000],y_train[:1000]
    X_test,y_test=X_test[:200],y_test[:200]

    trainer=ModelTrainer()
    model_path=trainer.initiate_model_trainer(X_train,y_train,X_test,y_test)
    print("Model saved at: {model_path}")