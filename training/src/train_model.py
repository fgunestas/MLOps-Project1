import warnings

warnings.filterwarnings(action="ignore")
from functools import partial
from typing import Callable

import hydra
import joblib
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from hyperopt import STATUS_OK, Trials, fmin, hp,tpe
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def load_data(path:DictConfig):
    X_train=pd.read_csv(abspath(path.X_train.path))
    X_test=pd.read_csv(abspath(path.X_test.path))
    y_train=pd.read_csv(abspath(path.y_train.path))
    y_test=pd.read_csv(abspath(path.y_test.path))
    return X_train,X_test,y_train,y_test

def get_objective(X_train:pd.DataFrame,
                  y_train:pd.DataFrame,
                  X_test:pd.DataFrame,
                  y_test:pd.DataFrame,
                  config:DictConfig,
                  space:dict,
                  ):
    model=XGBClassifier()