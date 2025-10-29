import os, sys
import numpy as np
from functools import partial
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
try:
    from sklearn.metrics import root_mean_squared_error as mean_squared_error
except:
    from sklearn.metrics import mean_squared_error
    mean_squared_error = partial(mean_squared_error, squared=False)
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "LimiX"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from inference.predictor import LimiXPredictor

house_data = fetch_california_housing()
X, y = house_data.data, house_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 官方示例常用的 y 标准化
y_mean, y_std = y_train.mean(), y_train.std()
y_train_norm = (y_train - y_mean) / y_std
y_test_norm = (y_test - y_mean) / y_std

model_path = "/userdata/banyongping/tempcode/LimiX-16M.ckpt"
inference_config = "/userdata/banyongping/tempcode/LimiX/config/reg_default_noretrieval.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reg = LimiXPredictor(device=device, model_path=model_path, inference_config=inference_config)
y_pred = reg.predict(X_train, y_train_norm, X_test)

rmse = mean_squared_error(y_test_norm, y_pred)
r2 = r2_score(y_test_norm, y_pred)
print(f"RMSE: {rmse}")
print(f"R2: {r2}")