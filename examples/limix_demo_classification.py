import os, sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch

# 让 Python 能 import 本仓库
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "LimiX"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from inference.predictor import LimiXPredictor

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model_path = "/userdata/banyongping/tempcode/LimiX-16M.ckpt"
# 无检索配置，适合普通设备
inference_config = "/userdata/banyongping/tempcode/LimiX/config/cls_default_noretrieval.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf = LimiXPredictor(device=device, model_path=model_path, inference_config=inference_config)
proba = clf.predict(X_train, y_train, X_test)

print("roc_auc_score:", roc_auc_score(y_test, proba[:, 1]))
print("accuracy_score:", accuracy_score(y_test, np.argmax(proba, axis=1)))