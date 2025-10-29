from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn_extensions.post_hoc_ensembles import AutoTabPFNClassifier

# 加载数据
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# AutoTabPFN 分类器
clf = AutoTabPFNClassifier(
    max_time=1800,           # 训练上限时间（秒），可根据资源调整
    n_ensemble_models=10,    # 内部会随机生成若干 TabPFN 配置进行集成
    n_estimators=8,          # 每个单模型内部的 transformer 个数
    device="auto",           # 有 GPU 自动用 GPU
    eval_metric="accuracy",  # 指标，自定义也可以
    random_state=42,
)

# 训练与预测
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
preds = clf.predict(X_test)

print("ROC AUC:", roc_auc_score(y_test, proba[:, 1]))
print("Accuracy:", accuracy_score(y_test, preds))