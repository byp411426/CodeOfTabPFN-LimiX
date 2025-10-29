from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

# 加载数据
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# 初始化分类器（可调参：n_estimators 等）
clf = TabPFNClassifier(n_estimators=8)  # 提高到 16/32 可更稳，但更慢

# 训练
clf.fit(X_train, y_train)

# 概率预测与 ROC AUC
proba = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, proba[:, 1]))

# 类别预测与准确率
preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))