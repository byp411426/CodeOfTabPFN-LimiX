from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor

# 载入 Boston Housing（OpenML: data_id=531）
df = fetch_openml(data_id=531, as_frame=True)
X = df.data
y = df.target.astype(float)

# 切分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# 初始化回归器（可调参：n_estimators 等）
regressor = TabPFNRegressor(n_estimators=8)

# 训练与预测
regressor.fit(X_train, y_train)
preds = regressor.predict(X_test)

# 评估
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)