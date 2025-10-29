from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn_extensions.post_hoc_ensembles import AutoTabPFNRegressor

# 载入 Boston Housing（OpenML: data_id=531）
df = fetch_openml(data_id=531, as_frame=True)
X = df.data
y = df.target.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

reg = AutoTabPFNRegressor(
    max_time=1800,
    n_ensemble_models=10,
    n_estimators=8,
    device="auto",
    eval_metric="root_mean_squared_error",
    random_state=42,
)

reg.fit(X_train, y_train)
preds = reg.predict(X_test)

print("MSE:", mean_squared_error(y_test, preds))
print("R² Score:", r2_score(y_test, preds))