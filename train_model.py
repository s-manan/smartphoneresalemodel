"""Train the prototype model. Current included CSV is synthetic and for demonstration only."""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np, json

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "data_synthetic.csv")
features = ["brand", "model", "age_years", "condition", "storage_gb", "ram_gb", "battery_health_pct"]
target = "resale_price_inr"
cat = ["brand", "model", "condition"]
num = ["age_years", "storage_gb", "ram_gb", "battery_health_pct"]
prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                           ("num", "passthrough", num)])
pipe = Pipeline([("preprocess", prep),
                 ("regressor", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                     random_state=42, n_jobs=-1))])
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=.2, random_state=42)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
joblib.dump(pipe, ROOT / "models" / "resale_model.joblib")
metrics = {"rows": len(df), "test_rows": len(y_test),
           "MAE_INR": round(mean_absolute_error(y_test, pred), 2),
           "RMSE_INR": round(np.sqrt(mean_squared_error(y_test, pred)), 2),
           "R2": round(r2_score(y_test, pred), 4),
           "warning": "These metrics are only on synthetic data; they do not establish real-world accuracy."}
(ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
