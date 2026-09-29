import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA = "student_performance.csv"
MODEL = "student_model.joblib"

df = pd.read_csv(DATA)
features = ["study_hours","attendance","previous_score","assignments","sleep_hours","engagement"]
X = df[features]
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = RandomForestRegressor(
    n_estimators=250, max_depth=8, min_samples_leaf=2, random_state=42
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
print(f"R2:  {r2_score(y_test, pred):.3f}")

joblib.dump(model, MODEL)
print(f"Saved {MODEL}")
