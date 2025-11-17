import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_processed.csv"
MODEL_DIR = BASE_DIR / "model"

categorical_cols = [
    "protocol_type",
    "service",
    "flag",
]

numeric_cols = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(TRAIN_PATH)
    y = (df["is_attack"].astype(int)).values
    encoders = {}
    X_cat = []
    for col in categorical_cols:
        le = LabelEncoder()
        series = df[col].astype(str)
        le.fit(series)
        classes = list(le.classes_)
        mapping = {c: i for i, c in enumerate(classes)}
        if "__unknown__" not in mapping:
            mapping["__unknown__"] = len(mapping)
        encoders[col] = {"classes": classes, "mapping": mapping}
        X_cat.append(series.map(lambda v: mapping.get(v, mapping["__unknown__"])) .astype(int).values.reshape(-1, 1))
    X_cat_arr = np.hstack(X_cat) if X_cat else np.empty((len(df), 0))
    scaler = StandardScaler()
    X_num = df[numeric_cols].astype(float).values
    X_num_scaled = scaler.fit_transform(X_num)
    X = np.hstack([X_num_scaled, X_cat_arr])
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X, y)
    with open(MODEL_DIR / "intrusion_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump({"cols": numeric_cols, "scaler": scaler}, f)
    with open(MODEL_DIR / "label_encoders.pkl", "wb") as f:
        pickle.dump({"cols": categorical_cols, "encoders": encoders}, f)

if __name__ == "__main__":
    main()