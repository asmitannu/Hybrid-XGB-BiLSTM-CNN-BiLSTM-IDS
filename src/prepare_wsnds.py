import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_wsnds(csv_path, test_size=0.2):
    df = pd.read_csv(csv_path)

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Label column
    label_col = "Attack type"

    # Extract features + labels
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Encode label
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=test_size, random_state=42, stratify=y_enc
    )

    # Generate feature names
    feature_names = X.columns.tolist()

    return X_train, X_test, y_train, y_test, scaler, le, feature_names
