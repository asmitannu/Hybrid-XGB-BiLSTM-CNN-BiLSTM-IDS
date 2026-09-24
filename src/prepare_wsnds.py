import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_wsnds(csv_path, test_size=0.2):

    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Label column
    label_col = "Attack type"

    # Features and labels
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train test Split 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=test_size,
        random_state=42,
        stratify=y_enc
    )

    # Scaling  
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Feature names
    feature_names = X.columns.tolist()

    return X_train, X_test, y_train, y_test, scaler, le, feature_names