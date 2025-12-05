import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# NSL-KDD column names
NSL_KDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label","difficulty"
]

# 4-CLASS CATEGORY MAP (USED IN RESEARCH PAPER)
ATTACK_MAP = {
    # DoS
    "back":"dos","land":"dos","neptune":"dos","pod":"dos","smurf":"dos","teardrop":"dos",

    # Probe
    "ipsweep":"probe","nmap":"probe","portsweep":"probe","satan":"probe",

    # R2L
    "ftp_write":"r2l","guess_passwd":"r2l","imap":"r2l","multihop":"r2l",
    "phf":"r2l","spy":"r2l","warezclient":"r2l","warezmaster":"r2l",

    # U2R
    "buffer_overflow":"u2r","loadmodule":"u2r","perl":"u2r","rootkit":"u2r",
}

def convert_to_4class(label):
    if label == "normal":
        return "normal"
    return ATTACK_MAP.get(label, None)   # unknown attacks → None (drop)

def load_nsl_kdd(train_path, test_path):
    # Load
    train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    test  = pd.read_csv(test_path,  header=None, names=NSL_KDD_COLUMNS)

    # Drop difficulty
    train = train.drop(columns=["difficulty"])
    test = test.drop(columns=["difficulty"])

    # Convert to 4-class labels
    train["label"] = train["label"].apply(convert_to_4class)
    test["label"]  = test["label"].apply(convert_to_4class)

    # Drop rows with unknown attacks
    train = train.dropna(subset=["label"])
    test = test.dropna(subset=["label"])

    # Encode categorical
    cat_cols = ["protocol_type", "service", "flag"]
    encoders = {}
    for col in cat_cols:
        enc = LabelEncoder()
        train[col] = enc.fit_transform(train[col])
        test[col] = enc.transform(test[col])
        encoders[col] = enc

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.drop(columns=["label"]))
    X_test  = scaler.transform(test.drop(columns=["label"]))

    # Encode labels
    y_enc = LabelEncoder()
    y_train = y_enc.fit_transform(train["label"])
    y_test  = y_enc.transform(test["label"])

    return X_train, y_train, X_test, y_test, scaler, y_enc, encoders