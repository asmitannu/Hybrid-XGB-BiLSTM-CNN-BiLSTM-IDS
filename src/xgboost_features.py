import numpy as np
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")


def _fit_xgb_get_importances(
    X_train, y_train,
    n_estimators=80,        # FAST
    max_depth=6,            # FAST
    lr=0.1,
    seed=0,
    n_jobs=4                # FAST
):
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        random_state=seed,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=n_jobs,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model.feature_importances_


def select_top_features(
    X_train, y_train, X_test,
    num_features=40,
    random_state=42,
    n_runs=2,               # FAST
    n_estimators=80,        # FAST
    max_depth=6,            # FAST
    learning_rate=0.1,
    n_jobs=4                # FAST
):
    rng = np.random.RandomState(random_state)
    n_features = X_train.shape[1]
    importances = np.zeros((n_runs, n_features), dtype=float)

    # Ensemble (fast)
    for i in range(n_runs):
        seed = int(rng.randint(0, 2**31 - 1))
        imp = _fit_xgb_get_importances(
            X_train, y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
            lr=learning_rate,
            seed=seed,
            n_jobs=n_jobs
        )
        importances[i, :] = imp

    mean_importance = importances.mean(axis=0)

    # Rank
    rankings = np.argsort(mean_importance)
    if len(rankings) <= num_features:
        idx = rankings
    else:
        idx = rankings[-num_features:]

    idx = np.sort(idx)

    X_train_sel = X_train[:, idx]
    X_test_sel = X_test[:, idx]
    return X_train_sel, X_test_sel, idx
