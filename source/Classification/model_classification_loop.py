import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

import random
import torch


# -----------------------
# Reproducibility
# -----------------------
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------
# Helpers: indexing & lagging (no look-ahead)
# -----------------------
def ensure_datetime_index_from_timestamp(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col)
    df = df.set_index(ts_col)
    return df

def make_lag1_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].shift(1)  # X_t uses info up to t-1
    return X

def time_split_last20(X, y):
    n = len(y)
    if n < 5:
        return None
    cut = int(np.floor(0.8 * n))
    if cut < 1 or cut >= n:
        return None
    return X[:cut], y[:cut], X[cut:], y[cut:]


# -----------------------
# RandomForest tuning (time split last 20%)
# -----------------------
def tune_rf_time_split(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42):
    split = time_split_last20(X_train, y_train)
    if split is None:
        # fallback
        return RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )

    Xtr, ytr, Xval, yval = split

    grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 3, 6],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt", None],
    }

    best = None
    best_score = -np.inf

    for n_estimators in grid["n_estimators"]:
        for max_depth in grid["max_depth"]:
            for min_samples_leaf in grid["min_samples_leaf"]:
                for max_features in grid["max_features"]:
                    m = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=seed,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                    )
                    m.fit(Xtr, ytr)
                    pred = m.predict(Xval)
                    score = balanced_accuracy_score(yval, pred)
                    if score > best_score:
                        best_score = score
                        best = m

    # Refit best hyperparams on FULL train
    params = best.get_params()
    refit = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    return refit


# -----------------------
# Mantis data builder: sequence up to t-1 (no look-ahead)
# -----------------------
def mantis_resize_to_512(X: np.ndarray) -> np.ndarray:
    """
    X: (N, C, L) float
    returns: (N, C, 512)
    """
    import torch.nn.functional as F
    Xt = torch.tensor(X, dtype=torch.float32)
    Xt = F.interpolate(Xt, size=512, mode="linear", align_corners=False)
    return Xt.cpu().numpy()

def build_mantis_Xy(
    feats: np.ndarray,  # shape (T, C)
    y: np.ndarray,      # shape (T,)
    end_pos: int,       # build samples with label positions i in [context_len, end_pos)
    context_len: int,
):
    """
    Sample i corresponds to predicting y[i] using feats[i-context_len : i] (history up to i-1).
    Returns X (N,C,L) and y (N,).
    """
    X_list, y_list = [], []
    for i in range(context_len, end_pos):
        window = feats[i - context_len:i]  # (L,C)
        if not np.isfinite(window).all():
            continue
        X_list.append(window.T)  # (C,L)
        y_list.append(y[i])
    if len(X_list) == 0:
        return None, None
    X = np.stack(X_list, axis=0).astype(np.float32)  # (N,C,L)
    y_out = np.asarray(y_list, dtype=int)
    X = mantis_resize_to_512(X)
    return X, y_out

def build_mantis_X_one(
    feats: np.ndarray,  # (T,C)
    pos: int,           # predict y[pos] using feats[pos-context_len:pos]
    context_len: int,
):
    if pos < context_len:
        return None
    window = feats[pos - context_len:pos]  # (L,C)
    if not np.isfinite(window).all():
        return None
    X = window.T[None, ...].astype(np.float32)  # (1,C,L)
    X = mantis_resize_to_512(X)
    return X


# -----------------------
# Unified OOS loop with refit_every
# -----------------------
def expanding_oos_refit_every_cls(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str = "state",
    start_oos: str = "2007-01-01",
    start_date: str = "2000-01-05",
    min_train: int = 120,
    refit_every: int = 30,              # <-- n_days (trading days)
    model: str = "logit",               # "logit" | "rf" | "tabpfn25" | "mantis_head" | "mantis_rf_head"
    mantis_context_len: int = 60,       # used only for mantis models
    seed: int = 42,
    device: str | None = None,
    quiet: bool = False,
):
    """
    Predict day t using only info up to t-1.
    Refit the model only every refit_every steps in the OOS loop (trading days).

    Returns:
      y_true (np.ndarray), y_pred (np.ndarray), oos_dates (pd.DatetimeIndex)
    """
    set_global_seed(seed)

    df = ensure_datetime_index_from_timestamp(data, ts_col="timestamp")
    df = df.loc[pd.Timestamp(start_date):].copy()

    start_ts = pd.Timestamp(start_oos)
    loop_dates = df.index[df.index >= start_ts]

    # lag-1 tabular design for logit/rf/tabpfn
    X_lag = make_lag1_features(df, feature_cols)
    y_all = df[target_col].to_numpy()

    # mantis raw features matrix for sequence building (T,C)
    feats_raw = df[feature_cols].to_numpy(dtype=float)

    # models / state
    fitted_model = None
    last_fit_step = None

    # for mantis: load foundation model once
    mantis_network = None
    mantis_trainer = None
    rf_head = None

    # decide device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    y_true_list, y_pred_list, date_list = [], [], []

    for step, date_t in enumerate(loop_dates):
        pos = df.index.get_loc(date_t)  # integer position in full df

        # need label at date_t
        y_true = df.loc[date_t, target_col]
        if pd.isna(y_true):
            continue

        # training uses only samples with label date < date_t  (strictly past targets)
        # for lag-1 tabular: rows < date_t already incorporate t-1 predictors via shift
        train_mask = (df.index < date_t)

        # refit schedule
        need_refit = (fitted_model is None) or (last_fit_step is None) or ((step - last_fit_step) >= refit_every)

        if need_refit:
            set_global_seed(seed + step)

            if model in ("logit", "rf", "tabpfn25"):
                X_train_df = X_lag.loc[train_mask, feature_cols]
                y_train = df.loc[train_mask, target_col]

                # drop NaNs (from shifting) and any missing labels
                m = X_train_df.notna().all(axis=1) & y_train.notna()
                X_train = X_train_df.loc[m].to_numpy(float)
                y_train = y_train.loc[m].to_numpy(int)

                if len(y_train) < min_train:
                    fitted_model = None
                    continue

                if model == "logit":
                    fitted_model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            solver="lbfgs",
                            random_state=seed + step,
                        )),
                    ])
                    fitted_model.fit(X_train, y_train)

                elif model == "rf":
                    rf = tune_rf_time_split(X_train, y_train, seed=seed + step)
                    rf.fit(X_train, y_train)
                    fitted_model = rf

                elif model == "tabpfn25":
                    from tabpfn import TabPFNClassifier
                    fitted_model = TabPFNClassifier(device=device)
                    fitted_model.fit(X_train, y_train)

            elif model in ("mantis_head", "mantis_rf_head"):
                from mantis.architecture import Mantis8M
                from mantis.trainer import MantisTrainer

                # load backbone once
                if mantis_network is None:
                    mantis_network = Mantis8M(device=device)
                    mantis_network = mantis_network.from_pretrained("paris-noah/Mantis-8M")

                mantis_trainer = MantisTrainer(device=device, network=mantis_network)

                # build mantis training set using labels < date_t
                end_pos = pos  # labels indices [0..pos-1] are allowed
                X_tr, y_tr = build_mantis_Xy(feats_raw, y_all, end_pos=end_pos, context_len=mantis_context_len)

                if X_tr is None or len(y_tr) < min_train:
                    fitted_model = None
                    rf_head = None
                    continue

                if model == "mantis_head":
                    # fit head only (try fine_tuning_type='head', fallback to default)
                    try:
                        mantis_trainer.fit(X_tr, y_tr, fine_tuning_type="head")
                    except TypeError:
                        mantis_trainer.fit(X_tr, y_tr)
                    fitted_model = mantis_trainer
                    rf_head = None

                elif model == "mantis_rf_head":
                    # extract embeddings, fit RF head with time-split tuning
                    Z_tr = mantis_trainer.transform(X_tr)
                    Z_tr = np.asarray(Z_tr, float)

                    rf = tune_rf_time_split(Z_tr, y_tr, seed=seed + step)
                    rf.fit(Z_tr, y_tr)

                    fitted_model = mantis_trainer
                    rf_head = rf

            else:
                raise ValueError("Unknown model. Use: logit, rf, tabpfn25, mantis_head, mantis_rf_head")

            last_fit_step = step
            if not quiet:
                print(f"[{model}] refit at {date_t.date()} using data up to {df.index[pos-1].date() if pos>0 else 'N/A'}")

        # if we couldn't fit, skip
        if fitted_model is None:
            continue

        # -------- predict for date_t (using info up to t-1) --------
        if model in ("logit", "rf", "tabpfn25"):
            x_row = X_lag.loc[date_t, feature_cols]
            if x_row.isna().any():
                continue
            X_te = x_row.to_numpy(float).reshape(1, -1)
            y_hat = int(fitted_model.predict(X_te)[0])

        elif model in ("mantis_head", "mantis_rf_head"):
            X_one = build_mantis_X_one(feats_raw, pos=pos, context_len=mantis_context_len)
            if X_one is None:
                continue

            if model == "mantis_head":
                y_hat = int(fitted_model.predict(X_one)[0])

            else:
                Z_one = fitted_model.transform(X_one)
                Z_one = np.asarray(Z_one, float)
                y_hat = int(rf_head.predict(Z_one)[0])

        y_true_list.append(int(y_true))
        y_pred_list.append(int(y_hat))
        date_list.append(date_t)

    return np.asarray(y_true_list, int), np.asarray(y_pred_list, int), pd.DatetimeIndex(date_list)
