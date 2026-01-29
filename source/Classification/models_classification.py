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
    grid = {
        "n_estimators": [200],
        "max_depth": [None],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt"],
    }
    best = None
    best_score = -np.inf
    refit = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    return refit
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
    print(params)
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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def fit_mantis_head_frozen(mantis_trainer, X_tr, y_tr, head: str, seed: int):
    """
    Frozen backbone. Train only the head on top of embeddings.
    head: "lr" or "rf"
    """
    Z_tr = np.asarray(mantis_trainer.transform(X_tr), float)

    if head == "lr":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )),
        ])
        clf.fit(Z_tr, y_tr)
        return clf

    if head == "rf":
        rf = tune_rf_time_split(Z_tr, y_tr, seed=seed)
        rf.fit(Z_tr, y_tr)
        return rf

    raise ValueError("head must be 'lr' or 'rf'")

# -----------------------
# Unified OOS loop with refit_every
# -----------------------
# ... keep your imports and everything above ...

def expanding_oos_refit_every_cls(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str = "state",
    start_oos: str = "2007-01-01",
    start_date: str = "2000-01-05",
    min_train: int = 120,
    refit_every: int = 30,
    model: str = "logit",  # add "majority" here too
    mantis_context_len: int = 512,
    seed: int = 42,
    device: str | None = None,
    quiet: bool = False,
    max_train: int | None = None,   
    mantis_max_train_windows: int | None = None, 
):
    set_global_seed(seed)

    df = ensure_datetime_index_from_timestamp(data, ts_col="timestamp")
    df = df.loc[pd.Timestamp(start_date):].copy()

    start_ts = pd.Timestamp(start_oos)
    loop_dates = df.index[df.index >= start_ts]

    X_lag = make_lag1_features(df, feature_cols)
    y_all = df[target_col].to_numpy()
    feats_raw = df[feature_cols].to_numpy(dtype=float)

    fitted_model = None
    last_fit_step = None

    # --- NEW state for Mantis + Majority ---
    head_model = None          # <-- NEW: stores LR/RF head on Mantis embeddings
    majority_label = None      # <-- NEW: stores majority class for baseline

    mantis_network = None
    mantis_trainer = None

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    y_true_list, y_pred_list, date_list = [], [], []
    if model in ("mantis_head", "mantis_rf_head"):  #changed
        from mantis.architecture import Mantis8M  #changed
        from mantis.trainer import MantisTrainer  #changed

        if device is None:  #changed
            if torch.cuda.is_available():  #changed
                device = "cuda"  #changed
            elif torch.backends.mps.is_available():  #changed
                device = "mps"  #changed
            else:  #changed
                device = "cpu"  #changed

        # load backbone once  #changed
        mantis_network = Mantis8M(device=device)  #changed
        mantis_network = mantis_network.from_pretrained("paris-noah/Mantis-8M")  #changed
        mantis_trainer = MantisTrainer(device=device, network=mantis_network)  #changed

        def build_mantis_X_block(feats: np.ndarray, positions: list[int], context_len: int):  #changed
            X_list = []  #changed
            ok_pos = []  #changed
            for p in positions:  #changed
                if p < context_len:  #changed
                    continue  #changed
                w = feats[p - context_len:p]  #changed
                if not np.isfinite(w).all():  #changed
                    continue  #changed
                X_list.append(w.T)  #changed  # (C,L)
                ok_pos.append(p)  #changed
            if len(X_list) == 0:  #changed
                return None, None  #changed
            X = np.stack(X_list, axis=0).astype(np.float32)  #changed  # (N,C,L)
            X = mantis_resize_to_512(X)  #changed
            return X, ok_pos  #changed

        i = 0  #changed
        while i < len(loop_dates):  #changed
            date_t = loop_dates[i]  #changed
            pos0 = df.index.get_loc(date_t)  #changed

            # ---- TRAINING (strictly past labels) ----  #changed
            X_tr, y_tr = build_mantis_Xy(  #changed
                feats_raw, y_all, end_pos=pos0, context_len=mantis_context_len  #changed
            )  #changed

            if X_tr is None or len(y_tr) < min_train:  #changed
                i += 1  #changed
                continue  #changed

            if mantis_max_train_windows is not None and len(y_tr) > mantis_max_train_windows:  #changed
                X_tr = X_tr[-mantis_max_train_windows:]  #changed
                y_tr = y_tr[-mantis_max_train_windows:]  #changed

            set_global_seed(seed + i)  #changed

            # fit ONLY head (backbone frozen via transform inside helper)  #changed
            if model == "mantis_head":  #changed
                head_model = fit_mantis_head_frozen(  #changed
                    mantis_trainer, X_tr, y_tr, head="lr", seed=seed + i  #changed
                )  #changed
            else:  #changed
                head_model = fit_mantis_head_frozen(  #changed
                    mantis_trainer, X_tr, y_tr, head="rf", seed=seed + i  #changed
                )  #changed

            # ---- PREDICT BLOCK ----  #changed
            j = min(i + refit_every, len(loop_dates))  #changed
            block_dates = loop_dates[i:j]  #changed
            block_positions = [df.index.get_loc(d) for d in block_dates]  #changed

            X_blk, ok_positions = build_mantis_X_block(  #changed
                feats_raw, block_positions, context_len=mantis_context_len  #changed
            )  #changed
            if X_blk is not None:  #changed
                Z_blk = np.asarray(mantis_trainer.transform(X_blk), float)  #changed
                preds = head_model.predict(Z_blk).astype(int)  #changed

                # map ok_positions back to dates  #changed
                ok_dates = [df.index[p] for p in ok_positions]  #changed
                y_true_blk = df.loc[ok_dates, target_col].to_numpy(int)  #changed

                y_true_list.extend(y_true_blk.tolist())  #changed
                y_pred_list.extend(preds.tolist())  #changed
                date_list.extend(ok_dates)  #changed

            if not quiet:  #changed
                prev = df.index[pos0 - 1].date() if pos0 > 0 else "N/A"  #changed
                print(f"[{model}] refit at {date_t.date()} using data up to {prev} | predicted {j-i} days")  #changed

            i = j  #changed

        return np.asarray(y_true_list, int), np.asarray(y_pred_list, int), pd.DatetimeIndex(date_list)
    if model == "tabpfn25":  #changed
        from tabpfn import TabPFNClassifier  #changed

        i = 0  #changed
        while i < len(loop_dates):  #changed
            date_t = loop_dates[i]  #changed
            pos = df.index.get_loc(date_t)  #changed

            train_mask = (df.index < date_t)  #changed
            X_train_df = X_lag.loc[train_mask, feature_cols]  #changed
            y_train_s = df.loc[train_mask, target_col]  #changed

            m = X_train_df.notna().all(axis=1) & y_train_s.notna()  #changed
            X_train = X_train_df.loc[m].to_numpy(np.float32)  #changed
            y_train = y_train_s.loc[m].to_numpy(int)  #changed

            if max_train is not None and len(y_train) > max_train:  #changed
                X_train = X_train[-max_train:]  #changed
                y_train = y_train[-max_train:]  #changed

            if len(y_train) < min_train:  #changed
                i += 1  #changed
                continue  #changed

            set_global_seed(seed + i)  #changed
            model_tabpfn = TabPFNClassifier(device=device)  #changed
            model_tabpfn.fit(X_train, y_train)  #changed

            j = min(i + refit_every, len(loop_dates))  #changed
            block_dates = loop_dates[i:j]  #changed

            X_block_df = X_lag.loc[block_dates, feature_cols]  #changed
            y_block_s = df.loc[block_dates, target_col]  #changed

            ok = X_block_df.notna().all(axis=1) & y_block_s.notna()  #changed
            if ok.any():  #changed
                X_block = X_block_df.loc[ok].to_numpy(np.float32)  #changed
                preds = model_tabpfn.predict(X_block).astype(int)  #changed  # (no chunking)
                y_true_ok = y_block_s.loc[ok].to_numpy(int)  #changed
                dates_ok = X_block_df.index[ok]  #changed

                y_true_list.extend(y_true_ok.tolist())  #changed
                y_pred_list.extend(preds.tolist())  #changed
                date_list.extend(dates_ok.tolist())  #changed

            if not quiet:  #changed
                prev = df.index[pos - 1].date() if pos > 0 else "N/A"  #changed
                print(f"[tabpfn25] refit at {date_t.date()} using data up to {prev} | predicted {j-i} days")  #changed

            i = j  #changed

        return np.asarray(y_true_list, int), np.asarray(y_pred_list, int), pd.DatetimeIndex(date_list)  #changed

    for step, date_t in enumerate(loop_dates):
        pos = df.index.get_loc(date_t)

        y_true = df.loc[date_t, target_col]
        if pd.isna(y_true):
            continue

        train_mask = (df.index < date_t)

        need_refit = (fitted_model is None) or (last_fit_step is None) or ((step - last_fit_step) >= refit_every)

        if need_refit:
            set_global_seed(seed + step)

            # ---------------- TABULAR MODELS ----------------
            if model in ("logit", "rf", "tabpfn25"):
                X_train_df = X_lag.loc[train_mask, feature_cols]
                y_train_s = df.loc[train_mask, target_col]

                m = X_train_df.notna().all(axis=1) & y_train_s.notna()
                X_train = X_train_df.loc[m].to_numpy(float)
                y_train = y_train_s.loc[m].to_numpy(int)

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
                    fitted_model = TabPFNClassifier(device=device)  # TabPFN 2.5 default
                    fitted_model.fit(X_train, y_train)

                # clear mantis/majority state
                head_model = None
                majority_label = None

            # ---------------- MANTIS (FROZEN BACKBONE + HEAD ONLY) ----------------
            elif model in ("mantis_head", "mantis_rf_head"):
                from mantis.architecture import Mantis8M
                from mantis.trainer import MantisTrainer

                if mantis_network is None:
                    mantis_network = Mantis8M(device=device)
                    mantis_network = mantis_network.from_pretrained("paris-noah/Mantis-8M")

                mantis_trainer = MantisTrainer(device=device, network=mantis_network)

                X_tr, y_tr = build_mantis_Xy(
                    feats_raw, y_all, end_pos=pos, context_len=mantis_context_len
                )
                if X_tr is None or len(y_tr) < min_train:
                    fitted_model = None
                    head_model = None
                    continue

                # IMPORTANT: never call mantis_trainer.fit -> keeps backbone frozen
                if model == "mantis_head":
                    head_model = fit_mantis_head_frozen(
                        mantis_trainer, X_tr, y_tr, head="lr", seed=seed + step
                    )
                else:
                    head_model = fit_mantis_head_frozen(
                        mantis_trainer, X_tr, y_tr, head="rf", seed=seed + step
                    )

                fitted_model = mantis_trainer  # feature extractor only

                # clear majority state
                majority_label = None

            # ---------------- MAJORITY BASELINE ----------------
            elif model == "majority":
                y_train = df.loc[train_mask, target_col].dropna().to_numpy()
                if len(y_train) < min_train:
                    fitted_model = None
                    majority_label = None
                    continue

                vals, counts = np.unique(y_train.astype(int), return_counts=True)
                majority_label = int(vals[np.argmax(counts)])
                fitted_model = "majority"  # marker

                # clear mantis head state
                head_model = None

            else:
                raise ValueError("Unknown model. Use: logit, rf, tabpfn25, mantis_head, mantis_rf_head, majority")

            last_fit_step = step
            if not quiet:
                prev = df.index[pos-1].date() if pos > 0 else "N/A"
                print(f"[{model}] refit at {date_t.date()} using data up to {prev}")

        if fitted_model is None:
            continue

        # ---------------- PREDICT ----------------
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
            if head_model is None:
                continue
            Z_one = np.asarray(fitted_model.transform(X_one), float)
            y_hat = int(head_model.predict(Z_one)[0])

        elif model == "majority":
            if majority_label is None:
                continue
            y_hat = int(majority_label)

        else:
            raise RuntimeError("Internal: unknown model at prediction stage.")

        y_true_list.append(int(y_true))
        y_pred_list.append(int(y_hat))
        date_list.append(date_t)

    return np.asarray(y_true_list, int), np.asarray(y_pred_list, int), pd.DatetimeIndex(date_list)
