import sys
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
sys.path.insert(0, os.path.abspath('../../'))
from source.regression.modelling_utils import expanding_oos_tabular_cls
import random, numpy as np, torch

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_global_seed(42)

def make_logit_multifeature_lag_fit_predict_fn(
    base_cols: list[str] = ["SXXT", "SPX", "NKY", "SPTR", "EUR003M",
             "FEDL01", "GC1", "V2X", "MOVE", "VIX",
             "USYC2Y10", "VXJ","M1WO"],                # multiple continuous input variables
    target_col: str = "state",           # classification target (0/1)
    n_lags: int = 1,
    C: float = 1.0,
    class_weight=None,
    max_iter: int = 1000,
    return_proba: bool = False,
):
    """
    Create a time-series logistic regression fit_predict function that uses
    lagged values of multiple columns in `base_cols` to predict `target_col`.

    For a date_t, we predict y_t using base_cols[t-1], ..., base_cols[t-n_lags].
    Training uses only past data.
    """

    def fit_predict(est: pd.DataFrame, row_t: pd.Series):
        if len(est) <= n_lags:
            return np.nan

        # ---- TRAINING DATA: build lags on est ONLY (strictly past) ----
        tmp = est[base_cols + [target_col]].copy()
        for col in base_cols:
            for k in range(1, n_lags + 1):
                tmp[f"{col}_lag{k}"] = tmp[col].shift(k)

        lag_cols = [f"{col}_lag{k}" for col in base_cols for k in range(1, n_lags + 1)]
        tmp = tmp.dropna(subset=[target_col] + lag_cols)
        if tmp.empty or tmp[target_col].nunique() < 2:
            return np.nan

        X_train = tmp[lag_cols].to_numpy()
        y_train = tmp[target_col].astype(int).to_numpy()

        clf = LogisticRegression(
            C=C, class_weight=class_weight, max_iter=max_iter, solver="lbfgs"
        )
        clf.fit(X_train, y_train)

        # ---- PREDICTION FEATURES at date t: lags from est ----
        lag_values = []
        for col in base_cols:
            vals = est[col].iloc[-n_lags:]   # t-1,...,t-n
            if vals.isna().any():
                return np.nan
            lag_values.extend(vals.values)

        x_t = np.array(lag_values, dtype=float).reshape(1, -1)

        if return_proba:
            return float(clf.predict_proba(x_t)[0, 1])
        else:
            return int(clf.predict(x_t)[0])

    return fit_predict


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve



import numpy as np
import pandas as pd

def make_tabpfn_lag_cls_fit_predict_fn_fast(
    base_cols,
    target_col: str = "state",
    n_lags: int = 1,
    model_params=None,
):
    import numpy as np
    import pandas as pd
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    base_cols = list(base_cols)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_params == "2.5":
        clf = TabPFNClassifier(device=device)
    else:
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=device)

    lag_cols = [f"{col}_lag{k}" for col in base_cols for k in range(1, n_lags + 1)]

    def build_lag_frame(df: pd.DataFrame) -> pd.DataFrame:
        tmp = df[base_cols + [target_col]].copy()
        for col in base_cols:
            for k in range(1, n_lags + 1):
                tmp[f"{col}_lag{k}"] = tmp[col].shift(k)
        return tmp

    # ----- fallback single-step call (still works if batch path not used) -----
    def fit_predict(est: pd.DataFrame, row_t: pd.Series):
        if len(est) <= n_lags:
            return np.nan

        tmp = build_lag_frame(est)
        tmp = tmp.dropna(subset=lag_cols + [target_col])
        if tmp.empty or tmp[target_col].nunique() < 2:
            return np.nan

        X_train = tmp[lag_cols].to_numpy(float)
        y_train = tmp[target_col].astype(int).to_numpy()

        clf.fit(X_train, y_train)

        # build x_t from last n_lags rows of est
        lag_values = []
        for col in base_cols:
            vals = est[col].iloc[-n_lags:]
            if vals.isna().any():
                return np.nan
            lag_values.extend(vals.values)

        X_pred = np.asarray(lag_values, dtype=float).reshape(1, -1)
        return int(clf.predict(X_pred)[0])

    # ----- fast batch method attached to the function -----
    def batch_predict(df: pd.DataFrame, loop_dates: pd.DatetimeIndex, target_col: str, min_train: int, refit_every: int):
        tmp = build_lag_frame(df)

        # eligible OOS rows: in loop_dates, has y, has lags, enough past y
        in_oos = tmp.index.isin(loop_dates)
        has_y = tmp[target_col].notna()
        has_lags = tmp[lag_cols].notna().all(axis=1)
        past_y_count = tmp[target_col].notna().astype(int).cumsum().shift(1).fillna(0)
        enough_train = past_y_count >= min_train

        elig = in_oos & has_y & has_lags & enough_train
        oos_dates = tmp.index[elig]
        if len(oos_dates) == 0:
            return [], [], []

        if refit_every is None or refit_every <= 0:
            refit_every = len(oos_dates)

        trues_all, preds_all, dates_all = [], [], []

        # chunk loop: fit once per chunk, predict whole chunk at once
        y_true_full = tmp.loc[oos_dates, target_col].astype(int).to_numpy()

        for start in range(0, len(oos_dates), refit_every):
            end = min(start + refit_every, len(oos_dates))
            chunk_dates = oos_dates[start:end]

            cut_date = chunk_dates[0]
            train_tmp = tmp.loc[tmp.index < cut_date].dropna(subset=lag_cols + [target_col])

            if train_tmp.empty or train_tmp[target_col].nunique() < 2:
                continue

            X_train = train_tmp[lag_cols].to_numpy(float)
            y_train = train_tmp[target_col].astype(int).to_numpy()
            clf.fit(X_train, y_train)

            X_pred = tmp.loc[chunk_dates, lag_cols].to_numpy(float)
            y_hat = clf.predict(X_pred).astype(int)

            trues_all.extend(y_true_full[start:end].tolist())
            preds_all.extend(y_hat.tolist())
            dates_all.extend(chunk_dates.tolist())

        return trues_all, preds_all, dates_all

    fit_predict.batch_predict = batch_predict
    return fit_predict



def tabpfn_cls_oos(
    data: pd.DataFrame,
    base_cols,
    target_col: str = "state",
    start_oos: str = "1965-01-01",
    start_date: str = "1927-01-01",
    n_lags: int = 1,
    min_train: int = 120,
    quiet: bool = False,
    model_name: str = "TabPFN-CLS-lag",
    baseline_mode: str = "majority",
    model_params=None,
    refit_every: int = 10_000_000,  # default: fit once, predict all at once
):
    """
    Expanding-window OOS classification with TabPFN, but fast:
    - Fits only every `refit_every` predictions (default huge => fit once)
    - Predicts each chunk in one vectorized call
    """
    fit_fn = make_tabpfn_lag_cls_fit_predict_fn_fast(
        base_cols=base_cols,
        target_col=target_col,
        n_lags=n_lags,
        model_params=model_params,
    )

    metrics, y_true, y_pred, dates = expanding_oos_tabular_cls(
        data=data,
        target_col=target_col,
        start_oos=start_oos,
        start_date=start_date,
        min_train=min_train,
        min_history_months=None,
        quiet=quiet,
        model_name=model_name,
        model_fit_predict_fn=fit_fn,   # triggers batch fast-path
        baseline_mode=baseline_mode,
        refit_every=refit_every,
    )
    return metrics, y_true, y_pred, dates

from typing import Callable
import numpy as np
import pandas as pd

def make_moment_fit_predict_fn(
    feature_cols,
    target_col="state",
    seq_len=256,
    batch_size=64,
    epochs=1,
    lr=1e-4,
    model_id="AutonLab/MOMENT-1-small",
    tune_threshold="youden",
    use_class_weight=False,
    retrain_every=5,          # <--- NEW: retrain frequency
):
    """
    MOMENT classifier fit_predict function for expanding_oos_tabular_cls,
    with retraining every `retrain_every` steps instead of every step.
    """

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import roc_curve
    from momentfm import MOMENTPipeline
    from torch.nn.utils import clip_grad_norm_

    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]
    n_channels = len(feature_cols)

    # Choose device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ----------- Persistent state inside closure -----------
    model = None
    optimizer = None
    step_counter = 0
    last_thr = 0.5

    # ----------- Helper functions -----------
    def _make_windows(X_df, y_ser):
        X_hist = X_df.to_numpy(dtype=np.float32)
        y_hist = y_ser.to_numpy(dtype=np.int64)
        T_hist = X_hist.shape[0]
        if T_hist <= seq_len:
            return None, None
        windows, labels = [], []
        for t in range(seq_len, T_hist):
            w = X_hist[t - seq_len : t, :].T
            windows.append(w)
            labels.append(y_hist[t])
        X_train = torch.from_numpy(np.stack(windows)).float()
        y_train = torch.from_numpy(np.asarray(labels, np.int64))
        return X_train, y_train

    def _make_ctx(X_df):
        X_hist = X_df.to_numpy(dtype=np.float32)
        T_hist = X_hist.shape[0]
        if T_hist == 0:
            return None
        if T_hist >= seq_len:
            ctx = X_hist[T_hist - seq_len : T_hist, :]
        else:
            pad = np.zeros((seq_len - T_hist, n_channels), dtype=np.float32)
            ctx = np.vstack([pad, X_hist])
        return torch.from_numpy(ctx.T.reshape(1, n_channels, seq_len)).float()

    # ----------- The core fit_predict() function -----------
    def fit_predict(est, row_t):
        nonlocal model, optimizer, step_counter, last_thr
        step_counter += 1

        # Prepare data
        df = est.copy()
        X_df = df[feature_cols].astype("float32")
        y_ser = df[target_col].astype("int64")

        # Skip if insufficient data
        if len(df) <= seq_len or y_ser.nunique() < 2:
            return np.nan

        # Retrain every `retrain_every` steps or if no model yet
        if (model is None) or (step_counter % retrain_every == 0):
            X_train, y_train = _make_windows(X_df, y_ser)
            if X_train is None or len(X_train) < 10:
                return np.nan

            model = MOMENTPipeline.from_pretrained(
                model_id,
                model_kwargs={"task_name": "classification", "n_channels": n_channels, "num_class": 2},
            )
            model.init()
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
            
            # Loss
            if use_class_weight:
                n1 = int((y_train == 1).sum().item())
                n0 = int((y_train == 0).sum().item())
                w0 = (n1 + n0) / (2.0 * max(1, n0))
                w1 = (n1 + n0) / (2.0 * max(1, n1))
                class_weight = torch.tensor([w0, w1], dtype=torch.float32, device=device)
                criterion = nn.CrossEntropyLoss(weight=class_weight)
            else:
                criterion = nn.CrossEntropyLoss()

            # Train
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            model.train()
            num_updates = max(1, len(train_loader) * epochs)
            warmup = max(1, int(0.05 * num_updates))  # 5% warmup
            def lr_lambda(step):
                if step < warmup:
                    return float(step + 1) / warmup
                # cosine decay to 10% of base lr
                progress = (step - warmup) / max(1, num_updates - warmup)
                return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            for epoch in range(epochs):
                running_loss = 0.0
                batch_count = 0

                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    xb.requires_grad_(True)
                    out = model(x_enc=xb)
                    loss = criterion(out.logits, yb)
                    optimizer.zero_grad()

                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    running_loss += loss.item()
                    batch_count += 1
                avg_loss = running_loss / max(1, batch_count)
                print(f"[MOMENT retrain] Epoch {epoch + 1}/{epochs} | Avg loss = {avg_loss:.6f}")
            # Tune threshold on training data
            if tune_threshold is not None:
                model.eval()
                with torch.inference_mode():
                    preds = []
                    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
                    for xb, _ in loader:
                        xb = xb.to(device)
                        logits = model(x_enc=xb).logits
                        p1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        preds.append(p1)
                    p_tr = np.concatenate(preds)
                    y_tr = y_train.numpy()

                if tune_threshold == "majority":
                    last_thr = float(y_tr.mean())
                elif tune_threshold == "youden":
                    fpr, tpr, th = roc_curve(y_tr, p_tr)
                    j = tpr - fpr
                    last_thr = float(th[np.argmax(j)]) if len(th) else 0.5
                else:
                    last_thr = 0.5

        # Predict
        x_ctx = _make_ctx(X_df)
        if x_ctx is None or model is None:
            return np.nan
        model.eval()
        with torch.inference_mode():
            xb = x_ctx.to(device)
            logits = model(x_enc=xb).logits
            p1 = torch.softmax(logits, dim=1)[0, 1].item()

        return int(p1 >= last_thr)

    return fit_predict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler # <--- IMPORT ADDED
from torch.nn.utils import clip_grad_norm_
def make_moment_fit_predict_fn_improved(
    feature_cols,
    target_col="state",
    seq_len=64,             # Increased default, 8 is very short for Transformers
    batch_size=64,
    epochs=5,
    lr=1e-4,
    model_id="AutonLab/MOMENT-1-small",
    tune_threshold="youden",
    use_class_weight=False,
    retrain_every=5,
):
    """
    MOMENT classifier fit_predict function with Scaling and NaN handling.
    """
    from momentfm import MOMENTPipeline

    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]
    n_channels = len(feature_cols)

    # Choose device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ----------- Persistent state -----------
    model = None
    optimizer = None
    scaler = None             # <--- Persist the scaler
    step_counter = 0
    last_thr = 0.5

    # ----------- Helper: Create Windows -----------
    def _make_windows(X_scaled, y_ser):
        # X_scaled is already numpy float32 from the scaler
        y_hist = y_ser.to_numpy(dtype=np.int64)
        T_hist = X_scaled.shape[0]
        
        if T_hist <= seq_len:
            return None, None
            
        windows, labels = [], []
        for t in range(seq_len, T_hist):
            # Extract window (seq_len, n_channels)
            w = X_scaled[t - seq_len : t, :]
            # Transpose to (n_channels, seq_len) for MOMENT
            windows.append(w.T)
            labels.append(y_hist[t])
            
        X_train = torch.from_numpy(np.stack(windows)).float()
        y_train = torch.from_numpy(np.asarray(labels, np.int64))
        return X_train, y_train

    # ----------- Core Function -----------
    def fit_predict(est, row_t):
        nonlocal model, optimizer, scaler, step_counter, last_thr
        step_counter += 1

        # 1. Data Cleaning
        df = est.copy()
        # Fill NaNs (critical for rolling features at the start of history)
        df[feature_cols] = df[feature_cols].fillna(0.0) 
        
        X_df = df[feature_cols]
        y_ser = df[target_col].astype("int64")

        if len(df) <= seq_len or y_ser.nunique() < 2:
            return np.nan

        # 2. Retraining Logic
        if (model is None) or (step_counter % retrain_every == 0):
            # A. Fit Scaler on available history
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_df.values).astype(np.float32)
            
            # B. Create Tensor Data
            X_train, y_train = _make_windows(X_scaled, y_ser)
            if X_train is None or len(X_train) < 10:
                return np.nan

            # C. Initialize Model (Fresh start every retrain to avoid drift)
            # Note: This reloads from disk. If too slow, move this outside the 'if' 
            # and just use model.apply(weight_reset) or similar.
            model = MOMENTPipeline.from_pretrained(
                model_id,
                model_kwargs={
                    "task_name": "classification", 
                    "n_channels": n_channels, 
                    "num_class": 2
                },
            )
            model.init()
            model.to(device)
            
            # D. Setup Optimizer & Loss
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
            
            if use_class_weight:
                n1 = (y_train == 1).sum().item()
                n0 = (y_train == 0).sum().item()
                # Avoid division by zero
                if n0 == 0 or n1 == 0: 
                    crit = nn.CrossEntropyLoss()
                else:
                    w0 = (n1 + n0) / (2.0 * n0)
                    w1 = (n1 + n0) / (2.0 * n1)
                    cw = torch.tensor([w0, w1], dtype=torch.float32, device=device)
                    crit = nn.CrossEntropyLoss(weight=cw)
            else:
                crit = nn.CrossEntropyLoss()

            # E. Training Loop
            train_ds = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            
            model.train()
            # Learning Rate Schedule
            num_updates = max(1, len(train_loader) * epochs)
            warmup = max(1, int(0.05 * num_updates))
            
            def lr_lambda(step):
                if step < warmup: return (step + 1) / warmup
                progress = (step - warmup) / max(1, num_updates - warmup)
                return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
                
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            for _ in range(epochs):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    # MOMENT forward pass
                    out = model(x_enc=xb)
                    loss = crit(out.logits, yb)
                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

            # F. Threshold Tuning
            if tune_threshold is not None:
                model.eval()
                preds_tr = []
                # No shuffle for evaluation
                eval_loader = DataLoader(train_ds, batch_size=batch_size)
                with torch.inference_mode():
                    for xb, _ in eval_loader:
                        xb = xb.to(device)
                        logits = model(x_enc=xb).logits
                        # Probability of class 1
                        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        preds_tr.append(p)
                
                p_tr = np.concatenate(preds_tr)
                y_tr = y_train.numpy()
                
                if tune_threshold == "majority":
                    last_thr = float(y_tr.mean())
                elif tune_threshold == "youden":
                    # Need > 1 class to calc ROC
                    if len(np.unique(y_tr)) > 1:
                        fpr, tpr, th = roc_curve(y_tr, p_tr)
                        j = tpr - fpr
                        last_thr = float(th[np.argmax(j)])
                    else:
                        last_thr = 0.5
                else:
                    last_thr = 0.5

        # 3. Prediction (Inference)
        # Prepare single window
        X_curr = X_df.iloc[-seq_len:].to_numpy(dtype=np.float32) # (seq_len, n_channels)
        
        # CRITICAL: Handle case where history < seq_len (padding)
        if X_curr.shape[0] < seq_len:
            # This usually happens only at very start if min_train < seq_len
            # Pad with zeros
            pad = np.zeros((seq_len - X_curr.shape[0], n_channels), dtype=np.float32)
            X_curr = np.vstack([pad, X_curr])

        # Scale current window using the fitted scaler
        X_curr = scaler.transform(X_curr) # (seq_len, n_channels)

        # Reshape for MOMENT: (1, n_channels, seq_len)
        x_ctx = torch.from_numpy(X_curr.T).unsqueeze(0).float().to(device)

        model.eval()
        with torch.inference_mode():
            logits = model(x_enc=x_ctx).logits
            p1 = torch.softmax(logits, dim=1)[0, 1].item()

        return int(p1 >= last_thr)

    return fit_predict




def make_tree_ensemble_lag_cls_fit_predict_fn(
    base_cols,
    target_col: str = "state",
    n_lags: int = 12,
    class_weight=None,          # e.g., "balanced" for GBM/RF-like; XGB handles scale_pos_weight
    model_params=None,          # dict of XGB/GBM params
    return_proba: bool = False, # True -> return P(y=1), else hard label 0/1
):
    """
    Build lag features of base_cols and classify `target_col` at date t.
    Trains on `est` (past only). Predicts using last n_lags of each feature.
    """
    import numpy as np, pandas as pd
    base_cols = list(base_cols)
    if model_params is None:
        model_params = {}

    # Try XGBoost first, then fallback to GradientBoostingClassifier
    use_xgb = False
    try:
        from xgboost import XGBClassifier
        use_xgb = True
        default = dict(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", random_state=42,
            eval_metric="logloss"
        )
        default.update(model_params)
        def make_model(y_train):
            params = dict(default)
            # crude class-weighting for XGB via scale_pos_weight if requested
            if class_weight == "balanced":
                pos = (y_train == 1).sum()
                neg = (y_train == 0).sum()
                if pos > 0:
                    params["scale_pos_weight"] = float(max(1.0, neg / max(1, pos)))
            return XGBClassifier(**params)
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        default = dict(n_estimators=600, learning_rate=0.02, subsample=0.8, max_depth=3, random_state=42)
        default.update(model_params)
        def make_model(_y):
            # GBM has no direct class_weight; works fine as baseline
            return GradientBoostingClassifier(**default)

    def fit_predict(est: pd.DataFrame, row_t: pd.Series):
        if len(est) <= n_lags:
            return np.nan

        # Build lagged training matrix on past only
        tmp = est[base_cols + [target_col]].copy()
        for c in base_cols:
            for k in range(1, n_lags + 1):
                tmp[f"{c}_lag{k}"] = tmp[c].shift(k)

        lag_cols = [f"{c}_lag{k}" for c in base_cols for k in range(1, n_lags + 1)]
        tmp = tmp.dropna(subset=lag_cols + [target_col])
        if tmp.empty or tmp[target_col].nunique() < 2:
            return np.nan

        X_train = tmp[lag_cols].to_numpy(float)
        y_train = tmp[target_col].astype(int).to_numpy()

        clf = make_model(y_train)
        clf.fit(X_train, y_train)

        # Construct x_t from last n_lags *in est*
        lag_values = []
        for c in base_cols:
            v = est[c].iloc[-n_lags:]
            if v.isna().any() or len(v) < n_lags:
                return np.nan
            lag_values.extend(v.values)
        x_t = np.asarray(lag_values, float).reshape(1, -1)

        if return_proba and hasattr(clf, "predict_proba"):
            return float(clf.predict_proba(x_t)[0, 1])
        else:
            return int(clf.predict(x_t)[0])

    return fit_predict


import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class MLPCls(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),          # <<< GEÄNDERT (vorher 64)
            nn.ReLU(),
            nn.BatchNorm1d(128),           # <<< GEÄNDERT (neu hinzugefügt)
            nn.Dropout(0.2),               # <<< GEÄNDERT (neu hinzugefügt)
            nn.Linear(128, 64),            # <<< GEÄNDERT (vorher 32)
            nn.ReLU(),
            nn.Linear(64, 1),              # <<< GEÄNDERT (vorher 16 -> 1)
        )
    def forward(self, x):  # [B, d_in]
        return self.net(x).squeeze(-1)  # logits

def make_mlp_lag_cls_fit_predict_fn(
    base_cols,
    target_col: str = "state",
    n_lags: int = 12,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    retrain_every: int = 5,
    class_weight: bool = False,     # use balanced loss
    return_proba: bool = False,
    print_loss: bool = True,
    price_path: bool = False,
):
    base_cols = list(base_cols)
    if price_path:                   
        feature_col = base_cols[0]   
        d_in = n_lags                
    else:                            
        d_in = len(base_cols) * n_lags

    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")

    model, scaler, step = None, None, 0

    def build_lagged(df):
        tmp = df[base_cols + [target_col]].copy()
        for c in base_cols:
            for k in range(1, n_lags + 1):
                tmp[f"{c}_lag{k}"] = tmp[c].shift(k)
        lag_cols = [f"{c}_lag{k}" for c in base_cols for k in range(1, n_lags + 1)]
        tmp = tmp.dropna(subset=lag_cols + [target_col])
        return tmp, lag_cols

    def fit_predict(est: pd.DataFrame, row_t: pd.Series):
        nonlocal model, scaler, step
        step += 1
        if price_path:                                                   # <<< NEW
            if len(est) <= n_lags or est[target_col].nunique() < 2:      # <<< NEW
                return np.nan                                            # <<< NEW

            series = est[feature_col].astype("float32").to_numpy()       # <<< NEW
            labels = est[target_col].astype("float32").to_numpy()        # <<< NEW

            # (re)train periodically
            if (model is None) or (step % retrain_every == 0):           # <<< NEW
                X_list, y_list = [], []                                  # <<< NEW

                for i in range(n_lags, len(series)):                     # <<< NEW
                    w = series[i - n_lags : i]                           # <<< NEW
                    if np.isnan(w).any():                                # <<< NEW
                        continue                                         # <<< NEW
                    X_list.append(np.cumsum(w))                          # <<< NEW: PRICE PATH window
                    y_list.append(labels[i])                             # <<< NEW

                if len(X_list) < 20 or len(set(y_list)) < 2:             # <<< NEW
                    return np.nan                                        # <<< NEW

                X_train = np.stack(X_list).astype("float32")             # <<< NEW
                y_train = np.array(y_list).astype("float32")             # <<< NEW

                scaler = StandardScaler().fit(X_train)                   # <<< NEW
                Xs = scaler.transform(X_train).astype("float32")         # <<< NEW

                ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(y_train))  # <<< NEW
                loader = DataLoader(ds, batch_size=batch_size, shuffle=True)        # <<< NEW

                model = MLPCls(d_in=d_in).to(device)                     # <<< NEW
                opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # <<< NEW

                if class_weight:                                         # <<< NEW
                    pos = max(1.0, float((y_train == 1).sum()))          # <<< NEW
                    neg = max(1.0, float((y_train == 0).sum()))          # <<< NEW
                    w_pos = (pos + neg) / (2.0 * pos)                    # <<< NEW
                    w_neg = (pos + neg) / (2.0 * neg)                    # <<< NEW
                    crit = nn.BCEWithLogitsLoss(                         # <<< NEW
                        pos_weight=torch.tensor([w_pos / w_neg], device=device)
                    )
                else:                                                    # <<< NEW
                    crit = nn.BCEWithLogitsLoss()                        # <<< NEW

                model.train()                                            # <<< NEW
                for ep in range(epochs):                                 # <<< NEW
                    run, nb = 0.0, 0                                     # <<< NEW
                    for xb, yb in loader:                                # <<< NEW
                        xb, yb = xb.to(device), yb.to(device)            # <<< NEW
                        logit = model(xb)                                # <<< NEW
                        loss = crit(logit, yb)                           # <<< NEW
                        opt.zero_grad()                                  # <<< NEW
                        loss.backward()                                  # <<< NEW
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)# <<< NEW
                        opt.step()                                       # <<< NEW
                        run += loss.item()                               # <<< NEW
                        nb += 1                                          # <<< NEW
                    if print_loss:                                       # <<< NEW
                        print(f"[MLPCls (price_path) retrain] epoch {ep+1}/{epochs} | loss={run/max(1,nb):.6f}")  # <<< NEW

            # ---- Prediction in price-path mode ----                 # <<< NEW
            window = series[-n_lags:]                                   # <<< NEW
            if np.isnan(window).any() or len(window) < n_lags:          # <<< NEW
                return np.nan                                           # <<< NEW

            x_t = np.cumsum(window).reshape(1, -1)                      # <<< NEW
            x_tn = scaler.transform(x_t).astype("float32")              # <<< NEW
            x_tn = torch.from_numpy(x_tn).to(device)                    # <<< NEW

            model.eval()                                                # <<< NEW
            with torch.inference_mode():                                # <<< NEW
                p1 = torch.sigmoid(model(x_tn)).item()                  # <<< NEW

            return float(p1) if return_proba else int(p1 >= 0.5) 
        tmp, lag_cols = build_lagged(est)
        if tmp.empty or tmp[target_col].nunique() < 2:
            return np.nan

        X = tmp[lag_cols].to_numpy("float32")
        y = tmp[target_col].astype("float32").to_numpy()  # 0/1

        # (re)train periodically
        if (model is None) or (step % retrain_every == 0):
            scaler = StandardScaler().fit(X)
            Xs = scaler.transform(X).astype("float32")

            ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(y))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            model = MLPCls(d_in=d_in).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            if class_weight:
                pos = max(1.0, float((y == 1).sum()))
                neg = max(1.0, float((y == 0).sum()))
                w_pos = (pos + neg) / (2.0 * pos)
                w_neg = (pos + neg) / (2.0 * neg)
                crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w_pos / w_neg], device=device))
            else:
                crit = nn.BCEWithLogitsLoss()

            model.train()
            for ep in range(epochs):
                run, nb = 0.0, 0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logit = model(xb)
                    loss = crit(logit, yb)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    run += loss.item(); nb += 1
                if print_loss:
                    print(f"[MLPCls retrain] epoch {ep+1}/{epochs} | loss={run/max(1,nb):.6f}")

        # build x_t from last n_lags of each feature
        vals = []
        for c in base_cols:
            v = est[c].iloc[-n_lags:]
            if v.isna().any() or len(v) < n_lags: return np.nan
            vals.extend(v.values)
        x_t = np.asarray(vals, "float32").reshape(1, -1)

        x_tn = torch.from_numpy(scaler.transform(x_t).astype("float32")).to(device)
        model.eval()
        with torch.inference_mode():
            p1 = torch.sigmoid(model(x_tn)).item()

        return float(p1) if return_proba else int(p1 >= 0.5)

    return fit_predict




import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification
from chronos import ChronosPipeline

def make_chronos_t5_cls_fit_predict_fn(
    feature_col: str = "equity_premium",
    target_col: str = "state_true",
    seq_len: int = 64,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    retrain_every: int = 25,      # Update weights every N steps
    model_id: str = "amazon/chronos-t5-tiny",
    return_proba: bool = True,
    print_loss: bool = True,
):
    """
    Chronos-T5 fit_predict function for expanding window backtesting.
    
    CRITICAL: Converts Returns -> Cumulative Sum (Price Path) so T5 
    can detect volatility regimes via geometric shapes.
    """

    # ---- 1. Setup Device & Constants ----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ---- 2. Persistent State (Closure) ----
    # We load these lazily on the first call to avoid overhead if unused
    model = None
    pipeline = None 
    tokenizer = None
    optimizer = None
    step_counter = 0

    # ---- 3. Helper: Build Windows & Tokenize ----
    def prepare_data(df, is_training=True):
        """
        Converts dataframe -> Tokenized Tensors using 'Price Path' logic.
        Returns: (input_ids, attention_mask, labels)
        """
        # Extract arrays
        series = df[feature_col].to_numpy(dtype=np.float32)
        if is_training:
            labels_all = df[target_col].to_numpy(dtype=np.int64)
        
        T = len(series)
        
        # If predicting (inference), we just need the last window
        if not is_training:
            if T < seq_len: return None
            # Take last window
            window = series[-seq_len:]
            # TRANSFORMATION: Cumsum to get Price Path
            price_path = np.cumsum(window)
            # To Tensor [1, 1, seq_len] -> Tokenizer expects [batch, time]
            ts_tensor = torch.tensor(price_path).float().unsqueeze(0)
            return ts_tensor

        # If training, build all sliding windows
        if T <= seq_len: return None, None, None
        
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        # Optimization: If history is huge, maybe only train on recent history?
        # For now, we train on full expanding history.
        # We process in chunks or list comprehension. 
        
        # Create raw windows first
        raw_windows = []
        targets = []
        
        # Stride 1
        for t in range(seq_len, T):
            w = series[t-seq_len : t]
            # TRANSFORMATION: Cumsum
            raw_windows.append(np.cumsum(w))
            targets.append(labels_all[t]) # Label at time t (or t-1 depending on alignment)
            
        if not raw_windows: return None, None, None

        # Batch tokenization is faster than loop
        # Stack raw windows: [N, seq_len]
        batch_tensor = torch.tensor(np.stack(raw_windows)).float()
        
        # Tokenize on CPU to avoid MPS/CUDA IPC overhead during data loading
        # Note: usage of context_input_transform
        ids, mask, _ = tokenizer.context_input_transform(batch_tensor)
        
        return ids, mask, torch.tensor(targets, dtype=torch.long)

    # ---- 4. The fit_predict function ----
    def fit_predict(est: pd.DataFrame, row_t: pd.Series):
        nonlocal model, pipeline, tokenizer, optimizer, step_counter
        step_counter += 1
        
        # A. Initialize Model (First Run Only)
        if model is None:
            print(f"[Chronos] Loading {model_id} on {device}...")
            # Pipeline for Tokenizer
            pipeline = ChronosPipeline.from_pretrained(
                model_id, device_map=device, torch_dtype=torch.float32
            )
            tokenizer = pipeline.tokenizer
            
            # Classification Model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, num_labels=2, torch_dtype=torch.float32
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # B. Retrain Logic (Periodic)
        if step_counter % retrain_every == 0 or step_counter == 1:
            # Prepare Training Data
            ids, mask, y_train = prepare_data(est, is_training=True)
            
            if ids is not None and len(ids) > 10:
                ds = TensorDataset(ids, mask, y_train)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
                
                model.train()
                epoch_loss = 0.0
                
                # Training Loop
                for i in range(epochs):
                    print(f"[Chronos Retrain] Step {step_counter} | Epoch {i+1}/{epochs}")
                    run_loss = 0.0
                    batches = 0
                    l = 1
                    for b_ids, b_mask, b_y in loader:
                        l += 1
                        b_ids, b_mask, b_y = b_ids.to(device), b_mask.to(device), b_y.to(device)
                        
                        optimizer.zero_grad()
                        out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_y)
                        loss = out.loss
                        loss.backward()
                        optimizer.step()
                        
                        run_loss += loss.item()
                        batches += 1
                        print(f"[Chronos Retrain] Batch {l} | Loss: {loss.item():.4f}")
                    epoch_loss = run_loss / max(1, batches)
                
                if print_loss:
                    print(f"[Chronos Retrain] Step {step_counter} | Obs: {len(ids)} | Loss: {epoch_loss:.4f}")

        # C. Prediction Logic (Current Step)
        # 1. Prepare input
        ts_tensor = prepare_data(est, is_training=False)
        if ts_tensor is None:
            return np.nan
            
        # 2. Tokenize (Single sample)
        # context_input_transform expects [batch, time]
        ids, mask, _ = tokenizer.context_input_transform(ts_tensor)
        
        # 3. Move to device
        ids = ids.to(device)
        mask = mask.to(device)
        
        # 4. Inference
        model.eval()
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=mask)
            logits = out.logits # [1, 2]
            probs = torch.softmax(logits, dim=1) # [1, 2]
            prob_bull = probs[0, 1].item()
            
        return float(prob_bull) if return_proba else int(prob_bull >= 0.5)

    return fit_predict