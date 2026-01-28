import sys
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
sys.path.insert(0, os.path.abspath('../'))
from source.regression.modelling_utils import expanding_oos_tabular_cls


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


def make_tabpfn_lag_cls_fit_predict_fn(
    base_cols,
    target_col: str = "state",    # 0/1 (Bull/Bear)
    n_lags: int = 1,
    min_train: int = 120,
    model_params=None,            # e.g. "2.5" to mimic your reg code
):
    """
    Create a fit_predict(est, row_t) function for TabPFN *classification*.

    - Uses lagged values of multiple columns in base_cols as features.
    - Predicts `target_col` (0/1) at time t from lags at t-1,...,t-n_lags.
    - Training uses only past data = `est` (no look-ahead).
    """
    import torch
    try:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
    except Exception as e:
        raise RuntimeError("TabPFN not installed. Please `pip install tabpfn`.") from e

    base_cols = list(base_cols)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def fit_predict(est: pd.DataFrame, row_t: pd.Series):
        # Need enough history to build lags
        if len(est) <= n_lags:
            return np.nan

        # ---------- TRAIN ON PAST ONLY ----------
        # Build lag features on 'est' (strictly past)
        tmp = est[list(base_cols) + [target_col]].copy()
        for col in base_cols:
            for k in range(1, n_lags + 1):
                tmp[f"{col}_lag{k}"] = tmp[col].shift(k)

        lag_cols = [f"{col}_lag{k}" for col in base_cols for k in range(1, n_lags + 1)]
        tmp = tmp.dropna(subset=lag_cols + [target_col])

        # Must have data and both classes to fit a classifier
        if tmp.empty or tmp[target_col].nunique() < 2:
            return np.nan

        X_train = tmp[lag_cols].to_numpy(float)
        y_train = tmp[target_col].astype(int).to_numpy()

        # Choose TabPFN version
        if model_params == "2.5":
            clf = TabPFNClassifier(device=device)
        else:
            clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=device)

        clf.fit(X_train, y_train)

        # ---------- PREDICT AT date_t ----------
        # Build the feature vector for t from the last n_lags values in 'est'
        if len(est) < n_lags:
            return np.nan

        lag_values = []
        for col in base_cols:
            vals = est[col].iloc[-n_lags:]  # t-1, t-2, ..., t-n
            if vals.isna().any():
                return np.nan
            lag_values.extend(vals.values)

        X_pred = np.asarray(lag_values, dtype=float).reshape(1, -1)

        y_hat = clf.predict(X_pred)[0]
        return int(y_hat)

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
    model_params=None,           # e.g. "2.5"
):
    """
    Expanding-window 1-step-ahead OOS CLASSIFICATION with TabPFN.

    - Uses lagged values of `base_cols` as features.
    - Predicts binary `target_col` (e.g. Bull/Bear state).
    - Re-fits TabPFNClassifier from scratch at each OOS step, like your
      logistic baseline.

    Returns
    -------
    metrics : dict
        Output from evaluate_oos_classification.
    y_true : np.ndarray
        True labels at OOS dates.
    y_pred : np.ndarray
        Predicted labels at OOS dates.
    dates : pd.DatetimeIndex
        OOS evaluation dates.
    """
    # build the fit_predict function for TabPFN
    fit_fn = make_tabpfn_lag_cls_fit_predict_fn(
        base_cols=base_cols,
        target_col=target_col,
        n_lags=n_lags,
        min_train=min_train,
        model_params=model_params,
    )

    # plug into your generic classification OOS driver
    metrics, y_true, y_pred, dates = expanding_oos_tabular_cls(
        data=data,
        target_col=target_col,
        start_oos=start_oos,
        start_date=start_date,
        min_train=min_train,
        min_history_months=None,
        quiet=quiet,
        model_name=model_name,
        model_fit_predict_fn=fit_fn,
        baseline_mode=baseline_mode,
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
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
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
):
    base_cols = list(base_cols)
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


class LSTMCls(nn.Module):
    def __init__(self, n_feat, hidden=64, num_layers=1, bidir=False, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=bidir,
            dropout=(dropout if num_layers > 1 else 0.0)
        )
        d = hidden * (2 if bidir else 1)
        self.head = nn.Linear(d, 1)
    def forward(self, x):  # [B, L, F]
        h, _ = self.lstm(x)
        z = h[:, -1, :]
        return self.head(z).squeeze(-1)  # logits

def make_lstm_seq_cls_fit_predict_fn(
    feature_cols,
    target_col: str = "state",
    seq_len: int = 24,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    retrain_every: int = 25,
    hidden: int = 64,
    num_layers: int = 1,
    bidir: bool = False,
    dropout: float = 0.0,
    scale: bool = True,
    class_weight: bool = False,
    return_proba: bool = False,
    print_loss: bool = True,
):
    import numpy as np, pandas as pd, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    feature_cols = list(feature_cols)
    n_feat = len(feature_cols)

    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    model, scaler, step = None, None, 0

    def make_windows(X_df, y_ser):
        X = X_df.to_numpy("float32")
        y = y_ser.to_numpy("float32")
        T = len(X)
        if T <= seq_len: return None, None
        Xs, ys = [], []
        for t in range(seq_len, T):
            Xs.append(X[t-seq_len:t, :])
            ys.append(y[t])
        return np.stack(Xs).astype("float32"), np.asarray(ys, "float32")

    def fit_predict(est: pd.DataFrame, row_t: pd.Series):
        nonlocal model, scaler, step
        step += 1
        df = est.copy()
        if any(c not in df.columns for c in feature_cols+[target_col]): return np.nan
        if len(df) <= seq_len or df[target_col].nunique() < 2: return np.nan

        X_df = df[feature_cols].astype("float32")
        y_ser = df[target_col].astype("float32")

        # scale history
        if scale:
            scaler = StandardScaler().fit(X_df.values)
            Xn = scaler.transform(X_df.values).astype("float32")
            Xn_df = pd.DataFrame(Xn, index=X_df.index, columns=X_df.columns)
        else:
            Xn_df = X_df

        # (re)train
        if (model is None) or (step % retrain_every == 0):
            X_train, y_train = make_windows(Xn_df, y_ser)
            if X_train is None or len(X_train) < 10: return np.nan

            ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            model = LSTMCls(n_feat=n_feat, hidden=hidden, num_layers=num_layers,
                            bidir=bidir, dropout=dropout).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            if class_weight:
                pos = max(1.0, float((y_train == 1).sum()))
                neg = max(1.0, float((y_train == 0).sum()))
                w = torch.tensor([neg / pos], dtype=torch.float32, device=device)  # pos_weight
                crit = nn.BCEWithLogitsLoss(pos_weight=w)
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
                    print(f"[LSTMCls retrain] epoch {ep+1}/{epochs} | loss={run/max(1,nb):.6f}")

        # context window for t from last seq_len rows (past only)
        X_hist = df[feature_cols].to_numpy("float32")
        if len(X_hist) < seq_len: return np.nan
        if scale and scaler is not None:
            X_hist = scaler.transform(X_hist).astype("float32")
        ctx = X_hist[-seq_len:, :].reshape(1, seq_len, n_feat)
        xb = torch.from_numpy(ctx).to(device)

        model.eval()
        with torch.inference_mode():
            p1 = torch.sigmoid(model(xb)).item()

        return float(p1) if return_proba else int(p1 >= 0.5)

    return fit_predict
import math, torch, torch.nn as nn

class PosEnc(nn.Module):
    def __init__(self, d_model=128, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)

class TransformerCls(nn.Module):
    def __init__(self, n_feat, d_model=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_feat, d_model)
        self.pos = PosEnc(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x):   # [B, L, F]
        h = self.inp(x)
        h = self.pos(h)
        h = self.enc(h)
        z = h[:, -1, :]
        return self.head(z).squeeze(-1)  # logits

def make_transformer_seq_cls_fit_predict_fn(
    feature_cols,
    target_col: str = "state",
    seq_len: int = 24,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    retrain_every: int = 25,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    scale: bool = True,
    class_weight: bool = False,
    return_proba: bool = False,
    print_loss: bool = True,
):
    import numpy as np, pandas as pd, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    feature_cols = list(feature_cols)
    n_feat = len(feature_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    model, scaler, step = None, None, 0

    def make_windows(X_df, y_ser):
        X = X_df.to_numpy("float32")
        y = y_ser.to_numpy("float32")
        T = len(X)
        if T <= seq_len: return None, None
        Xs, ys = [], []
        for t in range(seq_len, T):
            Xs.append(X[t-seq_len:t, :])
            ys.append(y[t])
        return np.stack(Xs).astype("float32"), np.asarray(ys, "float32")

    def fit_predict(est: pd.DataFrame, row_t: pd.Series):
        nonlocal model, scaler, step
        step += 1
        df = est.copy()
        if any(c not in df.columns for c in feature_cols+[target_col]): return np.nan
        if len(df) <= seq_len or df[target_col].nunique() < 2: return np.nan

        X_df = df[feature_cols].astype("float32")
        y_ser = df[target_col].astype("float32")

        if scale:
            scaler = StandardScaler().fit(X_df.values)
            Xn = scaler.transform(X_df.values).astype("float32")
            Xn_df = pd.DataFrame(Xn, index=X_df.index, columns=X_df.columns)
        else:
            Xn_df = X_df

        if (model is None) or (step % retrain_every == 0):
            X_train, y_train = make_windows(Xn_df, y_ser)
            if X_train is None or len(X_train) < 10: return np.nan

            ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            model = TransformerCls(n_feat=n_feat, d_model=d_model, nhead=nhead,
                                   num_layers=num_layers, dim_ff=dim_feedforward,
                                   dropout=dropout).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            if class_weight:
                pos = max(1.0, float((y_train == 1).sum()))
                neg = max(1.0, float((y_train == 0).sum()))
                w = torch.tensor([neg / pos], dtype=torch.float32, device=device)
                crit = nn.BCEWithLogitsLoss(pos_weight=w)
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
                    print(f"[TransformerCls retrain] epoch {ep+1}/{epochs} | loss={run/max(1,nb):.6f}")

        # one-step-ahead prediction: last seq_len rows of history (no look-ahead)
        X_hist = df[feature_cols].to_numpy("float32")
        if len(X_hist) < seq_len: return np.nan
        if scale and scaler is not None:
            X_hist = scaler.transform(X_hist).astype("float32")
        ctx = X_hist[-seq_len:, :].reshape(1, seq_len, n_feat)
        xb = torch.from_numpy(ctx).to(device)

        model.eval()
        with torch.inference_mode():
            p1 = torch.sigmoid(model(xb)).item()
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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from chronos import BaseChronosPipeline
from tqdm.auto import tqdm

# --- 1. Custom Classifier Wrapper ---
# This wraps the Bolt backbone so we can train a head on top of it
class ChronosBoltClassifier(nn.Module):
    def __init__(self, backbone, d_model=512): # Bolt Small usually has d_model=512 (Base=768)
        super().__init__()
        self.backbone = backbone
        # We only need the encoder part of Bolt for classification
        self.encoder = backbone.encoder if hasattr(backbone, "encoder") else backbone.model.encoder
        
        # Simple Classification Head
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output 1 logit (Binary)
        )

    def forward(self, input_ids, attention_mask):
        # 1. Pass through Bolt Encoder
        # Bolt/T5 encoders return a specific object, we want 'last_hidden_state'
        encoder_outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden_state = encoder_outputs.last_hidden_state # [Batch, Seq_Len, d_model]
        
        # 2. Mean Pooling (Average all tokens to get one vector per window)
        # Mask out padding tokens so they don't drag down the average
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # 3. Classify
        logits = self.head(pooled)
        return logits


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from chronos import BaseChronosPipeline

# ==========================================
# 1. SIMPLE MLP CLASSIFIER
# ==========================================
class AugmentedMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output Logits
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. THE FACTORY FUNCTION
# ==========================================
def make_chronos_forecast_mlp_cls_fn(
    feature_col: str = "equity_premium",
    target_col: str = "state_true",
    seq_len: int = 64,          # History length
    pred_len: int = 12,         # How far to ask Chronos to look ahead
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    retrain_every: int = 50,    # Retrain MLP every N steps
    model_id: str = "amazon/chronos-bolt-small",
    return_proba: bool = True,
    train_window_limit: int = 300 # Only train MLP on last N samples to save time
):
    # ---- Device Setup ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps"

    # ---- Persistent State ----
    # We cache the generated features (History + Forecast) so we don't 
    # run Chronos 2000 times every loop.
    state = {
        "pipe": None,
        "mlp": None,
        "optimizer": None,
        "loss_fn": None,
        "feature_cache": {}, # Maps timestamp -> numpy array of (seq_len + pred_len)
        "step": 0
    }

    # ---- Helper: Feature Generator (The Core Logic) ----
    def get_augmented_features(series_window, timestamp):
        """
        Runs Chronos Bolt to get a forecast, then combines 
        Past + Forecast into one feature vector.
        """
        # 1. Check Cache
        if timestamp in state["feature_cache"]:
            return state["feature_cache"][timestamp]

        # 2. Prepare Input (Cumulative Sum Price Path)
        # Chronos needs 'price' shapes, not returns noise.
        price_path = np.cumsum(series_window)
        
        # 3. Run Chronos (Standard Public API)
        # We pass a list of tensors.
        with torch.inference_mode():
            # context shape: [1, seq_len]
            ctx_tensor = torch.tensor(price_path, device=device).float()
            
            # predict_quantiles returns (forecast, quantiles)
            # We just want the median (0.5) forecast
            forecast_tensor = state["pipe"].predict_quantiles(
                context=ctx_tensor.unsqueeze(0), 
                prediction_length=pred_len,
                quantile_levels=[0.5],
                limit_prediction_length=False 
            )[0] # [1, pred_len, 1]
            
            # Extract the median forecast
            forecast_values = forecast_tensor[0, :, 0].cpu().numpy() # [pred_len]
            
        # 4. Normalize & Combine
        # We normalize the Combined vector so the MLP trains easily.
        # We zero-center based on the *history start*.
        combined = np.concatenate([price_path, forecast_values])
        
        # Simple MinMax/Standard scaling relative to the window itself
        # (This makes the pattern invariant to absolute price level)
        combined = combined - combined[0] # Start at 0
        scale = np.max(np.abs(combined)) + 1e-6
        combined = combined / scale
        
        # 5. Cache and Return
        state["feature_cache"][timestamp] = combined.astype(np.float32)
        return combined

    # ---- Main Fit/Predict ----
    def fit_predict(est: pd.DataFrame, row_t: pd.Series):
        state["step"] += 1
        current_time = row_t.name # Timestamp index
        
        # 1. Initialize Pipeline (Once)
        if state["pipe"] is None:
            print(f"[Chronos-MLP] Loading {model_id} on {device}...")
            state["pipe"] = BaseChronosPipeline.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=torch.float32
            )
            # Input dim = History + Forecast
            state["mlp"] = AugmentedMLP(input_dim=seq_len + pred_len).to(device)
            state["optimizer"] = torch.optim.AdamW(state["mlp"].parameters(), lr=lr)
            state["loss_fn"] = nn.BCEWithLogitsLoss()

        # 2. Retrain MLP (Periodically)
        if state["step"] % retrain_every == 0 or state["step"] == 1:
            # A. Build Training Set
            # We iterate backwards from current time to build a training batch
            X_list = []
            y_list = []
            
            # Get feature/target arrays
            feats = est[feature_col].values
            targs = est[target_col].values
            times = est.index
            
            # Limit training to recent history to keep it fast
            # (Generating forecasts is slower than just loading numbers)
            indices = range(len(feats) - seq_len, seq_len, -1) # Backwards
            count = 0
            
            for i in indices:
                if count >= train_window_limit: break
                
                # Window: [i - seq_len : i]
                # Target: label at time i
                w = feats[i-seq_len : i]
                ts = times[i]
                y = targs[i]
                
                # Generate Features (Uses Cache if available)
                x_vec = get_augmented_features(w, ts)
                
                X_list.append(x_vec)
                y_list.append(y)
                count += 1
            
            # B. Train MLP
            if len(X_list) > 10:
                X_train = torch.tensor(np.stack(X_list)).float().to(device)
                y_train = torch.tensor(np.array(y_list)).float().unsqueeze(1).to(device)
                
                ds = TensorDataset(X_train, y_train)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
                
                state["mlp"].train()
                losses = []
                for _ in range(epochs):
                    for bx, by in loader:
                        state["optimizer"].zero_grad()
                        logits = state["mlp"](bx)
                        loss = state["loss_fn"](logits, by)
                        loss.backward()
                        state["optimizer"].step()
                        losses.append(loss.item())
                
                # print(f"[Step {state['step']}] MLP Loss: {np.mean(losses):.4f}")

        # 3. Inference (Current Step)
        # Get window
        cur_series = est[feature_col].to_numpy()
        if len(cur_series) < seq_len: return np.nan
        
        window = cur_series[-seq_len:]
        
        # Generate features (Forecast)
        # We use current_time as key
        x_vec = get_augmented_features(window, current_time)
        
        # MLP Prediction
        state["mlp"].eval()
        with torch.inference_mode():
            x_tensor = torch.tensor(x_vec).float().unsqueeze(0).to(device)
            logits = state["mlp"](x_tensor)
            prob = torch.sigmoid(logits).item()

        return float(prob) if return_proba else int(prob >= 0.5)

    return fit_predict




# Todo chronos and timesfm with classification head
#FT‑Transforme
#Lightwood
#xtab