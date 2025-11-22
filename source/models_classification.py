import sys
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
sys.path.insert(0, os.path.abspath('../'))
from source.modelling_utils import expanding_oos_tabular_cls


def make_logit_multifeature_lag_fit_predict_fn(
    base_cols: list[str] = ["SXXT", "SPX", "NKY", "SPTR", "EUR003M",
             "FEDL01", "GC1", "V2X", "MOVE", "VIX",
             "USYC2Y10", "VXJ"],                # multiple continuous input variables
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




# def moment_cls_oos(
#     data: pd.DataFrame,
#     feature_cols=("equity_premium",),   # one or many continuous features
#     target_col="equity_premium_c",      # 0/1 target
#     start_oos="2007-01-01",
#     min_context=120,
#     seq_len=256,
#     batch_size=64,
#     epochs_per_step=1,
#     lr=1e-4,
#     quiet=False,
#     model_id="AutonLab/MOMENT-1-small",
#     tune_threshold="youden",            # 'youden' | 'majority' | None
#     use_class_weight=False,
#     train_first_step_only=False,
#     max_steps=None,
# ):
#     """
#     Expanding-window one-step-ahead OOS classification with MOMENT.

#     - Input: past values of one or more continuous features (feature_cols)
#     - Target: binary label (target_col, e.g., Bull/Bear) at time t
#     - Model: MOMENT transformer (sequence classifier)

#     Parameters
#     ----------
#     data : DataFrame
#         Must contain columns in `feature_cols` and `target_col`.
#         Index should be a DatetimeIndex or convertible to datetime.
#     feature_cols : sequence of str
#         Names of continuous input features. Each becomes one channel.
#     target_col : str
#         Name of binary target column (0/1).
#     start_oos : str or Timestamp
#         First date to start OOS evaluation.
#     min_context : int
#         Minimum number of past observations before first OOS prediction.
#     seq_len : int
#         Length of sequence window fed into MOMENT.
#     batch_size : int
#         Training mini-batch size.
#     epochs_per_step : int
#         Number of epochs per (re)training step.
#     lr : float
#         Learning rate for Adam.
#     tune_threshold : {'youden', 'majority', None}
#         How to choose the classification threshold:
#         - 'youden': maximize TPR - FPR on training windows
#         - 'majority': fraction of class 1 in training windows
#         - None: fixed 0.5
#     use_class_weight : bool
#         If True, use class weights in CrossEntropyLoss.
#     train_first_step_only : bool
#         If True, train only for the first OOS origin and reuse weights.
#     max_steps : int or None
#         Optional cap on number of OOS points (for speed).

#     Returns
#     -------
#     acc : float
#         OOS accuracy (using time-varying thresholds).
#     brier : float
#         Brier score on predicted probabilities.
#     y_true : np.ndarray
#         True labels at OOS dates.
#     y_prob : np.ndarray
#         Predicted probabilities P(y=1) at OOS dates.
#     thr_arr : np.ndarray
#         Threshold used at each OOS date.
#     dates : DatetimeIndex
#         OOS evaluation dates.
#     """
#     # --------------------
#     # 0. Device
#     # --------------------#
#     try:
#         from momentfm import MOMENTPipeline
#     except Exception as e:
#             raise RuntimeError("TabPFN not installed. Please `pip install tabpfn`.") from e

#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#     if not quiet:
#         print(f"[MOMENT-CLS] device={device}")

#     # --------------------
#     # 1. Data prep
#     # --------------------
#     if isinstance(feature_cols, str):
#         feature_cols = [feature_cols]
#     else:
#         feature_cols = list(feature_cols)

#     # ensure datetime index
#     df = data.copy()
#     if not isinstance(df.index, pd.DatetimeIndex):
#         df.index = pd.to_datetime(df.index)

#     # keep only needed columns, monthly frequency
#     cols_needed = feature_cols + [target_col]
#     df.index = pd.to_datetime(df.timestamp)

#     X_df = df[feature_cols].astype("float32")  # (T, F)
#     y_cls = df[target_col].astype("int64")     # (T,)

#     n_channels = len(feature_cols)

#     start_oos = pd.Timestamp(start_oos)
#     test_idx = y_cls.index[y_cls.index >= start_oos]

#     # --------------------
#     # 2. Model init
#     # --------------------
#     model = MOMENTPipeline.from_pretrained(
#         model_id,
#         model_kwargs={
#             "task_name": "classification",
#             "n_channels": n_channels,
#             "num_class": 2,
#         },
#     )
#     model.init()
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     # --------------------
#     # 3. Helpers
#     # --------------------
#     def make_windows(upto_pos: int):
#         """
#         Build training windows from history up to (but not including) upto_pos.

#         Returns
#         -------
#         X_train : torch.Tensor, shape (N, n_channels, seq_len)
#         y_train : torch.Tensor, shape (N,)
#         """
#         X_hist = X_df.iloc[:upto_pos].to_numpy(dtype=np.float32)  # (T_hist, F)
#         y_hist = y_cls.iloc[:upto_pos].to_numpy(dtype=np.int64)

#         T_hist = X_hist.shape[0]
#         if T_hist <= seq_len:
#             return None, None

#         windows, labels = [], []
#         for t in range(seq_len, T_hist):
#             # slice window for all features: (seq_len, F), then transpose -> (F, seq_len)
#             w = X_hist[t - seq_len : t, :].T
#             windows.append(w)
#             labels.append(y_hist[t])

#         if not windows:
#             return None, None

#         X_train = torch.from_numpy(np.stack(windows)).float()  # (N, F, L)
#         y_train = torch.from_numpy(np.asarray(labels, dtype=np.int64))
#         return X_train, y_train

#     def make_ctx(upto_pos: int):
#         """
#         Build single context window for prediction at position upto_pos.

#         Returns
#         -------
#         x_ctx : torch.Tensor, shape (1, n_channels, seq_len)
#         """
#         X_hist = X_df.iloc[:upto_pos].to_numpy(dtype=np.float32)  # (T_hist, F)
#         T_hist = X_hist.shape[0]

#         if T_hist >= seq_len:
#             ctx = X_hist[T_hist - seq_len : T_hist, :]  # (seq_len, F)
#         else:
#             pad = np.zeros((seq_len - T_hist, n_channels), dtype=np.float32)
#             ctx = np.vstack([pad, X_hist])             # (seq_len, F)

#         ctx = ctx.T.reshape(1, n_channels, seq_len)    # (1, F, L)
#         return torch.from_numpy(ctx).float()

#     def train_on_history(X_train: torch.Tensor, y_train: torch.Tensor, epochs: int):
#         if use_class_weight:
#             n1 = int((y_train == 1).sum().item())
#             n0 = int((y_train == 0).sum().item())
#             w0 = (n1 + n0) / (2.0 * max(1, n0))
#             w1 = (n1 + n0) / (2.0 * max(1, n1))
#             class_weight = torch.tensor([w0, w1], dtype=torch.float32, device=device)
#             criterion = nn.CrossEntropyLoss(weight=class_weight)
#         else:
#             criterion = nn.CrossEntropyLoss()

#         train_loader = DataLoader(
#             TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
#         )

#         model.train()
#         for _ in range(epochs):
#             for xb, yb in train_loader:
#                 xb, yb = xb.to(device), yb.to(device)
#                 xb.requires_grad_(True)
#                 out = model(x_enc=xb)
#                 loss = criterion(out.logits, yb)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#     def predict_proba(x_ctx: torch.Tensor) -> float:
#         """
#         Predict P(y=1) for a single context window.
#         x_ctx: shape (1, n_channels, seq_len)
#         """
#         model.eval()
#         with torch.inference_mode():
#             xb = x_ctx.to(device)
#             logits = model(x_enc=xb).logits
#             p1 = torch.softmax(logits, dim=1)[0, 1].item()
#         return float(p1)

#     # --------------------
#     # 4. Expanding OOS loop
#     # --------------------
#     probs, trues, dates, thr_list = [], [], [], []
#     has_trained_once = False
#     step_count = 0

#     for date_t in test_idx:
#         step_count += 1
#         if max_steps is not None and step_count > max_steps:
#             break
#         if not quiet:
#             print(date_t)

#         pos = y_cls.index.get_loc(date_t)
#         # ensure enough past data & room for seq_len
#         if pos < max(min_context, seq_len + 1):
#             continue

#         # --- build training windows from past only ---
#         X_train, y_train = make_windows(upto_pos=pos)
#         if X_train is None or len(X_train) < 10 or len(torch.unique(y_train)) < 2:
#             continue

#         # --- training (once or every step) ---
#         if (not train_first_step_only) or (train_first_step_only and not has_trained_once):
#             train_on_history(X_train, y_train, epochs_per_step)
#             has_trained_once = True

#         # --- threshold tuning on training windows ---
#         thr = 0.5
#         if tune_threshold is not None:
#             model.eval()
#             with torch.inference_mode():
#                 p_tr_list = []
#                 loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
#                 for xb, yb in loader:
#                     xb = xb.to(device)
#                     logits = model(x_enc=xb).logits
#                     pr = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
#                     p_tr_list.append(pr)
#                 p_tr = np.concatenate(p_tr_list)
#                 y_tr = y_train.numpy()

#             if tune_threshold == "majority":
#                 thr = float(y_tr.mean())
#             elif tune_threshold == "youden":
#                 fpr, tpr, th = roc_curve(y_tr, p_tr)
#                 j = tpr - fpr
#                 thr = float(th[np.argmax(j)]) if len(th) else 0.5

#         thr_list.append(thr)

#         # --- predict one-step ahead at date_t ---
#         x_ctx = make_ctx(upto_pos=pos)
#         p1 = predict_proba(x_ctx)

#         probs.append(float(p1))
#         trues.append(int(y_cls.iloc[pos]))
#         dates.append(date_t)

#     if not probs:
#         raise RuntimeError("No valid MOMENT predictions; increase history / adjust start_oos / min_context.")

#     y_prob = np.asarray(probs, float)
#     y_true = np.asarray(trues, int)
#     thr_arr = np.asarray(thr_list, float)
#     dates = pd.DatetimeIndex(dates)

#     # --- apply variable threshold ---
#     y_hat = (y_prob >= thr_arr).astype(int)
#     acc = (y_hat == y_true).mean()
#     brier = np.mean((y_prob - y_true) ** 2)

#     if not quiet:
#         print(f"[MOMENT-CLS] steps={len(y_true)}  Acc={acc:.4f}  Brier={brier:.4f}")

#     return acc, brier, y_true, y_prob, thr_arr, dates, y_hat



import numpy as np
import pandas as pd


def make_tabpfn_lag_cls_fit_predict_fn(
    base_cols,
    target_col: str = "state",    # 0/1 (Bull/Bear)
    n_lags: int = 1,
    retrain_every = 1,
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
    step_counter = 0
    clf = None
    def fit_predict(est: pd.DataFrame, row_t: pd.Series):
        nonlocal step_counter, clf
        step_counter += 1
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
        if (clf is None) or (step_counter % retrain_every == 0):
                
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
    retrain_every = 1,
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
        model_params=model_params,
        retrain_every = retrain_every,
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
