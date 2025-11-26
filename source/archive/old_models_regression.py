
# 5.2 TimesFM
import timesfm

def timesfm_oos(
    data: pd.DataFrame,
    target_col="equity_premium",
    start_oos="1965-01-01",
    freq="MS",
    prediction_length: int = 1,
    min_context=120,
    max_context=512,
    ct_cutoff=True,
    quiet=False,
    mode = "mean"
):
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if not quiet:
        print(f"[TimesFM] Using device hint: {device}")

    df = ensure_datetime_index(data)
    y = align_monthly(df[[target_col]], freq, col=target_col)[target_col].astype("float32")
    if len(y) == 0:
        raise ValueError("No target data after cleaning.")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    cfg = timesfm.ForecastConfig(
        max_context=max_context,
        max_horizon=max(prediction_length, 128),
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
    )
    model.compile(cfg)

    def forecast_multi_step(y_hist: pd.Series, date_t, H: int) -> np.ndarray:
        context = y_hist.to_numpy(dtype="float32")
        if len(context) < min_context:
            return np.full(H, np.nan, dtype="float32")
        if len(context) > cfg.max_context:
            context = context[-cfg.max_context:]
        if np.isnan(context).any() or np.std(context) < 1e-6:
            return np.full(H, np.nan, dtype="float32")

        with torch.inference_mode():
            point_fcst, _ = model.forecast(horizon=H, inputs=[context])
        # shape: (1, H)
        return np.asarray(point_fcst[0, :H], dtype="float32")

    r2, trues, preds, dates = expanding_oos_univariate(
        y,
        start_oos=start_oos,
        prediction_length=prediction_length,
        min_history_months=0,  # rely on min_context
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name="TimesFM",
        forecast_multi_step_fn=forecast_multi_step,
        mode = mode
    )
    return r2, trues, preds, dates


# 5.3 FlowState
from tsfm_public import FlowStateForPrediction

def flowstate_oos(
    data: pd.DataFrame,
    target_col="equity_premium",
    start_oos="1965-01-01",
    ctx=240,
    freq="M",
    prediction_length: int = 1,
    scale_factor=0.25,
    quantile=0.5,
    ct_cutoff=False,
    quiet=False,
    model_name="FlowState (expanding)",
    mode = "mean"
):
    df = ensure_datetime_index(data)
    s = df[[target_col]].copy()

    def align_freq(series, f):
        z = series.copy()
        z.index = z.index.to_period(f).to_timestamp(f)
        z = z[~z.index.duplicated(keep="last")].sort_index().asfreq(f)
        z[target_col] = z[target_col].ffill()
        return z

    if freq in {"M", "MS"}:
        s = align_freq(s, freq)
        if s[target_col].isna().all():
            alt = "M" if freq == "MS" else "MS"
            s = align_freq(df[[target_col]], alt)
            if not quiet:
                print(f"[FlowState] Retried with freq='{alt}' because '{freq}' produced all-NaN.")
            freq = alt
    elif freq is not None:
        s = s.asfreq(freq)
        s[target_col] = s[target_col].ffill()

    y = s[target_col].astype("float32")
    if y.isna().all():
        raise ValueError("Target is all NaN after preprocessing/alignment.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = FlowStateForPrediction.from_pretrained("ibm-research/flowstate").to(device)

    q_idx = {"value": None}

    def forecast_multi_step(y_hist: pd.Series, date_t, H: int) -> np.ndarray:
        if len(y_hist) < ctx:
            return np.full(H, np.nan, dtype="float32")

        ctx_vals = y_hist.iloc[-ctx:].to_numpy(dtype="float32")
        if np.isnan(ctx_vals).any():
            return np.full(H, np.nan, dtype="float32")

        ctx_tensor = torch.from_numpy(ctx_vals[:, None, None]).to(torch.float32).to(device)
        with torch.inference_mode():
            out = predictor(ctx_tensor, scale_factor=scale_factor, prediction_length=H, batch_first=False)
        po = out.prediction_outputs  # expected shape: (1, num_quantiles, H, 1)

        if q_idx["value"] is None:
            if hasattr(out, "quantile_values"):
                qs = torch.tensor(out.quantile_values, device=po.device)
                q_idx["value"] = int(torch.argmin(torch.abs(qs - quantile)).item())
            else:
                q_idx["value"] = po.shape[1] // 2

        # (H,) vector
        vec = po[0, q_idx["value"], :H, 0].detach().cpu().numpy().astype("float32")
        return vec

    r2, trues, preds, dates = expanding_oos_univariate(
        y,
        start_oos=start_oos,
        prediction_length=prediction_length,
        min_history_months=0,  # rely on ctx
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name=model_name,
        forecast_multi_step_fn=forecast_multi_step,
        mode = mode
    )
    return r2, trues, preds, dates


import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

class MLPReg(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):  # x: [B, d_in]
        return self.net(x).squeeze(-1)  # [B]

def make_mlp_lag_reg_fit_predict_fn(
    base_cols,
    target_col="equity_premium",
    n_lags=12,
    epochs=5,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-2,
    retrain_every=25,
    print_loss=True,
):
    base_cols = list(base_cols)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model, scaler, step = None, None, 0
    d_in = len(base_cols) * n_lags

    def build_lagged(df: pd.DataFrame):
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

        # Prepare training data (past only)
        tmp, lag_cols = build_lagged(est)
        if tmp.empty:
            return np.nan
        X = tmp[lag_cols].to_numpy(np.float32)
        y = tmp[target_col].to_numpy(np.float32)

        # retrain periodically or on first call
        if (model is None) or (step % retrain_every == 0):
            scaler = StandardScaler().fit(X)
            Xs = scaler.transform(X).astype(np.float32)

            ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(y))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            model = MLPReg(d_in=d_in).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            crit = nn.MSELoss()

            model.train()
            for ep in range(epochs):
                run, nb = 0.0, 0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = crit(pred, yb)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    run += loss.item(); nb += 1
                if print_loss:
                    print(f"[MLPReg retrain] epoch {ep+1}/{epochs} | loss={run/max(1,nb):.6f}")

        # Build x_t from the last n_lags of each base feature
        lag_vals = []
        for c in base_cols:
            v = est[c].iloc[-n_lags:]
            if len(v) < n_lags or v.isna().any():
                return np.nan
            lag_vals.extend(v.values)
        x_t = np.asarray(lag_vals, np.float32).reshape(1, -1)

        x_tn = torch.from_numpy(scaler.transform(x_t).astype(np.float32)).to(device)
        model.eval()
        with torch.inference_mode():
            y_hat = float(model(x_tn).item())
        return y_hat

    return fit_predict



import torch.nn as nn

class LSTMReg(nn.Module):
    def __init__(self, n_feat, hidden=64, num_layers=1, bidir=False, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=bidir,
            dropout=(dropout if num_layers > 1 else 0.0)
        )
        d = hidden * (2 if bidir else 1)
        self.head = nn.Linear(d, 1)
    def forward(self, x):        # x: [B, L, F]
        out, _ = self.lstm(x)
        z = out[:, -1, :]        # last timestep
        return self.head(z).squeeze(-1)  # [B]

def make_lstm_seq_reg_fit_predict_fn(
    feature_cols,
    target_col="equity_premium",
    seq_len=24,
    epochs=5,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-2,
    retrain_every=25,
    hidden=64,
    num_layers=1,
    bidir=False,
    dropout=0.0,
    scale=True,
    print_loss=True,
):
    feature_cols = list(feature_cols)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    from sklearn.preprocessing import StandardScaler

    model, scaler, step = None, None, 0
    n_feat = len(feature_cols)

    def make_windows(X_df: pd.DataFrame, y_ser: pd.Series):
        X = X_df.to_numpy(np.float32)  # [T, F]
        y = y_ser.to_numpy(np.float32) # [T]
        T = len(X)
        if T <= seq_len: return None, None
        Xs, ys = [], []
        for t in range(seq_len, T):
            Xs.append(X[t-seq_len:t, :])   # [L, F]
            ys.append(y[t])
        return np.stack(Xs).astype(np.float32), np.asarray(ys, np.float32)

    def fit_predict(est: pd.DataFrame, row_t: pd.Series):
        nonlocal model, scaler, step
        step += 1
        df = est.copy()
        X_df = df[feature_cols].astype("float32")
        y_ser = df[target_col].astype("float32")
        if len(df) <= seq_len:
            return np.nan

        # scale features on history
        if scale:
            scaler = StandardScaler().fit(X_df.values)
            X_hist = scaler.transform(X_df.values).astype(np.float32)
            Xn_df = pd.DataFrame(X_hist, index=X_df.index, columns=X_df.columns)
        else:
            Xn_df = X_df

        # retrain periodically or on first call
        if (model is None) or (step % retrain_every == 0):
            X_train, y_train = make_windows(Xn_df, y_ser)
            if X_train is None or len(X_train) < 10:
                return np.nan

            ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            model = LSTMReg(n_feat=n_feat, hidden=hidden, num_layers=num_layers,
                            bidir=bidir, dropout=dropout).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            crit = nn.MSELoss()

            model.train()
            for ep in range(epochs):
                run, nb = 0.0, 0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = crit(pred, yb)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    run += loss.item(); nb += 1
                if print_loss:
                    print(f"[LSTMReg retrain] epoch {ep+1}/{epochs} | loss={run/max(1,nb):.6f}")

        # build one context window at t from last seq_len rows of est
        X_hist = X_df.values.astype(np.float32)
        if scale and scaler is not None:
            X_hist = scaler.transform(X_hist).astype(np.float32)
        if len(X_hist) < seq_len:
            return np.nan
        ctx = X_hist[-seq_len:, :]                     # [L, F]
        x_ctx = torch.from_numpy(ctx.reshape(1, seq_len, n_feat)).to(device)

        model.eval()
        with torch.inference_mode():
            y_hat = float(model(x_ctx).item())
        return y_hat

    return fit_predict


import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)  # [L, d_model]

    def forward(self, x):  # x: [B, L, d_model]
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)

class TransformerReg(nn.Module):
    """
    Transformer encoder -> last-timestep pooling -> linear head -> scalar regression
    """
    def __init__(self, n_feat: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(n_feat, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.posenc = PositionalEncoding(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):     # x: [B, L, F]
        h = self.in_proj(x)   # [B, L, d_model]
        h = self.posenc(h)    # + positional enc.
        h = self.encoder(h)   # [B, L, d_model]
        z = h[:, -1, :]       # use last timestep representation
        y = self.head(z).squeeze(-1)  # [B]
        return y

def make_transformer_seq_reg_fit_predict_fn(
    feature_cols,
    target_col: str = "equity_premium",
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
    print_loss: bool = True,
):
    """
    Builds a fit_predict(est,row_t)->float for expanding_oos_tabular (regression).
    Uses a Transformer encoder over the last `seq_len` steps of `feature_cols`.
    """
    device = pick_device()
    feature_cols = list(feature_cols)
    n_feat = len(feature_cols)

    model, scaler, step = None, None, 0
    crit = nn.MSELoss()

    def make_windows(X_df: pd.DataFrame, y_ser: pd.Series):
        X = X_df.to_numpy(np.float32)      # [T, F]
        y = y_ser.to_numpy(np.float32)     # [T]
        T = len(X)
        if T <= seq_len:
            return None, None
        Xs, ys = [], []
        for t in range(seq_len, T):
            Xs.append(X[t-seq_len:t, :])   # [L, F]
            ys.append(y[t])
        return np.stack(Xs).astype(np.float32), np.asarray(ys, np.float32)

    def fit_predict(est: pd.DataFrame, row_t: pd.Series):
        nonlocal model, scaler, step
        step += 1

        df = est.copy()
        if any(c not in df.columns for c in feature_cols+[target_col]):
            return np.nan
        X_df = df[feature_cols].astype("float32")
        y_ser = df[target_col].astype("float32")
        if len(df) <= seq_len:
            return np.nan

        # scale features on history
        if scale:
            scaler = StandardScaler().fit(X_df.values)
            X_hist = scaler.transform(X_df.values).astype(np.float32)
            Xn_df = pd.DataFrame(X_hist, index=X_df.index, columns=X_df.columns)
        else:
            Xn_df = X_df

        # (re)train periodically
        need_train = (model is None) or (step % retrain_every == 0)
        if need_train:
            X_train, y_train = make_windows(Xn_df, y_ser)
            if X_train is None or len(X_train) < 10:
                return np.nan

            ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            model = TransformerReg(
                n_feat=n_feat, d_model=d_model, nhead=nhead,
                num_layers=num_layers, dim_feedforward=dim_feedforward,
                dropout=dropout
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            model.train()
            for ep in range(epochs):
                run, nb = 0.0, 0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = crit(pred, yb)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    run += loss.item(); nb += 1
                if print_loss:
                    print(f"[TransformerReg retrain] epoch {ep+1}/{epochs} | loss={run/max(1,nb):.6f}")

        # ---- one-step-ahead prediction at time t ----
        # build single context window from raw history, then scale if needed
        X_hist_raw = df[feature_cols].to_numpy(np.float32)
        if len(X_hist_raw) < seq_len:
            return np.nan
        if scale and scaler is not None:
            X_hist_raw = scaler.transform(X_hist_raw).astype(np.float32)

        ctx = X_hist_raw[-seq_len:, :]  # [L, F]
        x_ctx = torch.from_numpy(ctx.reshape(1, seq_len, n_feat)).to(device)
        model.eval()
        with torch.inference_mode():
            y_hat = float(model(x_ctx).item())
        return y_hat

    return fit_predict

