# bootstrap_refit_batch.py

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Optional imports (guarded inside functions):
# - tabpfn
# - pmdarima

# ----------------------------
# Seeding
# ----------------------------
import random
import torch

def lep_to_ep_hat(lep_hat: np.ndarray, rf_prev: np.ndarray) -> np.ndarray:  #Changed
    """
    Convert log equity premium forecast (LEP) to simple equity premium forecast (EP).
    EP_hat = (1 + Rf_{t-1}) * (exp(LEP_hat) - 1)

    lep_hat: array of LEP forecasts for period t (made at t-1)
    rf_prev: array of risk-free returns dated t-1 aligned with lep_hat
    """
    lep_hat = np.asarray(lep_hat, float)  #Changed
    rf_prev = np.asarray(rf_prev, float)  #Changed
    out = np.full_like(lep_hat, np.nan, dtype=float)  #Changed
    m = np.isfinite(lep_hat) & np.isfinite(rf_prev)  #Changed
    out[m] = (1.0 + rf_prev[m]) * (np.exp(lep_hat[m]) - 1.0)  #Changed
    return out  #Changed


def compute_weights_from_premium_forecast(  #Changed
    ep_hat: np.ndarray,
    var_hat: float,
    *,
    gamma: float = 5.0,
    w_min: float = 0.0,
    w_max: float = 1.5,
):
    ep_hat = np.asarray(ep_hat, float)  #Changed
    if (not np.isfinite(var_hat)) or var_hat <= 0:  #Changed
        return np.zeros_like(ep_hat, dtype=float)  #Changed
    w = ep_hat / (gamma * var_hat)  #Changed
    return np.clip(w, w_min, w_max)  #Changed
def portfolio_returns_from_weights(  #Changed
    w_risky: np.ndarray,
    r_risky: np.ndarray,
    r_rf: np.ndarray,
):
    w_risky = np.asarray(w_risky, float)  #Changed
    r_risky = np.asarray(r_risky, float)  #Changed
    r_rf    = np.asarray(r_rf, float)  #Changed

    m = np.isfinite(w_risky) & np.isfinite(r_risky) & np.isfinite(r_rf)  #Changed
    rp = np.full_like(r_risky, np.nan, dtype=float)  #Changed
    rp[m] = w_risky[m] * r_risky[m] + (1.0 - w_risky[m]) * r_rf[m]  #Changed
    return rp  #Changed

def compute_order_free_metrics(rp: np.ndarray, r_rf: np.ndarray, *, A: int, gamma: float):  #Changed
    """
    Metrics that do NOT require returns to be in time order:  #Changed
      - AnnVol
      - SR_ann
      - CER_ann
    (No TR/CAGR/MaxDD here, because your bootstrap breaks chronology.)  #Changed
    """
    rp = np.asarray(rp, float)  #Changed
    r_rf = np.asarray(r_rf, float)  #Changed
    m = np.isfinite(rp) & np.isfinite(r_rf)  #Changed
    rp, r_rf = rp[m], r_rf[m]  #Changed
    if rp.size < 3:  #Changed
        return {"AnnVol": np.nan, "SR_ann": np.nan, "CER_ann": np.nan, "n": int(rp.size)}  #Changed

    mu = float(np.mean(rp))  #Changed
    var = float(np.var(rp, ddof=1))  #Changed
    sd = float(np.sqrt(var))  #Changed

    annvol = float(np.sqrt(A) * sd)  #Changed

    re = rp - r_rf  #Changed
    mu_e = float(np.mean(re))  #Changed
    sd_e = float(np.std(re, ddof=1))  #Changed
    sr_ann = float(np.sqrt(A) * (mu_e / sd_e)) if sd_e > 0 else np.nan  #Changed

    cer_ann = float(A * (mu - 0.5 * gamma * var))  #Changed

    return {"AnnVol": annvol, "SR_ann": sr_ann, "CER_ann": cer_ann, "n": int(rp.size)}  #Changed


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------------------
# Utilities
# ----------------------------
def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        elif "Date" in df.columns:
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)
    return df.sort_index()

def make_lagged_features(df: pd.DataFrame, vars_, lag: int) -> pd.DataFrame:
    df = df.copy()
    for L in range(1, lag + 1):
        for v in vars_:
            df[f"{v}_lag{L}"] = df[v].shift(L)
    return df

def r2_oos_vs_bench(y_true, y_pred, y_bench):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    y_bench = np.asarray(y_bench, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_bench)
    y_true, y_pred, y_bench = y_true[m], y_pred[m], y_bench[m]
    denom = np.sum((y_true - y_bench) ** 2)
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)

def summarize_bootstrap(vals, ci=0.90):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"boot": vals, "mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    a = (1.0 - ci) / 2.0
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "ci_lower": float(np.quantile(vals, a)),
        "ci_upper": float(np.quantile(vals, 1 - a)),
        "ci": float(ci),
        "n": int(len(vals)),
    }

def bootstrap_indices_iid(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)

def bootstrap_indices_mbb(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    # simple moving-block bootstrap
    block = max(1, min(block, n))
    k = int(np.ceil(n / block))
    start_max = n - block
    starts = rng.integers(0, start_max + 1, size=k) if start_max >= 0 else np.zeros(k, dtype=int)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx.astype(int)

def build_benchmark_HA(df: pd.DataFrame, target_col: str) -> pd.Series:
    # expanding mean up to t-1 (skipna=True by default)
    return df[target_col].expanding().mean().shift(1)


# ----------------------------
# Batch fit→predict functions
# ----------------------------

def tabpfn_fit_predict_batch(
    X_train,
    y_train,
    X_test,
    *,
    seed: int,
    model_params=None,  # "2.5" -> TabPFNRegressor(...), else V2 default (like your oos fn)
):
    """
    Simple batch TabPFN fit->predict that matches your model selection and adds:
      - deterministic seeding
      - drops training rows with any NaNs (TabPFN can't handle NaNs)
      - requires test rows to have no NaNs (returns NaN for those rows)

    Accepts numpy arrays or pandas DataFrames.
    Returns yhat shape (n_test,).
    """
    import numpy as np
    import torch
    from tabpfn import TabPFNRegressor
    try:
        from tabpfn.constants import ModelVersion
    except Exception:
        ModelVersion = None

    set_global_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_params == "2.5" or ModelVersion is None:
        model = TabPFNRegressor(device=device)
    else:
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=device)

    Xtr = np.asarray(X_train, dtype=float)
    ytr = np.asarray(y_train, dtype=float).reshape(-1)
    Xte = np.asarray(X_test, dtype=float)

    if Xtr.ndim != 2 or Xte.ndim != 2:
        raise ValueError("X_train and X_test must be 2D arrays.")
    if ytr.shape[0] != Xtr.shape[0]:
        raise ValueError("y_train length must match X_train rows.")
    if Xtr.shape[1] != Xte.shape[1]:
        raise ValueError("X_train and X_test must have the same number of columns.")

    # drop NaN rows in training (TabPFN cannot fit with NaNs)
    tr_ok = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
    Xtr_fit = Xtr[tr_ok]
    ytr_fit = ytr[tr_ok]

    if Xtr_fit.shape[0] == 0:
        return np.full((Xte.shape[0],), np.nan, dtype=float)

    model.fit(Xtr_fit, ytr_fit)

    # predict; for test rows with NaNs, return NaN
    yhat = np.full((Xte.shape[0],), np.nan, dtype=float)
    te_ok = np.all(np.isfinite(Xte), axis=1)
    if np.any(te_ok):
        yhat[te_ok] = np.asarray(model.predict(Xte[te_ok], output_type="mean"), float).reshape(-1)

    return yhat


def pcr_fit_predict_batch(X_train, y_train, X_test, *, k: int = 1):
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_train)
    Xs_te = scaler.transform(X_test)

    pca = PCA(n_components=None)
    Z_tr = pca.fit_transform(Xs_tr)

    k_eff = min(k, Z_tr.shape[1])
    reg = LinearRegression().fit(Z_tr[:, :k_eff], y_train)

    Z_te = pca.transform(Xs_te)[:, :k_eff]
    return reg.predict(Z_te)

def combo_ols_fit_predict_batch(X_train_dict, y_train, X_test_dict, *, combo="mean", trim_q=0.1):
    # X_train_dict: var -> (n_train, p_var), X_test_dict: var -> (n_test, p_var)
    preds = []
    for v, Xtr in X_train_dict.items():
        Xte = X_test_dict[v]
        reg = LinearRegression().fit(Xtr, y_train)
        preds.append(reg.predict(Xte).reshape(1, -1))
    P = np.vstack(preds)  # shape (n_models, n_test)

    if combo == "mean":
        return P.mean(axis=0)
    if combo == "median":
        return np.median(P, axis=0)
    if combo == "trimmed_mean":
        m = P.shape[0]
        if m < 3:
            return P.mean(axis=0)
        k = int(np.floor(trim_q * m))
        Ps = np.sort(P, axis=0)
        return Ps[k:m-k, :].mean(axis=0) if 2*k < m else Ps.mean(axis=0)

    raise ValueError("combo must be mean/median/trimmed_mean")

def autoarima_fit_predict_batch(y_train, n_test, *, seed: int, auto_arima_kwargs=None):
    from pmdarima import auto_arima
    if auto_arima_kwargs is None:
        auto_arima_kwargs = {}

    # pmdarima can be made more deterministic by fixing random_state where applicable
    model = auto_arima(
        y_train,
        seasonal=True,
        m=12,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        random_state=seed,
        **auto_arima_kwargs,
    )
    fc = model.predict(n_periods=n_test)
    return np.asarray(fc, float).reshape(-1)


# ----------------------------
# Main refit-bootstrap runner
# ----------------------------
def bootstrap_refit_train_test(
    df: pd.DataFrame,
    *,
    target_col: str,                 #Changed  <-- your y_t column for R2 evaluation (LEP or EP)
    feature_cols: list[str],
    start_oos: str = "1965-01-01",
    start_data: str = "1927-01-01",
    min_train: int = 240,
    B: int = 1000,
    seed: int = 42,
    bootstrap: str = "iid",          # "iid" or "mbb"
    block_size: int = 12,
    model: str = "pcr",              # "pcr" | "combo" | "chronos_2" | "moirai_2" | ...
    combo: str = "mean",
    pcr_k: int = 1,
    trim_q: float = 0.1,
    ct_cutoff: bool = True,
    y_pred_given: np.ndarray | list[float] | None = None,

    # ---------- trading settings ----------
    compute_trading: bool = True,
    risky_ret_col = "ret",   #Changed  <-- e.g. "R_m"
    rf_ret_col = "Rfree",      #Changed  <-- e.g. "R_f"
    rf_lag_col: str = "Rfree_lag1",      #Changed  <-- e.g. "R_f_lag1" (needed if pred_is_lep=True)
    A: int = 12,                        #Changed  <-- 12 monthly, 252 daily
    gamma: float = 5.0,
    var_window: int = 60,
    w_min: float = 0.0,
    w_max: float = 1.5,
    compute_delta_u: bool = True,
    pred_is_lep: bool = True,          
    ep_realized_col: str = "equity_premium"
):
    df = ensure_datetime_index(df).copy()

    # restrict to sample start
    df = df[df.index >= start_data]
    df_additional = df[(df.index < start_oos) & (df.index >= start_data)]
    n_train = len(df)

    # infer return columns if not provided
    # bootstrap distributions
    boot_r2 = np.full(B, np.nan, float)
    if compute_trading:
        boot_sr     = np.full(B, np.nan, float)
        boot_cer    = np.full(B, np.nan, float)
        boot_annvol = np.full(B, np.nan, float)
        boot_du_ha  = np.full(B, np.nan, float)
        boot_du_w50 = np.full(B, np.nan, float)
        boot_du_w100= np.full(B, np.nan, float)

    # helpers for combo (kept compatible with your current style)
    def _infer_feat_map(cols: list[str]) -> dict[str, list[str]]:
        feat_map = {}
        for c in cols:
            base = c.split("_lag")[0] if "_lag" in c else c
            feat_map.setdefault(base, []).append(c)
        for k in list(feat_map.keys()):
            feat_map[k] = sorted(
                feat_map[k],
                key=lambda s: int(s.split("_lag")[-1]) if "_lag" in s else 0
            )
        return feat_map

    def _combine_preds(P: np.ndarray, method: str, trim_q: float) -> np.ndarray:
        if P.size == 0:
            return np.full(P.shape[1], np.nan, dtype=float)
        if method == "mean":
            return np.nanmean(P, axis=0)
        if method == "median":
            return np.nanmedian(P, axis=0)
        if method == "trimmed_mean":
            out = np.full(P.shape[1], np.nan, dtype=float)
            for j in range(P.shape[1]):
                vals = P[:, j]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                vals.sort()
                k = int(np.floor(trim_q * vals.size))
                out[j] = float(np.mean(vals[k:-k])) if 2 * k < vals.size else float(np.mean(vals))
            return out
        raise ValueError("combo must be 'mean', 'median', or 'trimmed_mean'.")

    # -----------------------------
    # main bootstrap loop
    # -----------------------------
    for b in range(B):
        rng = np.random.default_rng(seed + b)

        # resample indices from df (not df_additional)
        if bootstrap == "iid":
            idx = bootstrap_indices_iid(n_train, rng)
        elif bootstrap == "mbb":
            idx = bootstrap_indices_mbb(n_train, block_size, rng)
        else:
            raise ValueError("bootstrap must be 'iid' or 'mbb'")

        train_star = df.iloc[idx].copy()

        # out-of-bag set (may be unordered, that's fine for SR/CER/vol)
        test_mask = ~df.index.isin(train_star.index)
        df_test = df.loc[test_mask].copy()

        # --- if test set is too small, skip replicate ---
        if len(df_test) < 5:
            continue

        y_test = df_test[target_col].to_numpy(float)

        # HA computed from the bootstrap TRAIN draw (before adding additional history)
        ha = float(np.nanmean(train_star[target_col].to_numpy(float)))

        # add additional pre-OOS data (keeps chronology inside train_star)
        train_star_full = pd.concat([df_additional, train_star]).sort_index()

        # ---------- fit/predict (refit / non-refit) ----------
        if model == "pcr":
            Xtr = train_star_full[feature_cols].to_numpy(float)
            ytr = train_star_full[target_col].to_numpy(float)
            Xte = df_test[feature_cols].to_numpy(float)
            yhat = pcr_fit_predict_batch(Xtr, ytr, Xte, k=pcr_k)

        elif model == "combo":
            feat_map = _infer_feat_map(feature_cols)
            preds = []
            for v, cols_v in feat_map.items():
                est_v = train_star_full.dropna(subset=cols_v + [target_col])
                if len(est_v) < min_train:
                    continue
                ok_te = df_test[cols_v].notna().all(axis=1)
                if not ok_te.any():
                    continue
                Xtr_v = est_v[cols_v].to_numpy(float)
                ytr_v = est_v[target_col].to_numpy(float)
                reg = LinearRegression().fit(Xtr_v, ytr_v)

                p = np.full(len(df_test), np.nan, dtype=float)
                Xte_v = df_test.loc[ok_te, cols_v].to_numpy(float)
                p[ok_te.to_numpy()] = reg.predict(Xte_v).astype(float)
                preds.append(p)

            yhat = np.full(len(df_test), np.nan, dtype=float) if len(preds) == 0 else _combine_preds(np.vstack(preds), combo, trim_q)

        elif (model == "chronos_2") or (model == "moirai_2"):
            if y_pred_given is None:
                raise ValueError("chronos_2/moirai_2 require y_pred_given.  #Changed")
            # IMPORTANT: y_pred_given must be aligned to df.index order (same length as df)  #Changed
            y_pred_given_arr = np.asarray(y_pred_given, float)
            if y_pred_given_arr.shape[0] != df.shape[0]:
                raise ValueError("y_pred_given must have same length as df (after start_data filter).  #Changed")
            yhat = y_pred_given_arr[test_mask]  #Changed  <-- correct alignment for OOB rows

        else:
            raise ValueError(f"Unknown model: {model}")

        # CT cutoff on the FORECAST (in its own units)
        if ct_cutoff:
            yhat = np.maximum(yhat, 0.0)

        # ---------- R2 vs HA ----------
        ha_test = np.full_like(y_test, ha, dtype=float)
        boot_r2[b] = r2_oos_vs_bench(y_test, yhat, ha_test)

        # ---------- Trading metrics (order-free) ----------
        if compute_trading:
            r_risky = df_test[risky_ret_col].to_numpy(float)
            r_rf    = df_test[rf_ret_col].to_numpy(float)

            # Convert forecasts to EP if they are LEP (needed for weight rule)
            if pred_is_lep:
                rf_prev = df_test[rf_lag_col].to_numpy(float)  #Changed
                ep_hat_model = lep_to_ep_hat(yhat, rf_prev)
            else:
                ep_hat_model = yhat

            # prevailing variance estimate from TRAIN history tail on REALIZED EP
            ep_hist = train_star_full[ep_realized_col].to_numpy(float)
            ep_hist = ep_hist[np.isfinite(ep_hist)]
            tail = ep_hist[-min(var_window, len(ep_hist)):]
            var_hat = float(np.var(tail, ddof=1)) if tail.size > 1 else np.nan

            # model strategy
            w_model = compute_weights_from_premium_forecast(ep_hat_model, var_hat, gamma=gamma, w_min=w_min, w_max=w_max)
            rp_model = portfolio_returns_from_weights(w_model, r_risky, r_rf)
            m_model = compute_order_free_metrics(rp_model, r_rf, A=A, gamma=gamma)

            boot_sr[b]     = m_model["SR_ann"]
            boot_cer[b]    = m_model["CER_ann"]
            boot_annvol[b] = m_model["AnnVol"]

            if compute_delta_u:
                # HA timing benchmark forecast in SAME units as model output
                yhat_ha = np.full(len(df_test), ha, dtype=float)
                if ct_cutoff:
                    yhat_ha = np.maximum(yhat_ha, 0.0)

                if pred_is_lep:
                    rf_prev = df_test[rf_lag_col].to_numpy(float)  #Changed
                    ep_hat_ha = lep_to_ep_hat(yhat_ha, rf_prev)
                else:
                    ep_hat_ha = yhat_ha

                w_ha = compute_weights_from_premium_forecast(ep_hat_ha, var_hat, gamma=gamma, w_min=w_min, w_max=w_max)
                rp_ha = portfolio_returns_from_weights(w_ha, r_risky, r_rf)
                m_ha = compute_order_free_metrics(rp_ha, r_rf, A=A, gamma=gamma)

                # W50 / W100 benchmarks
                rp_w50  = portfolio_returns_from_weights(np.full(len(df_test), 0.5), r_risky, r_rf)
                rp_w100 = portfolio_returns_from_weights(np.full(len(df_test), 1.0), r_risky, r_rf)
                m_w50   = compute_order_free_metrics(rp_w50,  r_rf, A=A, gamma=gamma)
                m_w100  = compute_order_free_metrics(rp_w100, r_rf, A=A, gamma=gamma)

                boot_du_ha[b]   = m_model["CER_ann"] - m_ha["CER_ann"]
                boot_du_w50[b]  = m_model["CER_ann"] - m_w50["CER_ann"]
                boot_du_w100[b] = m_model["CER_ann"] - m_w100["CER_ann"]

    # -----------------------------
    # summarize outputs
    # -----------------------------
    out = {"r2": summarize_bootstrap(boot_r2, ci=0.90)}

    if compute_trading:
        out["SR_ann"]  = summarize_bootstrap(boot_sr, ci=0.90)
        out["CEV_ann"] = summarize_bootstrap(boot_cer, ci=0.90)
        out["AnnVol"]  = summarize_bootstrap(boot_annvol, ci=0.90)
        if compute_delta_u:
            out["Delta_u_vs_HA"]   = summarize_bootstrap(boot_du_ha, ci=0.90)
            out["Delta_u_vs_W50"]  = summarize_bootstrap(boot_du_w50, ci=0.90)
            out["Delta_u_vs_W100"] = summarize_bootstrap(boot_du_w100, ci=0.90)

    return out
