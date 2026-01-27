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
    target_col: str,
    feature_cols: list[str],
    start_oos: str,
    min_train: int = 240,
    B: int = 1000,
    seed: int = 42,
    bootstrap: str = "iid",     # "iid" or "mbb"
    block_size: int = 12,
    model: str = "pcr",         # "tabpfn" | "pcr" | "combo" | "autoarima"
    combo: str = "mean",        # for combo
    pcr_k: int = 1,
    trim_q: float = 0.1,
    ct_cutoff: bool = True,
    auto_arima_kwargs: dict | None = None,
):
    df = ensure_datetime_index(df).copy()
    df = df[df.index >= start_oos]
    # Bootstrap distribution
    boot_r2 = np.empty(B, float)
    n_train = len(df)
    def _infer_feat_map(cols: list[str]) -> dict[str, list[str]]:  #changed
        feat_map = {}  #changed
        for c in cols:  #changed
            base = c.split("_lag")[0] if "_lag" in c else c  #changed
            feat_map.setdefault(base, []).append(c)  #changed
        # keep a stable order of lag columns if they exist  #changed
        for k in list(feat_map.keys()):  #changed
            feat_map[k] = sorted(feat_map[k], key=lambda s: int(s.split("_lag")[-1]) if "_lag" in s else 0)  #changed
        return feat_map  #changed
    def _combine_preds(P: np.ndarray, method: str, trim_q: float) -> np.ndarray:  #changed
        if P.size == 0:  #changed
            return np.full(P.shape[1], np.nan, dtype=float)  #changed
        if method == "mean":  #changed
            return np.nanmean(P, axis=0)  #changed
        if method == "median":  #changed
            return np.nanmedian(P, axis=0)  #changed
        if method == "trimmed_mean":  #changed
            out = np.full(P.shape[1], np.nan, dtype=float)  #changed
            for j in range(P.shape[1]):  #changed
                vals = P[:, j]  #changed
                vals = vals[np.isfinite(vals)]  #changed
                if vals.size == 0:  #changed
                    continue  #changed
                vals.sort()  #changed
                k = int(np.floor(trim_q * vals.size))  #changed
                if 2 * k >= vals.size:  #changed
                    out[j] = float(np.mean(vals))  #changed
                else:  #changed
                    out[j] = float(np.mean(vals[k:-k]))  #changed
            return out  #changed
        raise ValueError("combo must be 'mean', 'median', or 'trimmed_mean'.")  #changed

    for b in range(B):
        rng = np.random.default_rng(seed + b)
        set_global_seed(seed + b)

        if bootstrap == "iid":
            idx = bootstrap_indices_iid(n_train, rng)
        elif bootstrap == "mbb":
            idx = bootstrap_indices_mbb(n_train, block_size, rng)
        else:
            raise ValueError("bootstrap must be 'iid' or 'mbb'")

        train_star = df.iloc[idx].copy()
        #df_test as all indeces not in idx
        test_mask = ~df.index.isin(train_star.index)
        df_test = df.loc[test_mask].copy()
        y_test = df_test[target_col].to_numpy(float)
        ha = train_star[target_col].mean()
        #ha_test = ha.loc[df_test.index].to_numpy(float) 
        #print prersentage of observations in df_test
        print(f"Bootstrap {b+1}/{B}: df_test has {len(df_test)}/{len(df)} observations ({len(df_test)/len(df)*100:.2f}%)")

        if model == "autoarima":
            ytr = train_star[target_col].to_numpy(float)
            yhat = autoarima_fit_predict_batch(ytr, n_test=len(df_test), seed=seed + b, auto_arima_kwargs=auto_arima_kwargs)

        elif model == "tabpfn":
            Xtr = train_star[feature_cols].to_numpy(float)
            ytr = train_star[target_col].to_numpy(float)
            Xte = df_test[feature_cols].to_numpy(float)
            yhat = tabpfn_fit_predict_batch(Xtr, ytr, Xte, seed=seed + b)

        elif model == "pcr":
            Xtr = train_star[feature_cols].to_numpy(float)
            ytr = train_star[target_col].to_numpy(float)
            Xte = df_test[feature_cols].to_numpy(float)
            yhat = pcr_fit_predict_batch(Xtr, ytr, Xte, k=pcr_k)
        elif model == "combo":  #changed
            feat_map = _infer_feat_map(feature_cols)  #changed
            preds = []  #changed
            for v, cols_v in feat_map.items():  #changed
                est_v = train_star.dropna(subset=cols_v + [target_col])  #changed
                if len(est_v) < min_train:  #changed
                    continue  #changed
                ok_te = df_test[cols_v].notna().all(axis=1)  #changed
                if not ok_te.any():  #changed
                    continue  #changed
                Xtr_v = est_v[cols_v].to_numpy(float)  #changed
                ytr_v = est_v[target_col].to_numpy(float)  #changed
                reg = LinearRegression().fit(Xtr_v, ytr_v)  #changed
                p = np.full(len(df_test), np.nan, dtype=float)  #changed
                Xte_v = df_test.loc[ok_te, cols_v].to_numpy(float)  #changed
                p[ok_te.to_numpy()] = reg.predict(Xte_v).astype(float)  #changed
                preds.append(p)  #changed
                if len(preds) == 0:  #changed
                    yhat = np.full(len(df_test), np.nan, dtype=float)  #changed
                else:  #changed
                    P = np.vstack(preds)  #changed
                    yhat = _combine_preds(P, combo, trim_q)  #changed

        if ct_cutoff:
            yhat = np.maximum(yhat, 0.0)
        # scale ha to length of y_test
        ha_test = np.full_like(y_test, ha, dtype=float)
        boot_r2[b] = r2_oos_vs_bench(y_test, yhat, ha_test)

    stats = summarize_bootstrap(boot_r2, ci=0.90)
    return  stats


def bootstrap_refit_combo(
    df: pd.DataFrame,
    *,
    target_col: str,
    start_oos: str,
    variables: list[str],
    lag: int = 1,
    min_train: int = 240,
    B: int = 1000,
    seed: int = 42,
    bootstrap: str = "iid",
    block_size: int = 12,
    combo: str = "mean",
    trim_q: float = 0.1,
    ct_cutoff: bool = False,
):
    df = ensure_datetime_index(df).copy()
    df = make_lagged_features(df, variables, lag)

    feat_map = {v: [f"{v}_lag{L}" for L in range(1, lag + 1)] for v in variables}
    all_cols = [c for cols in feat_map.values() for c in cols]

    ha = build_benchmark_HA(df, target_col)

    start_oos_ts = pd.Timestamp(start_oos)
    train_mask = df.index < start_oos_ts
    test_mask  = df.index >= start_oos_ts

    # require y + each var's lags
    train_valid = train_mask & df[target_col].notna()
    for c in all_cols:
        train_valid &= df[c].notna()

    test_valid = test_mask & df[target_col].notna() & ha.notna()
    for c in all_cols:
        test_valid &= df[c].notna()

    df_train = df.loc[train_valid].copy()
    df_test  = df.loc[test_valid].copy()

    if len(df_train) < min_train:
        raise ValueError(f"Not enough training data after filtering: {len(df_train)} < {min_train}")

    y_test = df_test[target_col].to_numpy(float)
    ha_test = ha.loc[df_test.index].to_numpy(float)

    # point estimate
    X_train_dict0 = {v: df_train[feat_map[v]].to_numpy(float) for v in variables}
    X_test_dict0  = {v: df_test[feat_map[v]].to_numpy(float)  for v in variables}
    y_train0 = df_train[target_col].to_numpy(float)

    yhat0 = combo_ols_fit_predict_batch(X_train_dict0, y_train0, X_test_dict0, combo=combo, trim_q=trim_q)
    if ct_cutoff:
        yhat0 = np.maximum(yhat0, 0.0)
    r2_point = r2_oos_vs_bench(y_test, yhat0, ha_test)

    # bootstrap
    boot_r2 = np.empty(B, float)
    n_train = len(df_train)

    for b in range(B):
        rng = np.random.default_rng(seed + b)
        if bootstrap == "iid":
            idx = bootstrap_indices_iid(n_train, rng)
        elif bootstrap == "mbb":
            idx = bootstrap_indices_mbb(n_train, block_size, rng)
        else:
            raise ValueError("bootstrap must be 'iid' or 'mbb'")

        train_star = df_train.iloc[idx].copy()

        X_train_dict = {v: train_star[feat_map[v]].to_numpy(float) for v in variables}
        y_train = train_star[target_col].to_numpy(float)

        yhat = combo_ols_fit_predict_batch(X_train_dict, y_train, X_test_dict0, combo=combo, trim_q=trim_q)
        if ct_cutoff:
            yhat = np.maximum(yhat, 0.0)

        boot_r2[b] = r2_oos_vs_bench(y_test, yhat, ha_test)

    stats = summarize_bootstrap(boot_r2, ci=0.90)
    return r2_point, stats
