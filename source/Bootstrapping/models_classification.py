import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os, sys
sys.path.insert(0, os.path.abspath('../'))
from source.Classification.models_classification import (
    make_lag1_features,
    build_mantis_X_block,
    precompute_mantis_embeddings,
    set_global_seed,
    ensure_datetime_index_from_timestamp
)



def summarize_bootstrap(vals, ci=0.90):  #Changed
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "ci": float(ci), "n": 0}
    a = (1.0 - ci) / 2.0
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "ci_lower": float(np.quantile(vals, a)),
        "ci_upper": float(np.quantile(vals, 1 - a)),
        "ci": float(ci),
        "n": int(len(vals)),
    }

def _boot_idx_iid(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n, endpoint=False)

def _boot_idx_block(n: int, rng: np.random.Generator, block_len: int) -> np.ndarray:
    """
    Circular moving block bootstrap.
    """
    if block_len <= 1:
        return _boot_idx_iid(n, rng)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks, endpoint=False)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n] % n
    return idx

def _boot_idx(n: int, rng: np.random.Generator, method: str, block_len: int) -> np.ndarray:
    if method == "iid":
        return _boot_idx_iid(n, rng)
    if method == "block":
        return _boot_idx_block(n, rng, block_len)
    raise ValueError("bootstrap_method must be 'iid' or 'block'")

def _ensure_both_classes(y: np.ndarray, idx: np.ndarray) -> bool:
    u = np.unique(y[idx])
    return (0 in u) and (1 in u)

def _sample_boot_idx_binary(
    y_train: np.ndarray,
    rng: np.random.Generator,
    *,
    method: str,
    block_len: int,
    max_tries: int = 30,
) -> np.ndarray:
    """
    Logistic regression can fail if a bootstrap draw has only one class.
    This keeps sampling until both classes appear (or falls back to original).
    """
    n = len(y_train)
    for _ in range(max_tries):
        idx = _boot_idx(n, rng, method, block_len)
        if _ensure_both_classes(y_train, idx):
            return idx
    return np.arange(n)


# -----------------------
# Binary fit/predict helpers
# -----------------------
def _fit_binary_model(
    model: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    device: str | None = None,
    rf_n_jobs: int = -1,
):
    if model == "logit":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )),
        ])
        clf.fit(X_train, y_train)
        return clf

    if model == "rf":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=seed,
            n_jobs=rf_n_jobs,
            class_weight="balanced_subsample",
        )
        clf.fit(X_train, y_train)
        return clf

    if model == "tabpfn25":
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(device=device)
        clf.fit(X_train.astype(np.float32), y_train.astype(int))
        return clf

    raise ValueError("model must be 'logit', 'rf', or 'tabpfn25' for tabular bootstrap.")

def _predict_p1(model_obj, X: np.ndarray) -> np.ndarray:
    """
    Return P(y=1). Assumes binary classifier.
    Falls back to hard prediction if no predict_proba exists.
    """
    if hasattr(model_obj, "predict_proba"):
        P = model_obj.predict_proba(X)  # columns are in model_obj.classes_
        classes = model_obj.classes_
        # index of class 1
        j1 = int(np.where(classes == 1)[0][0])
        return P[:, j1].astype(float)
    # fallback
    yhat = model_obj.predict(X).astype(int)
    return yhat.astype(float)


# -----------------------
# Main bootstrap OOS
# -----------------------
# def expanding_oos_refit_every_cls_bootstrap_binary(
#     data: pd.DataFrame,
#     *,
#     feature_cols: list[str],
#     target_col: str = "state",
#     start_oos: str = "2010-01-01",
#     start_date: str = "2000-01-05",
#     min_train: int = 120,
#     refit_every: int = 30,
#     model: str = "logit",  # "logit", "rf", "tabpfn25", "mantis_head", "mantis_rf_head"
#     mantis_context_len: int = 512,
#     seed: int = 42,
#     device: str | None = None,
#     max_train: int | None = None,
#     mantis_max_train_windows: int | None = None,
#     # bootstrap controls
#     n_boot: int = 200,
#     bootstrap_method: str = "block",     # "block" recommended
#     bootstrap_block_len: int | None = None,
#     alpha: float = 0.05,
#     threshold: float = 0.5,
#     bootstrap_n_jobs: int = 1,
#     quiet: bool = False,
#     # mantis speed
#     mantis_cache_embeddings: bool = True,
#     mantis_embed_chunk: int = 256,
# ):
#     """
#     Returns:
#       pred_df: index timestamp with y_true, p_mean, p_lo, p_hi, y_pred
#       pred_draws_df: index (timestamp, boot_id) with y_true, p1, y_pred_draw
#     """
#     set_global_seed(seed)

#     df = ensure_datetime_index_from_timestamp(data, ts_col="timestamp")
#     df = df.loc[pd.Timestamp(start_date):].copy()

#     start_ts = pd.Timestamp(start_oos)
#     loop_dates = df.index[df.index >= start_ts]
#     if len(loop_dates) == 0:
#         empty = pd.DataFrame().set_index(pd.DatetimeIndex([]))
#         return empty, empty

#     if device is None:
#         if torch.cuda.is_available():
#             device = "cuda"
#         elif torch.backends.mps.is_available():
#             device = "mps"
#         else:
#             device = "cpu"

#     if bootstrap_block_len is None:
#         bootstrap_block_len = refit_every

#     # shared arrays
#     X_lag = make_lag1_features(df, feature_cols)
#     X_lag_np = X_lag[feature_cols].to_numpy(dtype=np.float32)
#     y_np = df[target_col].to_numpy(dtype=float)
#     feats_raw = df[feature_cols].to_numpy(dtype=float)

#     idx_map = pd.Series(np.arange(len(df.index)), index=df.index)
#     loop_pos = idx_map.loc[loop_dates].to_numpy(dtype=int)

#     # Mantis precompute if needed
#     Z_all, ok_embed, mantis_trainer, head_kind = None, None, None, None
#     if model in ("mantis_head", "mantis_rf_head"):
#         from mantis.architecture import Mantis8M
#         from mantis.trainer import MantisTrainer

#         mantis_network = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
#         mantis_network.eval()
#         mantis_trainer = MantisTrainer(device=device, network=mantis_network)
#         head_kind = "logit" if model == "mantis_head" else "rf"

#         if mantis_cache_embeddings:
#             with torch.inference_mode():
#                 Z_all, ok_embed = precompute_mantis_embeddings(
#                     mantis_trainer,
#                     feats_raw=feats_raw,
#                     context_len=mantis_context_len,
#                     chunk_size=mantis_embed_chunk,
#                 )

#     rows_summary = []
#     rows_draws = []

#     i = 0
#     while i < len(loop_dates):
#         pos0 = loop_pos[i]
#         date_t = loop_dates[i]
#         j = min(i + refit_every, len(loop_dates))
#         block_pos = loop_pos[i:j]
#         block_dates = df.index[block_pos]

#         # -------------------------
#         # Build training + test sets
#         # -------------------------
#         if model in ("logit", "rf", "tabpfn25"):
#             X_train = X_lag_np[:pos0]
#             y_train = y_np[:pos0]
#             m = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
#             X_train = X_train[m]
#             y_train = y_train[m].astype(int)

#             if max_train is not None and len(y_train) > max_train:
#                 X_train = X_train[-max_train:]
#                 y_train = y_train[-max_train:]

#             if len(y_train) < min_train:
#                 i = j
#                 continue

#             X_blk = X_lag_np[block_pos]
#             y_blk = y_np[block_pos]
#             ok = np.isfinite(X_blk).all(axis=1) & np.isfinite(y_blk)
#             if not ok.any():
#                 i = j
#                 continue

#             X_test = X_blk[ok]
#             y_test = y_blk[ok].astype(int)
#             dates_ok = block_dates[ok]

#             # ensure binary present in training at all
#             if not ((0 in np.unique(y_train)) and (1 in np.unique(y_train))):
#                 i = j
#                 continue

#             # if we parallelize bootstraps, avoid RF nested parallel
#             rf_n_jobs = 1 if bootstrap_n_jobs != 1 else -1

#             def _one_boot(b: int):
#                 rng = np.random.default_rng(seed + 100000 * i + b)
#                 idx = _sample_boot_idx_binary(
#                     y_train, rng,
#                     method=bootstrap_method,
#                     block_len=bootstrap_block_len,
#                 )
#                 mdl = _fit_binary_model(
#                     model, X_train[idx], y_train[idx],
#                     seed=seed + 100000 * i + b,
#                     device=device,
#                     rf_n_jobs=rf_n_jobs,
#                 )
#                 p1 = _predict_p1(mdl, X_test)
#                 return p1

#         elif model in ("mantis_head", "mantis_rf_head"):
#             train_pos = np.arange(mantis_context_len, pos0, dtype=int)
#             if len(train_pos) == 0:
#                 i = j
#                 continue

#             if mantis_cache_embeddings:
#                 m_tr = ok_embed[train_pos] & np.isfinite(y_np[train_pos])
#                 Z_train = Z_all[train_pos][m_tr]
#                 y_train = y_np[train_pos][m_tr].astype(int)

#                 if mantis_max_train_windows is not None and len(y_train) > mantis_max_train_windows:
#                     Z_train = Z_train[-mantis_max_train_windows:]
#                     y_train = y_train[-mantis_max_train_windows:]

#                 if len(y_train) < min_train:
#                     i = j
#                     continue

#                 m_blk = ok_embed[block_pos] & np.isfinite(y_np[block_pos])
#                 if not m_blk.any():
#                     i = j
#                     continue

#                 Z_test = Z_all[block_pos][m_blk]
#                 y_test = y_np[block_pos][m_blk].astype(int)
#                 dates_ok = block_dates[m_blk]

#             else:
#                 # slower fallback
#                 X_tr, ok_tr_pos = build_mantis_X_block(feats_raw, train_pos, context_len=mantis_context_len)
#                 if X_tr is None:
#                     i = j
#                     continue
#                 y_tr = y_np[ok_tr_pos]
#                 m = np.isfinite(y_tr)
#                 X_tr = X_tr[m]
#                 y_train = y_tr[m].astype(int)

#                 if mantis_max_train_windows is not None and len(y_train) > mantis_max_train_windows:
#                     X_tr = X_tr[-mantis_max_train_windows:]
#                     y_train = y_train[-mantis_max_train_windows:]

#                 if len(y_train) < min_train:
#                     i = j
#                     continue

#                 with torch.inference_mode():
#                     Z_train = np.asarray(mantis_trainer.transform(X_tr), dtype=np.float32)

#                 X_blk, ok_blk_pos = build_mantis_X_block(feats_raw, block_pos, context_len=mantis_context_len)
#                 if X_blk is None:
#                     i = j
#                     continue

#                 with torch.inference_mode():
#                     Z_test = np.asarray(mantis_trainer.transform(X_blk), dtype=np.float32)

#                 y_test = y_np[ok_blk_pos].astype(int)
#                 dates_ok = df.index[ok_blk_pos]

#             if not ((0 in np.unique(y_train)) and (1 in np.unique(y_train))):
#                 i = j
#                 continue

#             rf_n_jobs = 1 if bootstrap_n_jobs != 1 else -1

#             def _one_boot(b: int):
#                 rng = np.random.default_rng(seed + 100000 * i + b)
#                 idx = _sample_boot_idx_binary(
#                     y_train, rng,
#                     method=bootstrap_method,
#                     block_len=bootstrap_block_len,
#                 )

#                 if model == "mantis_head":
#                     head = Pipeline([
#                         ("scaler", StandardScaler()),
#                         ("clf", LogisticRegression(
#                             max_iter=2000,
#                             class_weight="balanced",
#                             solver="lbfgs",
#                             random_state=seed + 100000 * i + b,
#                         )),
#                     ])
#                     head.fit(Z_train[idx], y_train[idx])
#                 else:
#                     head = RandomForestClassifier(
#                         n_estimators=200,
#                         max_depth=None,
#                         min_samples_leaf=5,
#                         max_features="sqrt",
#                         random_state=seed + 100000 * i + b,
#                         n_jobs=rf_n_jobs,
#                         class_weight="balanced_subsample",
#                     )
#                     head.fit(Z_train[idx], y_train[idx])

#                 p1 = _predict_p1(head, Z_test)
#                 return p1

#         else:
#             raise ValueError("Use: logit, rf, tabpfn25, mantis_head, mantis_rf_head")

#         # -------------------------
#         # Run bootstrap fits (produce p1 draws)
#         # -------------------------
#         if bootstrap_n_jobs == 1:
#             P1_boot = np.stack([_one_boot(b) for b in range(n_boot)], axis=0)  # (B, N)
#         else:
#             P1_list = Parallel(n_jobs=bootstrap_n_jobs, backend="loky")(
#                 delayed(_one_boot)(b) for b in range(n_boot)
#             )
#             P1_boot = np.stack(P1_list, axis=0)

#         # -------------------------
#         # Store "all predictions" in long form
#         # -------------------------
#         # rows for each boot draw and timestamp
#         # (this can be large: N_dates * n_boot)
#         for b in range(n_boot):
#             p1b = P1_boot[b]
#             ypb = (p1b >= threshold).astype(int)
#             for t, yt, p1, yp in zip(dates_ok, y_test, p1b, ypb):
#                 rows_draws.append({
#                     "timestamp": t,
#                     "boot_id": b,
#                     "y_true": int(yt),
#                     "p1": float(p1),
#                     "y_pred_draw": int(yp),
#                 })

#         # -------------------------
#         # Summary per timestamp
#         # -------------------------
#         p_mean = P1_boot.mean(axis=0)
#         p_lo = np.quantile(P1_boot, alpha / 2.0, axis=0)
#         p_hi = np.quantile(P1_boot, 1.0 - alpha / 2.0, axis=0)
#         y_pred = (p_mean >= threshold).astype(int)

#         for t, yt, pm, plo, phi, yp in zip(dates_ok, y_test, p_mean, p_lo, p_hi, y_pred):
#             rows_summary.append({
#                 "timestamp": t,
#                 "y_true": int(yt),
#                 "p_mean": float(pm),
#                 "p_lo": float(plo),
#                 "p_hi": float(phi),
#                 "y_pred": int(yp),
#             })

#         if not quiet:
#             prev = df.index[pos0 - 1].date() if pos0 > 0 else "N/A"
#             print(f"[{model}+bootstrap] refit at {date_t.date()} using data up to {prev} | predicted {len(dates_ok)} days | B={n_boot}")

#         i = j

#     pred_df = pd.DataFrame(rows_summary).set_index("timestamp").sort_index()
#     pred_draws_df = pd.DataFrame(rows_draws).set_index(["timestamp", "boot_id"]).sort_index()
#     return pred_df, pred_draws_df



# ==========================================================
# COMPLETE CODE: classification OOS bootstrap + trading CIs  #Changed
# - Keeps your original expanding_oos_refit_every_cls_bootstrap_binary logic
# - Adds (optional) trading backtest + bootstrap CIs INSIDE the function
# - Returns:
#     pred_df, pred_draws_df, boot_metrics_df, trading_summary
#   (if compute_trading=False -> returns only pred_df, pred_draws_df)
# ==========================================================

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import os, sys
sys.path.insert(0, os.path.abspath('../'))

from source.Classification.models_classification import (
    make_lag1_features,
    build_mantis_X_block,
    precompute_mantis_embeddings,
    set_global_seed,
    ensure_datetime_index_from_timestamp
)

# ==========================================================
# METRICS HELPERS
# ==========================================================

def summarize_bootstrap(vals, ci=0.90):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "ci": float(ci), "n": 0}
    a = (1.0 - ci) / 2.0
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "ci_lower": float(np.quantile(vals, a)),
        "ci_upper": float(np.quantile(vals, 1 - a)),
        "ci": float(ci),
        "n": int(len(vals)),
    }

def perf_stats(total_returns, excess_returns=None, periods_per_year=252):
    rt = pd.Series(total_returns).dropna()
    if len(rt) < 2:
        return {}

    wealth = (1 + rt).cumprod()
    total_ret = wealth.iloc[-1] - 1
    cagr = wealth.iloc[-1] ** (periods_per_year / len(rt)) - 1
    ann_vol = rt.std(ddof=0) * np.sqrt(periods_per_year)

    if excess_returns is None:
        excess_returns = rt
    re = pd.Series(excess_returns).dropna()

    sharpe = np.nan
    if len(re) > 1:
        std = re.std(ddof=0)
        sharpe = (re.mean() / std) * np.sqrt(periods_per_year) if std > 0 else np.nan

    peak = wealth.cummax()
    max_dd = (wealth / peak - 1).min()

    return {
        "TotalReturn": float(total_ret),
        "CAGR": float(cagr),
        "AnnVol": float(ann_vol),
        "Sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "MaxDrawdown": float(max_dd),
    }

def ann_utility(excess_returns, gamma=5.0, periods_per_year=252):
    r = pd.Series(excess_returns).dropna()
    if len(r) < 2:
        return np.nan
    mean_r = r.mean()
    var_r  = r.var(ddof=0)
    u_per = mean_r - (gamma / 2.0) * var_r
    return float(periods_per_year * u_per)

# ==========================================================
# BACKTEST: regime switching + baselines (same as your paper)
# ==========================================================

def backtest_paper_regime_switch(
    df: pd.DataFrame,
    price_col: str,
    regime_col: str = "pred_regime",   # 1=bear (cash), 0=bull (equity)
    rf_col: str | None = None,
    rf_const: float = 0.0,
    tc_bps: float = 0.0,
    bear_label: str = "bear",
):
    d = df.copy()
    d = d.dropna(subset=[price_col, regime_col]).copy()

    price = pd.to_numeric(d[price_col], errors="coerce")
    r_eq_fwd = price.pct_change().shift(-1)

    if rf_col is not None and rf_col in d.columns:
        rf = pd.to_numeric(d[rf_col], errors="coerce").fillna(rf_const)
    else:
        rf = pd.Series(rf_const, index=d.index, dtype=float)
    r_rf_fwd = rf.shift(-1)

    reg = d[regime_col]
    if reg.dtype == "O":
        is_bear = reg.astype(str).str.lower().eq(str(bear_label).lower())
    else:
        is_bear = reg.astype(float).fillna(0.0).astype(int).eq(1)

    w = (~is_bear).astype(float)  # bull=1 equity, bear=0 cash

    turnover = w.diff().abs().fillna(0.0)
    cost = (tc_bps / 10000.0) * turnover

    strat_gross = w * r_eq_fwd + (1 - w) * r_rf_fwd
    strat_net = strat_gross - cost

    out = pd.DataFrame(index=d.index)
    out["strategy_net"] = strat_net
    out["w"] = w
    out["turnover"] = turnover

    # baselines
    out["buy_hold_eq"] = r_eq_fwd
    out["buy_hold_rf"] = r_rf_fwd
    out["static_50_50"] = 0.5 * r_eq_fwd + 0.5 * r_rf_fwd

    out = out.dropna()

    return out

# ==========================================================
# SUMMARY TABLE: includes HA (majority vote) + W50/W100
# ==========================================================

def compare_regime_strategies_with_HA(
    bt_model: pd.DataFrame,
    bt_ha: pd.DataFrame,
    *,
    periods_per_year: int = 252,
    gamma: float = 5.0,
):
    """
    Builds a unified summary with rows:
      Model, HA (majority vote), W50, W100

    Utility uses excess returns over BuyHoldRF (risk-free), consistent with your other function.
    Δu keys match regression naming:
      Delta_u_vs_HA, Delta_u_vs_W50, Delta_u_vs_W100
    """
    # common sample across all series
    R = pd.DataFrame({
        "Model": bt_model["strategy_net"].astype(float),
        "HA": bt_ha["strategy_net"].astype(float),
        "W50": bt_model["static_50_50"].astype(float),
        "W100": bt_model["buy_hold_eq"].astype(float),
        "RF": bt_model["buy_hold_rf"].astype(float),
    }).dropna(how="any")

    if R.empty or len(R) < 20:
        return pd.DataFrame()

    rf = R["RF"]
    RE = R.sub(rf, axis=0)  # excess returns vs RF for all strategies

    rows = []
    for col in ["Model", "HA", "W50", "W100"]:
        stats = perf_stats(R[col], RE[col], periods_per_year=periods_per_year)
        stats["CER_ann"] = ann_utility(RE[col], gamma=gamma, periods_per_year=periods_per_year)
        stats["Strategy"] = col
        rows.append(stats)

    summary = pd.DataFrame(rows).set_index("Strategy")

    # Δu (GW-style) on the Model row
    summary["Delta_u_vs_HA"] = np.nan
    summary["Delta_u_vs_W50"] = np.nan
    summary["Delta_u_vs_W100"] = np.nan
    summary.loc["Model", "Delta_u_vs_HA"] = summary.loc["Model", "CER_ann"] - summary.loc["HA", "CER_ann"]
    summary.loc["Model", "Delta_u_vs_W50"] = summary.loc["Model", "CER_ann"] - summary.loc["W50", "CER_ann"]
    summary.loc["Model", "Delta_u_vs_W100"] = summary.loc["Model", "CER_ann"] - summary.loc["W100", "CER_ann"]

    # Rename Sharpe to SR_ann like your regression dict output
    if "Sharpe" in summary.columns:
        summary = summary.rename(columns={"Sharpe": "SR_ann"})

    return summary

# ==========================================================
# BOOTSTRAP INDEX GENERATORS (yours)
# ==========================================================

def _boot_idx_iid(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n, endpoint=False)

def _boot_idx_block(n: int, rng: np.random.Generator, block_len: int) -> np.ndarray:
    if block_len <= 1:
        return _boot_idx_iid(n, rng)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks, endpoint=False)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n] % n
    return idx

def _boot_idx(n: int, rng: np.random.Generator, method: str, block_len: int) -> np.ndarray:
    if method == "iid":
        return _boot_idx_iid(n, rng)
    if method == "block":
        return _boot_idx_block(n, rng, block_len)
    raise ValueError("bootstrap_method must be 'iid' or 'block'")

def _ensure_both_classes(y: np.ndarray, idx: np.ndarray) -> bool:
    u = np.unique(y[idx])
    return (0 in u) and (1 in u)

def _sample_boot_idx_binary(
    y_train: np.ndarray,
    rng: np.random.Generator,
    *,
    method: str,
    block_len: int,
    max_tries: int = 30,
) -> np.ndarray:
    n = len(y_train)
    for _ in range(max_tries):
        idx = _boot_idx(n, rng, method, block_len)
        if _ensure_both_classes(y_train, idx):
            return idx
    return np.arange(n)

# ==========================================================
# FIT/PREDICT HELPERS (yours)
# ==========================================================

def _fit_binary_model(
    model: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    device: str | None = None,
    rf_n_jobs: int = -1,
):
    if model == "logit":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )),
        ])
        clf.fit(X_train, y_train)
        return clf

    if model == "rf":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=seed,
            n_jobs=rf_n_jobs,
            class_weight="balanced_subsample",
        )
        clf.fit(X_train, y_train)
        return clf

    if model == "tabpfn25":
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(device=device)
        clf.fit(X_train.astype(np.float32), y_train.astype(int))
        return clf

    raise ValueError("model must be 'logit', 'rf', or 'tabpfn25' for tabular bootstrap.")

def _predict_p1(model_obj, X: np.ndarray) -> np.ndarray:
    if hasattr(model_obj, "predict_proba"):
        P = model_obj.predict_proba(X)
        classes = model_obj.classes_
        j1 = int(np.where(classes == 1)[0][0])
        return P[:, j1].astype(float)
    yhat = model_obj.predict(X).astype(int)
    return yhat.astype(float)

def _majority_vote_class(y_train: np.ndarray) -> int:
    """
    Majority vote baseline on the training sample (real-time up to t-1).
    Tie-break -> bull (0), so we choose 0 if counts equal.
    """
    y = np.asarray(y_train, int)
    c0 = int(np.sum(y == 0))
    c1 = int(np.sum(y == 1))
    return 1 if c1 > c0 else 0

# ==========================================================
# MAIN: expanding OOS + bootstrap refit + trading CIs
# ==========================================================

def expanding_oos_refit_every_cls_bootstrap_binary(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str = "state",
    start_oos: str = "2010-01-01",
    start_date: str = "2000-01-05",
    min_train: int = 120,
    refit_every: int = 30,
    model: str = "logit",  # "logit", "rf", "tabpfn25", "mantis_head", "mantis_rf_head"
    mantis_context_len: int = 512,
    seed: int = 42,
    device: str | None = None,
    max_train: int | None = None,
    mantis_max_train_windows: int | None = None,
    # bootstrap controls
    n_boot: int = 200,
    bootstrap_method: str = "block",
    bootstrap_block_len: int | None = None,
    alpha: float = 0.05,
    threshold: float = 0.5,
    bootstrap_n_jobs: int = 1,
    quiet: bool = False,
    # mantis speed
    mantis_cache_embeddings: bool = True,
    mantis_embed_chunk: int = 256,

    # -------- TRADING (NEW) --------
    compute_trading: bool = True,
    price_col: str | None = None,   #Changed  <-- REQUIRED if compute_trading=True
    rf_col: str | None = None,      #Changed  <-- optional
    rf_const: float = 0.0,
    tc_bps: float = 0.0,
    periods_per_year: int = 252,
    gamma: float = 5.0,
    trading_ci: float = 0.90,
    bear_label: str = "bear",       #Changed  <-- only matters if your regimes are strings
):
    """
    Returns:
      if compute_trading=False:
        pred_df, pred_draws_df
      else:
        pred_df, pred_draws_df, boot_metrics_df, trading_summary_dict
    """

    set_global_seed(seed)

    df = ensure_datetime_index_from_timestamp(data, ts_col="timestamp")
    df = df.loc[pd.Timestamp(start_date):].copy()

    if compute_trading:
        if price_col is None or price_col not in df.columns:
            raise ValueError("compute_trading=True requires valid price_col (price level series).  #Changed")
        if rf_col is not None and rf_col not in df.columns:
            raise ValueError("rf_col not found in df.  #Changed")

    start_ts = pd.Timestamp(start_oos)
    loop_dates = df.index[df.index >= start_ts]
    if len(loop_dates) == 0:
        empty = pd.DataFrame().set_index(pd.DatetimeIndex([]))
        if compute_trading:
            return empty, empty, empty, {}
        return empty, empty

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if bootstrap_block_len is None:
        bootstrap_block_len = refit_every

    # shared arrays
    X_lag = make_lag1_features(df, feature_cols)
    X_lag_np = X_lag[feature_cols].to_numpy(dtype=np.float32)
    y_np = df[target_col].to_numpy(dtype=float)
    feats_raw = df[feature_cols].to_numpy(dtype=float)

    idx_map = pd.Series(np.arange(len(df.index)), index=df.index)
    loop_pos = idx_map.loc[loop_dates].to_numpy(dtype=int)

    # Mantis precompute if needed
    Z_all, ok_embed, mantis_trainer = None, None, None
    if model in ("mantis_head", "mantis_rf_head"):
        from mantis.architecture import Mantis8M
        from mantis.trainer import MantisTrainer

        mantis_network = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
        mantis_network.eval()
        mantis_trainer = MantisTrainer(device=device, network=mantis_network)

        if mantis_cache_embeddings:
            with torch.inference_mode():
                Z_all, ok_embed = precompute_mantis_embeddings(
                    mantis_trainer,
                    feats_raw=feats_raw,
                    context_len=mantis_context_len,
                    chunk_size=mantis_embed_chunk,
                )

    rows_summary = []
    rows_draws = []

    # For trading returns collection (per boot)
    boot_returns_model = []  # rows: timestamp, boot_id, strategy_net, buy_hold_eq, buy_hold_rf, static_50_50
    boot_returns_ha = []     # rows: timestamp, boot_id, strategy_net (HA baseline), same baselines

    i = 0
    while i < len(loop_dates):
        pos0 = loop_pos[i]
        date_t = loop_dates[i]
        j = min(i + refit_every, len(loop_dates))
        block_pos = loop_pos[i:j]
        block_dates = df.index[block_pos]

        # -------------------------
        # Build training + test sets
        # -------------------------
        if model in ("logit", "rf", "tabpfn25"):
            X_train = X_lag_np[:pos0]
            y_train = y_np[:pos0]
            m = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
            X_train = X_train[m]
            y_train = y_train[m].astype(int)

            if max_train is not None and len(y_train) > max_train:
                X_train = X_train[-max_train:]
                y_train = y_train[-max_train:]

            if len(y_train) < min_train:
                i = j
                continue

            X_blk = X_lag_np[block_pos]
            y_blk = y_np[block_pos]
            ok = np.isfinite(X_blk).all(axis=1) & np.isfinite(y_blk)
            if not ok.any():
                i = j
                continue

            X_test = X_blk[ok]
            y_test = y_blk[ok].astype(int)
            dates_ok = block_dates[ok]

            if not ((0 in np.unique(y_train)) and (1 in np.unique(y_train))):
                i = j
                continue

            rf_n_jobs = 1 if bootstrap_n_jobs != 1 else -1

            def _one_boot(b: int):
                rng = np.random.default_rng(seed + 100000 * i + b)
                idx = _sample_boot_idx_binary(
                    y_train, rng,
                    method=bootstrap_method,
                    block_len=bootstrap_block_len,
                )
                mdl = _fit_binary_model(
                    model, X_train[idx], y_train[idx],
                    seed=seed + 100000 * i + b,
                    device=device,
                    rf_n_jobs=rf_n_jobs,
                )
                p1 = _predict_p1(mdl, X_test)
                return p1

        elif model in ("mantis_head", "mantis_rf_head"):
            train_pos = np.arange(mantis_context_len, pos0, dtype=int)
            if len(train_pos) == 0:
                i = j
                continue

            if mantis_cache_embeddings:
                m_tr = ok_embed[train_pos] & np.isfinite(y_np[train_pos])
                Z_train = Z_all[train_pos][m_tr]
                y_train = y_np[train_pos][m_tr].astype(int)

                if mantis_max_train_windows is not None and len(y_train) > mantis_max_train_windows:
                    Z_train = Z_train[-mantis_max_train_windows:]
                    y_train = y_train[-mantis_max_train_windows:]

                if len(y_train) < min_train:
                    i = j
                    continue

                m_blk = ok_embed[block_pos] & np.isfinite(y_np[block_pos])
                if not m_blk.any():
                    i = j
                    continue

                Z_test = Z_all[block_pos][m_blk]
                y_test = y_np[block_pos][m_blk].astype(int)
                dates_ok = block_dates[m_blk]

            else:
                X_tr, ok_tr_pos = build_mantis_X_block(feats_raw, train_pos, context_len=mantis_context_len)
                if X_tr is None:
                    i = j
                    continue
                y_tr = y_np[ok_tr_pos]
                m = np.isfinite(y_tr)
                X_tr = X_tr[m]
                y_train = y_tr[m].astype(int)

                if mantis_max_train_windows is not None and len(y_train) > mantis_max_train_windows:
                    X_tr = X_tr[-mantis_max_train_windows:]
                    y_train = y_train[-mantis_max_train_windows:]

                if len(y_train) < min_train:
                    i = j
                    continue

                with torch.inference_mode():
                    Z_train = np.asarray(mantis_trainer.transform(X_tr), dtype=np.float32)

                X_blk, ok_blk_pos = build_mantis_X_block(feats_raw, block_pos, context_len=mantis_context_len)
                if X_blk is None:
                    i = j
                    continue

                with torch.inference_mode():
                    Z_test = np.asarray(mantis_trainer.transform(X_blk), dtype=np.float32)

                y_test = y_np[ok_blk_pos].astype(int)
                dates_ok = df.index[ok_blk_pos]

            if not ((0 in np.unique(y_train)) and (1 in np.unique(y_train))):
                i = j
                continue

            rf_n_jobs = 1 if bootstrap_n_jobs != 1 else -1

            def _one_boot(b: int):
                rng = np.random.default_rng(seed + 100000 * i + b)
                idx = _sample_boot_idx_binary(
                    y_train, rng,
                    method=bootstrap_method,
                    block_len=bootstrap_block_len,
                )

                if model == "mantis_head":
                    head = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            solver="lbfgs",
                            random_state=seed + 100000 * i + b,
                        )),
                    ])
                    head.fit(Z_train[idx], y_train[idx])
                else:
                    head = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        random_state=seed + 100000 * i + b,
                        n_jobs=rf_n_jobs,
                        class_weight="balanced_subsample",
                    )
                    head.fit(Z_train[idx], y_train[idx])

                p1 = _predict_p1(head, Z_test)
                return p1

        else:
            raise ValueError("Use: logit, rf, tabpfn25, mantis_head, mantis_rf_head")

        # -------------------------
        # Run bootstrap fits (produce p1 draws)
        # -------------------------
        if bootstrap_n_jobs == 1:
            P1_boot = np.stack([_one_boot(b) for b in range(n_boot)], axis=0)  # (B, N)
        else:
            P1_list = Parallel(n_jobs=bootstrap_n_jobs, backend="loky")(
                delayed(_one_boot)(b) for b in range(n_boot)
            )
            P1_boot = np.stack(P1_list, axis=0)

        # -------------------------
        # Store long-form predictions
        # -------------------------
        for b in range(n_boot):
            p1b = P1_boot[b]
            ypb = (p1b >= threshold).astype(int)
            for t, yt, p1, yp in zip(dates_ok, y_test, p1b, ypb):
                rows_draws.append({
                    "timestamp": t,
                    "boot_id": b,
                    "y_true": int(yt),
                    "p1": float(p1),
                    "y_pred_draw": int(yp),
                })

        # -------------------------
        # Summary per timestamp
        # -------------------------
        p_mean = P1_boot.mean(axis=0)
        p_lo = np.quantile(P1_boot, alpha / 2.0, axis=0)
        p_hi = np.quantile(P1_boot, 1.0 - alpha / 2.0, axis=0)
        y_pred = (p_mean >= threshold).astype(int)

        for t, yt, pm, plo, phi, yp in zip(dates_ok, y_test, p_mean, p_lo, p_hi, y_pred):
            rows_summary.append({
                "timestamp": t,
                "y_true": int(yt),
                "p_mean": float(pm),
                "p_lo": float(plo),
                "p_hi": float(phi),
                "y_pred": int(yp),
            })

        # -------------------------
        # TRADING: Model + HA majority-vote baseline
        # -------------------------
        if compute_trading:
            d_block = df.loc[dates_ok, [price_col] + ([rf_col] if rf_col is not None else [])].copy()  #Changed
            # compute HA/majority vote using training info up to t-1 (real-time)
            maj = _majority_vote_class(y_train)

            for b in range(n_boot):
                # ---- Model strategy
                ypb = (P1_boot[b] >= threshold).astype(int)
                d_m = d_block.copy()
                d_m["pred_regime"] = ypb
                bt_m = backtest_paper_regime_switch(
                    d_m,
                    price_col=price_col,
                    regime_col="pred_regime",
                    rf_col=rf_col,
                    rf_const=rf_const,
                    tc_bps=tc_bps,
                    bear_label=bear_label,   #Changed
                )
                bt_m_small = bt_m[["strategy_net", "buy_hold_eq", "buy_hold_rf", "static_50_50"]].copy()
                bt_m_small["boot_id"] = b
                bt_m_small = bt_m_small.reset_index().rename(columns={"index": "timestamp"})
                boot_returns_model.append(bt_m_small)

                # ---- HA (majority vote) strategy
                d_h = d_block.copy()
                d_h["pred_regime"] = maj
                bt_h = backtest_paper_regime_switch(
                    d_h,
                    price_col=price_col,
                    regime_col="pred_regime",
                    rf_col=rf_col,
                    rf_const=rf_const,
                    tc_bps=tc_bps,
                    bear_label=bear_label,   #Changed
                )
                bt_h_small = bt_h[["strategy_net", "buy_hold_eq", "buy_hold_rf", "static_50_50"]].copy()
                bt_h_small["boot_id"] = b
                bt_h_small = bt_h_small.reset_index().rename(columns={"index": "timestamp"})
                boot_returns_ha.append(bt_h_small)

        if not quiet:
            prev = df.index[pos0 - 1].date() if pos0 > 0 else "N/A"
            print(f"[{model}+bootstrap] refit at {date_t.date()} using data up to {prev} | predicted {len(dates_ok)} days | B={n_boot}")

        i = j

    pred_df = pd.DataFrame(rows_summary).set_index("timestamp").sort_index()
    pred_draws_df = pd.DataFrame(rows_draws).set_index(["timestamp", "boot_id"]).sort_index()

    if not compute_trading:
        return pred_df, pred_draws_df

    # ==========================================================
    # Aggregate trading metrics per boot_id
    # ==========================================================

    if len(boot_returns_model) == 0 or len(boot_returns_ha) == 0:
        return pred_df, pred_draws_df, pd.DataFrame(), {}

    ret_m = pd.concat(boot_returns_model, axis=0)
    ret_h = pd.concat(boot_returns_ha, axis=0)

    ret_m["timestamp"] = pd.to_datetime(ret_m["timestamp"])
    ret_h["timestamp"] = pd.to_datetime(ret_h["timestamp"])
    ret_m = ret_m.set_index(["timestamp", "boot_id"]).sort_index()
    ret_h = ret_h.set_index(["timestamp", "boot_id"]).sort_index()

    metrics_rows = []
    for b in range(n_boot):
        if b not in ret_m.index.get_level_values("boot_id"):
            continue
        if b not in ret_h.index.get_level_values("boot_id"):
            continue

        bt_m = ret_m.xs(b, level="boot_id").sort_index().dropna(how="any")
        bt_h = ret_h.xs(b, level="boot_id").sort_index().dropna(how="any")

        # align common sample
        idx = bt_m.index.intersection(bt_h.index)
        bt_m = bt_m.loc[idx]
        bt_h = bt_h.loc[idx]

        if len(idx) < 20:
            continue

        summ = compare_regime_strategies_with_HA(
            bt_model=bt_m,
            bt_ha=bt_h,
            periods_per_year=periods_per_year,
            gamma=gamma,
        )
        if summ is None or summ.empty or "Model" not in summ.index:
            continue

        metrics_rows.append({
            "boot_id": b,
            "SR_ann": float(summ.loc["Model", "SR_ann"]) if "SR_ann" in summ.columns else np.nan,
            "CER_ann": float(summ.loc["Model", "CER_ann"]) if "CER_ann" in summ.columns else np.nan,
            "AnnVol": float(summ.loc["Model", "AnnVol"]) if "AnnVol" in summ.columns else np.nan,
            "TotalReturn": float(summ.loc["Model", "TotalReturn"]) if "TotalReturn" in summ.columns else np.nan,  #Changed
            "CAGR": float(summ.loc["Model", "CAGR"]) if "CAGR" in summ.columns else np.nan,  #Changed
            "Delta_u_vs_HA": float(summ.loc["Model", "Delta_u_vs_HA"]) if "Delta_u_vs_HA" in summ.columns else np.nan,
            "Delta_u_vs_W50": float(summ.loc["Model", "Delta_u_vs_W50"]) if "Delta_u_vs_W50" in summ.columns else np.nan,
            "Delta_u_vs_W100": float(summ.loc["Model", "Delta_u_vs_W100"]) if "Delta_u_vs_W100" in summ.columns else np.nan,
        })

    boot_metrics_df = pd.DataFrame(metrics_rows).set_index("boot_id").sort_index()

    ci = trading_ci
    trading_summary = {
        "SR_ann": summarize_bootstrap(boot_metrics_df["SR_ann"].to_numpy(), ci=ci),
        "CEV_ann": summarize_bootstrap(boot_metrics_df["CER_ann"].to_numpy(), ci=ci),
        "AnnVol": summarize_bootstrap(boot_metrics_df["AnnVol"].to_numpy(), ci=ci),
        "TotalReturn": summarize_bootstrap(boot_metrics_df["TotalReturn"].to_numpy(), ci=ci),  #Changed
        "CAGR": summarize_bootstrap(boot_metrics_df["CAGR"].to_numpy(), ci=ci),  #Changed
        "Delta_u_vs_HA": summarize_bootstrap(boot_metrics_df["Delta_u_vs_HA"].to_numpy(), ci=ci),
        "Delta_u_vs_W50": summarize_bootstrap(boot_metrics_df["Delta_u_vs_W50"].to_numpy(), ci=ci),
        "Delta_u_vs_W100": summarize_bootstrap(boot_metrics_df["Delta_u_vs_W100"].to_numpy(), ci=ci),
    }

    return pred_df, pred_draws_df, boot_metrics_df, trading_summary
