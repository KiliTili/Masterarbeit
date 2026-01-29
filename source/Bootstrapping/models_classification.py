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




# -----------------------
# Bootstrap index generators (binary TS-safe)
# -----------------------
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
def expanding_oos_refit_every_cls_bootstrap_binary(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str = "state",
    start_oos: str = "2007-01-01",
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
    bootstrap_method: str = "block",     # "block" recommended
    bootstrap_block_len: int | None = None,
    alpha: float = 0.05,
    threshold: float = 0.5,
    bootstrap_n_jobs: int = 1,
    quiet: bool = False,
    # mantis speed
    mantis_cache_embeddings: bool = True,
    mantis_embed_chunk: int = 256,
):
    """
    Returns:
      pred_df: index timestamp with y_true, p_mean, p_lo, p_hi, y_pred
      pred_draws_df: index (timestamp, boot_id) with y_true, p1, y_pred_draw
    """
    set_global_seed(seed)

    df = ensure_datetime_index_from_timestamp(data, ts_col="timestamp")
    df = df.loc[pd.Timestamp(start_date):].copy()

    start_ts = pd.Timestamp(start_oos)
    loop_dates = df.index[df.index >= start_ts]
    if len(loop_dates) == 0:
        empty = pd.DataFrame().set_index(pd.DatetimeIndex([]))
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
    Z_all, ok_embed, mantis_trainer, head_kind = None, None, None, None
    if model in ("mantis_head", "mantis_rf_head"):
        from mantis.architecture import Mantis8M
        from mantis.trainer import MantisTrainer

        mantis_network = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
        mantis_network.eval()
        mantis_trainer = MantisTrainer(device=device, network=mantis_network)
        head_kind = "logit" if model == "mantis_head" else "rf"

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

            # ensure binary present in training at all
            if not ((0 in np.unique(y_train)) and (1 in np.unique(y_train))):
                i = j
                continue

            # if we parallelize bootstraps, avoid RF nested parallel
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
                # slower fallback
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
        # Store "all predictions" in long form
        # -------------------------
        # rows for each boot draw and timestamp
        # (this can be large: N_dates * n_boot)
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

        if not quiet:
            prev = df.index[pos0 - 1].date() if pos0 > 0 else "N/A"
            print(f"[{model}+bootstrap] refit at {date_t.date()} using data up to {prev} | predicted {len(dates_ok)} days | B={n_boot}")

        i = j

    pred_df = pd.DataFrame(rows_summary).set_index("timestamp").sort_index()
    pred_draws_df = pd.DataFrame(rows_draws).set_index(["timestamp", "boot_id"]).sort_index()
    return pred_df, pred_draws_df