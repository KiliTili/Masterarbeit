import numpy as np
import pandas as pd
import random
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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
    # X_t uses info up to t-1
    return df[feature_cols].shift(1)


# -----------------------
# RF helper (simple + fast)
# -----------------------
def make_default_rf(seed: int = 42, n_jobs: int = -1):
    # Your tune_rf_time_split had dead/unreachable code; keeping a solid default.
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced_subsample",
    )


# -----------------------
# Mantis data builder (no look-ahead)
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


def build_mantis_X_block(
    feats: np.ndarray,          # (T, C)
    positions: np.ndarray,      # positions p where we want X from feats[p-context_len:p]
    context_len: int,
):
    """
    Returns:
      X: (N, C, 512) float32
      ok_positions: (N,) int positions that were valid
    """
    X_list = []
    ok_pos = []
    for p in positions.tolist():
        if p < context_len:
            continue
        w = feats[p - context_len:p]  # (L, C)
        if not np.isfinite(w).all():
            continue
        X_list.append(w.T)  # (C, L)
        ok_pos.append(p)

    if len(X_list) == 0:
        return None, None

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, C, L)
    X = mantis_resize_to_512(X)  # (N, C, 512)
    return X, np.asarray(ok_pos, dtype=int)


def precompute_mantis_embeddings(
    mantis_trainer,
    feats_raw: np.ndarray,      # (T, C)
    context_len: int,
    chunk_size: int = 256,
):
    """
    Precompute frozen-backbone embeddings Z[p] for all valid positions p >= context_len.
    Returns:
      Z_all: (T, D) float32 with NaN rows for invalid/uncomputed positions
      ok_embed: (T,) bool whether Z_all[p] is valid
    """
    T = feats_raw.shape[0]
    Z_all = None
    ok_embed = np.zeros(T, dtype=bool)

    positions = np.arange(context_len, T, dtype=int)

    for s in range(0, len(positions), chunk_size):
        pos_chunk = positions[s:s + chunk_size]
        X_chunk, ok_pos = build_mantis_X_block(feats_raw, pos_chunk, context_len=context_len)
        if X_chunk is None:
            continue

        Z_chunk = np.asarray(mantis_trainer.transform(X_chunk), dtype=np.float32)  # (N, D)

        if Z_all is None:
            D = Z_chunk.shape[1]
            Z_all = np.full((T, D), np.nan, dtype=np.float32)

        Z_all[ok_pos] = Z_chunk
        ok_embed[ok_pos] = True

    if Z_all is None:
        # no valid windows
        Z_all = np.full((T, 1), np.nan, dtype=np.float32)

    return Z_all, ok_embed


def fit_mantis_head_frozen(mantis_trainer, X_tr, y_tr, head: str, seed: int):
    """
    Frozen backbone. Train only the head on top of embeddings.
    head: "lr" or "rf"
    """
    Z_tr = np.asarray(mantis_trainer.transform(X_tr), dtype=np.float32)

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
        rf = make_default_rf(seed=seed, n_jobs=-1)
        rf.fit(Z_tr, y_tr)
        return rf

    raise ValueError("head must be 'lr' or 'rf'")


# -----------------------
# Main: expanding OOS with block refits (FAST)
# -----------------------
def expanding_oos_refit_every_cls(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str = "state",
    start_oos: str = "2007-01-01",
    start_date: str = "2000-01-05",
    min_train: int = 120,
    refit_every: int = 30,
    model: str = "logit",  # "logit", "rf", "tabpfn25", "mantis_head", "mantis_rf_head", "majority"
    mantis_context_len: int = 512,
    seed: int = 42,
    device: str | None = None,
    quiet: bool = False,
    max_train: int | None = None,
    mantis_max_train_windows: int | None = None,
    # new speed knobs:
    parallel_blocks: bool = False,     # only used for CPU models: logit/rf/majority
    n_jobs_outer: int = -1,            # joblib Parallel workers
    mantis_cache_embeddings: bool = True,
    mantis_embed_chunk: int = 256,
):
    """
    Fast version:
      - predicts in blocks for ALL models
      - for Mantis frozen backbone: optionally cache all embeddings once, then head trains/predicts from slices
      - optional block parallelism for CPU models (logit/rf/majority). If enabled, RF uses n_jobs=1 to avoid nested parallelism.
    """
    set_global_seed(seed)

    df = ensure_datetime_index_from_timestamp(data, ts_col="timestamp")
    df = df.loc[pd.Timestamp(start_date):].copy()

    start_ts = pd.Timestamp(start_oos)
    loop_dates = df.index[df.index >= start_ts]
    if len(loop_dates) == 0:
        return np.asarray([], int), np.asarray([], int), pd.DatetimeIndex([])

    # pick device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # lagged features (no look-ahead)
    X_lag = make_lag1_features(df, feature_cols)

    # numpy versions for speed
    X_lag_np = X_lag[feature_cols].to_numpy(dtype=np.float32)          # (T, F) with NaN
    y_np = df[target_col].to_numpy(dtype=float)                        # (T,) with NaN possible
    feats_raw = df[feature_cols].to_numpy(dtype=float) #Lookup since not sure                 # (T, F) raw feats for Mantis windows

    # map timestamps -> integer positions once (avoid get_loc in loops)
    idx_map = pd.Series(np.arange(len(df.index)), index=df.index)
    loop_pos = idx_map.loc[loop_dates].to_numpy(dtype=int)

    y_true_list, y_pred_list,y_prob, date_list =     [], [], [], []

    # -----------------------
    # CPU models: block worker (optionally parallel)
    # -----------------------
    def _fit_predict_block_cpu(block_start_i: int):
        # This function is pure CPU; safe for joblib parallel.
        i = block_start_i
        pos0 = loop_pos[i]
        date_t = loop_dates[i]
        print(f"CPU worker processing block starting at index {i}, date {date_t.date()}")
        # training strictly before pos0
        X_train = X_lag_np[:pos0]
        y_train = y_np[:pos0]

        m = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train = X_train[m]
        y_train = y_train[m].astype(int)

        if max_train is not None and len(y_train) > max_train:
            X_train = X_train[-max_train:]
            y_train = y_train[-max_train:]

        if len(y_train) < min_train:
            return (np.asarray([], int), np.asarray([], int), np.asarray([], "datetime64[ns]"), None)

        # fit
        set_global_seed(seed + i)

        if model == "logit":
            fitted = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed + i,
                )),
            ])
            fitted.fit(X_train, y_train)

        elif model == "rf":
            # avoid nested parallelism if we parallelize blocks outside
            rf_n_jobs = 1 if parallel_blocks else -1
            fitted = make_default_rf(seed=seed + i, n_jobs=rf_n_jobs)
            fitted.fit(X_train, y_train)

        elif model == "majority":
            vals, counts = np.unique(y_train, return_counts=True)
            maj = int(vals[np.argmax(counts)])
            fitted = maj

        else:
            raise ValueError("CPU worker called with non-CPU model")

        # predict block
        j = min(i + refit_every, len(loop_dates))
        block_pos = loop_pos[i:j]
        X_blk = X_lag_np[block_pos]
        y_blk = y_np[block_pos]

        ok = np.isfinite(X_blk).all(axis=1) & np.isfinite(y_blk)
        if not ok.any():
            return (np.asarray([], int), np.asarray([], int), np.asarray([], "datetime64[ns]"), (date_t, pos0, j - i))

        if model == "majority":
            preds = np.full(ok.sum(), fitted, dtype=int)
            prob = np.full(ok.sum(), 1.0 if fitted == 1 else 0.0, dtype=float)
        else:
            e = fitted.predict(X_blk[ok])
            preds = fitted.predict(X_blk[ok])
            prob = fitted.predict_proba(X_blk[ok])[:, 1]

        dates_ok = df.index[block_pos][ok].to_numpy(dtype="datetime64[ns]")
        y_true_ok = y_blk[ok].astype(int)

        return (y_true_ok, preds, prob, dates_ok, (date_t, pos0, j - i))

    # -----------------------
    # TABULAR CPU models (logit/rf/majority): blocked + optional parallel
    # -----------------------
    if model in ("logit", "rf", "majority"):
        block_starts = list(range(0, len(loop_dates), refit_every))

        if parallel_blocks and len(block_starts) > 1:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs_outer, backend="loky")(
                delayed(_fit_predict_block_cpu)(bi) for bi in block_starts
            )
        else:
            results = [_fit_predict_block_cpu(bi) for bi in block_starts]

        # stitch (already in chronological order)
        for (yt, yp,prob, dt, info) in results:
            if len(yt) == 0:
                continue
            y_true_list.extend(yt.tolist())
            y_pred_list.extend(yp.tolist())
            y_prob.extend(prob.tolist())
            date_list.extend(dt.tolist())

            if (not quiet) and info is not None:
                date_t, pos0, n_pred = info
                prev = df.index[pos0 - 1].date() if pos0 > 0 else "N/A"
                

        return np.asarray(y_true_list, int), np.asarray(y_pred_list, int),np.asarray(y_prob, float), pd.DatetimeIndex(date_list)

    # -----------------------
    # TabPFN (GPU/Metal/CPU): blocked sequential
    # -----------------------
    if model == "tabpfn25":
        from tabpfn import TabPFNClassifier

        i = 0
        while i < len(loop_dates):
            pos0 = loop_pos[i]
            date_t = loop_dates[i]

            X_train = X_lag_np[:pos0]
            y_train = y_np[:pos0]
            m = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
            X_train = X_train[m].astype(np.float32)
            y_train = y_train[m].astype(int)

            if max_train is not None and len(y_train) > max_train:
                X_train = X_train[-max_train:]
                y_train = y_train[-max_train:]

            if len(y_train) < min_train:
                i += 1
                continue

            set_global_seed(seed + i)
            clf = TabPFNClassifier(device=device)
            clf.fit(X_train, y_train)

            j = min(i + refit_every, len(loop_dates))
            block_pos = loop_pos[i:j]
            X_blk = X_lag_np[block_pos].astype(np.float32)
            y_blk = y_np[block_pos]

            ok = np.isfinite(X_blk).all(axis=1) & np.isfinite(y_blk)
            if ok.any():
                preds = clf.predict(X_blk[ok]).astype(int)
                probas = clf.predict(X_blk[ok])
                dates_ok = df.index[block_pos][ok]
                y_true_ok = y_blk[ok].astype(int)

                y_true_list.extend(y_true_ok.tolist())
                y_pred_list.extend(preds.tolist())
                y_prob.extend(probas.tolist())
                date_list.extend(dates_ok.tolist())

            if not quiet:
                prev = df.index[pos0 - 1].date() if pos0 > 0 else "N/A"
                print(f"[tabpfn25] refit at {date_t.date()} using data up to {prev} | predicted {j-i} days")

            i = j

        return np.asarray(y_true_list, int), np.asarray(y_pred_list, int),np.asarray(y_prob, float),  pd.DatetimeIndex(date_list)

    # -----------------------
    # Mantis (frozen backbone + head): blocked, with optional embedding cache
    # -----------------------
    if model in ("mantis_head", "mantis_rf_head"):
        from mantis.architecture import Mantis8M
        from mantis.trainer import MantisTrainer

        mantis_network = Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
        mantis_trainer = MantisTrainer(device=device, network=mantis_network)

        # Precompute embeddings once (big speed win)
        if mantis_cache_embeddings:
            Z_all, ok_embed = precompute_mantis_embeddings(
                mantis_trainer,
                feats_raw=feats_raw,
                context_len=mantis_context_len,
                chunk_size=mantis_embed_chunk,
            )
        else:
            Z_all, ok_embed = None, None  # fallback to per-block transform

        head_is_lr = (model == "mantis_head")

        i = 0
        while i < len(loop_dates):
            pos0 = loop_pos[i]
            date_t = loop_dates[i]

            # train positions must be strictly < pos0, and >= context_len
            train_pos = np.arange(mantis_context_len, pos0, dtype=int)
            if len(train_pos) == 0:
                i += 1
                continue

            if mantis_cache_embeddings:
                m_tr = ok_embed[train_pos] & np.isfinite(y_np[train_pos])
                Z_tr = Z_all[train_pos][m_tr]
                y_tr = y_np[train_pos][m_tr].astype(int)

                if mantis_max_train_windows is not None and len(y_tr) > mantis_max_train_windows:
                    Z_tr = Z_tr[-mantis_max_train_windows:]
                    y_tr = y_tr[-mantis_max_train_windows:]

                if len(y_tr) < min_train:
                    i += 1
                    continue

                set_global_seed(seed + i)

                if head_is_lr:
                    head_model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            solver="lbfgs",
                            random_state=seed + i,
                        )),
                    ])
                    head_model.fit(Z_tr, y_tr)
                else:
                    # RF on embeddings
                    rf_n_jobs = -1  # keep internal RF parallelism here; no outer parallel on GPU path
                    head_model = make_default_rf(seed=seed + i, n_jobs=rf_n_jobs)
                    head_model.fit(Z_tr, y_tr)

                # predict block from cached Z
                j = min(i + refit_every, len(loop_dates))
                block_pos = loop_pos[i:j]
                m_blk = ok_embed[block_pos] & np.isfinite(y_np[block_pos])

                if m_blk.any():
                    Z_blk = Z_all[block_pos][m_blk]
                    prob = head_model.predict_proba(Z_blk)[:, 1]
                    preds = head_model.predict(Z_blk).astype(int)

                    dates_ok = df.index[block_pos][m_blk]
                    y_true_ok = y_np[block_pos][m_blk].astype(int)

                    y_true_list.extend(y_true_ok.tolist())
                    y_pred_list.extend(preds.tolist())
                    date_list.extend(dates_ok.tolist())
                    y_prob.extend(prob.tolist())

                if not quiet:
                    prev = df.index[pos0 - 1].date() if pos0 > 0 else "N/A"
                    print(f"[{model}] refit at {date_t.date()} using data up to {prev} | predicted {j-i} days")

                i = j

            else:
                # fallback: per-block transform (slower; keeps your old behavior mostly)
                # Build training windows -> transform -> fit head
                X_tr, ok_tr_pos = build_mantis_X_block(
                    feats_raw, positions=train_pos, context_len=mantis_context_len
                )
                if X_tr is None:
                    i += 1
                    continue

                y_tr = y_np[ok_tr_pos]
                m = np.isfinite(y_tr)
                X_tr = X_tr[m]
                y_tr = y_tr[m].astype(int)

                if mantis_max_train_windows is not None and len(y_tr) > mantis_max_train_windows:
                    X_tr = X_tr[-mantis_max_train_windows:]
                    y_tr = y_tr[-mantis_max_train_windows:]

                if len(y_tr) < min_train:
                    i += 1
                    continue

                set_global_seed(seed + i)
                head_model = fit_mantis_head_frozen(
                    mantis_trainer, X_tr, y_tr,
                    head="lr" if head_is_lr else "rf",
                    seed=seed + i
                )

                # predict next block windows
                j = min(i + refit_every, len(loop_dates))
                block_pos = loop_pos[i:j]
                X_blk, ok_blk_pos = build_mantis_X_block(
                    feats_raw, positions=block_pos, context_len=mantis_context_len
                )
                if X_blk is not None and len(ok_blk_pos) > 0:
                    Z_blk = np.asarray(mantis_trainer.transform(X_blk), dtype=np.float32)
                    preds = head_model.predict(Z_blk).astype(int)

                    dates_ok = df.index[ok_blk_pos]
                    y_true_ok = y_np[ok_blk_pos].astype(int)

                    m_ok = np.isfinite(y_true_ok)
                    y_true_ok = y_true_ok[m_ok]
                    preds = preds[m_ok]
                    dates_ok = dates_ok[m_ok]

                    y_true_list.extend(y_true_ok.tolist())
                    y_pred_list.extend(preds.tolist())
                    date_list.extend(dates_ok.tolist())

                if not quiet:
                    prev = df.index[pos0 - 1].date() if pos0 > 0 else "N/A"
                    print(f"[{model}] refit at {date_t.date()} using data up to {prev} | predicted {j-i} days")

                i = j

        return np.asarray(y_true_list, int), np.asarray(y_pred_list, int), np.asarray(y_prob, float), pd.DatetimeIndex(date_list)

    raise ValueError(
        "Unknown model. Use: logit, rf, tabpfn25, mantis_head, mantis_rf_head, majority"
    )
