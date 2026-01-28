# Masterarbeti

def expanding_oos_refit_every_cls( # existing
data: pd.DataFrame,
\*,
feature_cols: list[str],
target_col: str = "state",
start_oos: str = "2007-01-01",
start_date: str = "2000-01-05",
min_train: int = 120,
refit_every: int = 30,
model: str = "logit",
mantis_context_len: int = 60,
seed: int = 42,
device: str | None = None,
quiet: bool = False,
max_train: int | None = None,
mantis_max_train_windows: int | None = None, #changed
): # ... keep your existing imports + setup exactly as in your last version ...

    # (keep everything up to the TabPFN fast path unchanged)

    # ============================================================
    # FAST PATH: Mantis refit every `refit_every` and predict block
    # ============================================================
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

        return np.asarray(y_true_list, int), np.asarray(y_pred_list, int), pd.DatetimeIndex(date_list)  #changed

    # ... keep your existing STANDARD PATH for the other models unchanged ...
