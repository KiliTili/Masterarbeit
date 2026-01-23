
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def _add_event_windows(ax, idx, events):  
    """Add shaded event windows (e.g., NBER recessions) to an axis."""  
    EVENT_COLOR = "red"   
    EVENT_ALPHA = 0.06    
    EVENT_LABEL = "NBER recessions"  
    if events is False or events is None:  
        return  

    _default_recessions = {  
        "Great Depression": ("1929-08-01", "1933-03-01"),
        "1937–38 recession": ("1937-05-01", "1938-06-01"),
        "1945 recession": ("1945-02-01", "1945-10-01"),
        "1948–49 recession": ("1948-11-01", "1949-10-01"),
        "1953–54 recession": ("1953-07-01", "1954-05-01"),
        "1957–58 recession": ("1957-08-01", "1958-04-01"),
        "1960–61 recession": ("1960-04-01", "1961-02-01"),
        "1969–70 recession": ("1969-12-01", "1970-11-01"),
        "Oil shock recession": ("1973-11-01", "1975-03-01"),
        "1980 recession": ("1980-01-01", "1980-07-01"),
        "1981–82 recession": ("1981-07-01", "1982-11-01"),
        "1990–91 recession": ("1990-07-01", "1991-03-01"),
        "2001 recession": ("2001-03-01", "2001-11-01"),
        "Great Recession": ("2007-12-01", "2009-06-01"),
        "COVID recession": ("2020-02-01", "2020-04-01"),
    }  

    if events is True:  
        _events = _default_recessions  
    elif isinstance(events, dict):  
        _events = events  
    else:  
        _events = {str(i): v for i, v in enumerate(events)}  

    x_min, x_max = pd.to_datetime(idx.min()), pd.to_datetime(idx.max())  
    _labeled = False  

    for _, win in _events.items():  
        if isinstance(win, (list, tuple, np.ndarray)) and len(win) == 2:  
            s, e = pd.to_datetime(win[0]), pd.to_datetime(win[1])  
        else:  
            s = e = pd.to_datetime(win)  

        if e < s:  
            s, e = e, s  

        if (e < x_min) or (s > x_max):  
            continue  

        s = max(s, x_min)  
        e = min(e, x_max)  
        if s == e:  
            ax.axvline(s, color=EVENT_COLOR, alpha=0.8, linewidth=1.0, zorder=0)  
        else:  
            ax.axvspan(  
                s, e, color=EVENT_COLOR, alpha=EVENT_ALPHA, zorder=0,  
                label=(EVENT_LABEL if not _labeled else None),  
            )  
            _labeled = True  

def plot_oos(
    y_true,
    y_pred,
    y_baseline,
    dates=None,
    title="Out-of-sample forecast",
    ylabel="Equity premium",
    save_path=None,
    show=True,
    y_lower = None,
    y_upper = None,
    ci_alpha = 0.2,
    ci_label = "Prediction 90% CI",
    events=False,

):
    """
    Plot true values, model predictions, and expanding-mean benchmark.
    Assumes 1-step-ahead series (use horizon=1 slice if you did multi-step).
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if y_lower is not None:                                          
        y_lower = np.asarray(y_lower, float)                         
    if y_upper is not None:                                          
        y_upper = np.asarray(y_upper, float)                         

    m = ~np.isnan(y_true) & ~np.isnan(y_pred)                       
    if y_lower is not None:                                         
        m &= ~np.isnan(y_lower)                                     
    if y_upper is not None:                                         
        m &= ~np.isnan(y_upper)                                     
    y_true, y_pred = y_true[m], y_pred[m]
    if y_lower is not None:                                           
        y_lower = y_lower[m]                                          
    if y_upper is not None:                                           
        y_upper = y_upper[m]                                          

    #csum = np.cumsum(y_true)
    #mean_forecast = csum / np.arange(1, len(y_true) + 1)
    mean_forecast = np.array(y_baseline)
    if dates is not None:
        x = pd.to_datetime(pd.Index(dates))[m]
    else:
        x = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y_true, label="True")
    ax.plot(x, y_pred, label="Prediction")
    if (y_lower is not None) and (y_upper is not None):               
        ax.fill_between(x, y_lower, y_upper, alpha=ci_alpha,label=ci_label) 
    ax.plot(x, mean_forecast, label="Expanding mean", linestyle="--")
    if events:  
        if dates is None:  
            raise ValueError("events=True requires `dates` (datetime-like) so event windows can be placed correctly.")  
        _add_event_windows(ax, x, events)  
        
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date" if dates is not None else "OOS step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def block_bootstrap_indices(T, block, rng):
    idx = []
    while len(idx) < T:
        start = rng.integers(0, T)
        idx.extend([(start + j) % T for j in range(block)])
    return np.array(idx[:T], dtype=int)

def bootstrap_cum_band(d, B=2000, block=12, seed=0, alpha=0.05, mode="null", block_bootstrap_indices_fn=None):
    """
    Pointwise block-bootstrap *null/reference* band for cumulative sum of d_t
    under the null E[d_t]=0, preserving short-run dependence via blocks.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    T = len(d)

    # impose null of equal average accuracy
    if block_bootstrap_indices_fn is None:
        # fallback: your existing helper must exist in your code base
        block_bootstrap_indices_fn = lambda T, block, rng: block_bootstrap_indices(T, block=block, rng=rng)
    cum_obs = np.cumsum(d)
    if mode == "null":
        d_resample = d - d.mean() # To force the model to have a zero mean so the distribution is zero, basically just centering 
    elif mode in ("delta", "tube"):
        d_resample = d
    else:
        raise ValueError(f"Unknown mode '{mode}' for bootstrap_cum_band")

    paths = np.empty((B, T), dtype=float)
    for b in range(B):
        ii = block_bootstrap_indices_fn(T, block=block, rng=rng)
        paths[b, :] = np.cumsum(d_resample[ii])
    if mode == "tube":
        mean_path = paths.mean(axis=0)
        dev = paths - mean_path
        lo_dev = np.quantile(dev, alpha/2, axis=0)
        hi_dev = np.quantile(dev, 1 - alpha/2, axis=0)
        lo = cum_obs + lo_dev
        hi = cum_obs + hi_dev
    else:
        lo = np.quantile(paths, alpha/2, axis=0)
        hi = np.quantile(paths, 1 - alpha/2, axis=0)
    return lo, hi

def plot_cum_dsse_with_bootstrap_band(
    y_true, y_pred, y_bench,
    dates=None,
    title="Cumulative ΔSSE (benchmark − model)",
    ylabel="Cumulative ΔSSE (benchmark − model)",
    xlabel="Year",
    add_band=True,
    B=2000,
    block=12,
    seed=0,
    band_alpha=0.20,
    ylim=None,
    ax=None,
    events = True,
    mode = "null" #delta, tube
):
    """
    Plots S_t = Σ[(y_t-ŷ_bench,t)^2 - (y_t-ŷ_model,t)^2].

    Optional shaded band: pointwise 95% circular block-bootstrap *null/reference band*
    for S_t under E[d_t]=0 (equal average predictive accuracy), constructed by
    resampling blocks of the centered loss differential d_t.
    """
    def _arr(x):
        if isinstance(x, (pd.Series, pd.Index)):
            x = x.to_numpy()
        return np.asarray(x, dtype=float).reshape(-1)

    y_true  = _arr(y_true)
    y_pred  = _arr(y_pred)
    y_bench = _arr(y_bench)

    if not (len(y_true) == len(y_pred) == len(y_bench)):
        raise ValueError("y_true, y_pred, y_bench must have the same length.")

    
    idx = pd.to_datetime(np.asarray(dates))
    idx_is_dt = True

    d = (y_true - y_bench) ** 2 - (y_true - y_pred) ** 2
    cum = np.cumsum(d)

    lower = upper = None
    if add_band:
        lower, upper = bootstrap_cum_band(d, B=B, block=block, seed=seed, alpha=0.05, mode = mode)

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 3.8))
    else:
        fig = ax.figure

    ax.plot(idx, cum, label="cum ΔSSE")
    ax.axhline(0.0, linewidth=1)

    if add_band:
        ax.fill_between(idx, lower, upper, alpha=band_alpha,
                        label=f"95% block-bootstrap null band (B={B}, block={block})")
    if events:  
        if dates is None:  
            raise ValueError("events=True requires `dates` (datetime-like) so event windows can be placed correctly.")
        _add_event_windows(ax, idx, events)  

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend(loc="best")

    if idx_is_dt:
        fig.autofmt_xdate()

    return fig, ax, pd.Series(cum, index=idx, name="cum_dsse"), (lower, upper)
