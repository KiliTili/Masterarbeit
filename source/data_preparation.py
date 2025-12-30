import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt


def load_data(file_path="../../Data/GoyalAndWelch.xlsx"):
    """Load data from an Excel file."""
    return pd.read_excel(file_path, sheet_name="Monthly")

def format_date(df):
    """Format the date column to datetime."""
    df['date'] = pd.to_datetime(df['yyyymm'], format='%Y%m', errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df.drop(columns=['yyyymm'], inplace=True)
    return df

def calc_equity_premium(df, uselog=True):
    # ret and Rfree are simple monthly returns (decimals) in this GW file
    if uselog:
        df["equity_premium"] = np.log1p(df["ret"]) - np.log1p(df["Rfree"])
        # equivalent: np.log((1+df["ret"]) / (1+df["Rfree"]))
    else:
        df["equity_premium"] = df["ret"] - df["Rfree"]
    return df

def drop_na_after_1926(df):
    df = df.copy()
    start = pd.Timestamp('1926-01-01')
    idx = df.index
    before = idx < start
    after  = (idx >= start) & df['equity_premium'].notna()
    return df.loc[before | after]

def prepare_data(file_path="../../Data/GoyalAndWelch.xlsx", uselog = True):
    """Load and prepare the data."""
    df = load_data(file_path)
    df = format_date(df)
    df = calc_equity_premium(df, uselog=uselog)
    df = drop_na_after_1926(df)
    return df


#### Classification specific functions #### 

def difference_over_variables(df,variables_dif = ('EUR003M',
       'FEDL01')  ,variables_log_dif = ('M1WO', 'NKY', 'SXXT','SPX','NKY','SPTR','GC1','CL1'), variables_log = ('V2X', 'MOVE', 'VIX', 'USYC2Y10', 'VXJ') ):
    for var in variables_log_dif:
        df = convert_stationary_variables_with_log(df, var)
    for var in variables_dif:
        df = convert_stationary_variables(df, var)
    for var in variables_log:
        df[var] = df[var].apply(lambda x: np.log(x) if x > 0 else 0)
    df = df.dropna().reset_index(drop=True)
    return df


def convert_stationary_variables(df, var):  
    df[var] = df[var].diff()
    #df = df.dropna().reset_index(drop=True)
    return df
def convert_stationary_variables_with_log(df, var):  
    df[var] = df[var].apply(lambda x: np.log(x) if x > 0 else 0)
    df[var] = df[var].diff()
    #df = df.dropna().reset_index(drop=True)
    return df


def label_states(df_diff, variable='M1WO', n_states=2, quiet=True, plot = True, random_state =42):
    returns = df_diff[variable].dropna().values.reshape(-1, 1)

    model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=1000, random_state = random_state) #full does not matter
    model.fit(returns)

    trans_mat = model.transmat_
    print("Transition matrix (rows sum to 1):")
    print(trans_mat)
    hidden_states = model.predict(returns)
    r = returns.reshape(-1)
    state_means = np.array([r[hidden_states == i].mean() for i in range(model.n_components)])
    state_stds  = np.array([r[hidden_states == i].std()  for i in range(model.n_components)])
    if not quiet:
        print("Model converged:", model.monitor_.converged)
        print("Converged in", model.monitor_.iter, "iterations (tol =", model.monitor_.tol, ")")

    print("State means:", state_means)
    print("State stds: ", state_stds)

    eps = 1e-8  # avoid divide-by-zero
    scores = state_means / (state_stds + eps)

    bull_state = np.argmax(scores)
    bear_state = np.argmin(scores)

    print(f"Bull state: {bull_state}, Bear state: {bear_state}")

    df_states = df_diff.loc[df_diff[variable].notna()].copy()
    df_states['state'] = hidden_states
    df_states['regime'] = df_states['state'].map({bull_state: 'Bull', bear_state: 'Bear'})
    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(df_states['timestamp'], df_states[variable], label='M1WO Returns', color='black')

        plt.fill_between(
            df_states['timestamp'], df_states[variable].min(), df_states[variable].max(),
            where=df_states['regime'] == 'Bear', color='red', alpha=0.25, label='Bear Regime'
        )
        plt.fill_between(
            df_states['timestamp'], df_states[variable].min(), df_states[variable].max(),
            where=df_states['regime'] == 'Bull', color='green', alpha=0.15, label='Bull Regime'
        )

        plt.title('M1WO: Bull & Bear Market Regimes (HMM)')
        plt.xlabel('Timestamp')
        plt.ylabel('M1WO Returns')
        plt.legend()
        plt.show()

    return df_states


def create_classification_data(file_path="../../Data/GoyalAndWelch.xlsx", quiet=True, random_state=42):
    df = pd.read_csv("../../Data/MSCI_World_Data.csv")
    df_diff = difference_over_variables(df)
    df_diff['timestamp'] = pd.to_datetime(df_diff['timestamp'])

    df_states = label_states(df_diff, quiet=quiet, random_state=random_state)
    return df_states


def simulate_hmm_equity_premium(
    n_samples: int = 2000,
    start_date: str = "1950-01-01",
    freq: str = "MS",
    seed: int = 42,
    # --- regime parameters (tune these for “easier/harder” prediction)
    mu_bull: float = 0.012,     # higher mean
    sigma_bull: float = 0.010,  # lower vol
    mu_bear: float = -0.006,    # lower (negative) mean, |mu_bear| < |mu_bull|
    sigma_bear: float = 0.030,  # higher vol
    p_bull_to_bear: float = 0.01,  # lower -> more persistent bull
    p_bear_to_bull: float = 0.05,  # bear lasts shorter on avg
    startprob: tuple = (0.9, 0.1),
    # engineered features for supervised models
    make_features: bool = True,
    vol_windows: tuple = (20, 60),
    mean_windows: tuple = (20, ),
    # target choice: "state_t" (current hidden state) or "state_t_plus_1" (next step)
    target: str = "state_t_plus_1",
):
    """
    Simulate equity-premium-like returns from a 2-state Gaussian HMM and
    return a DataFrame with useful labels/features for classification.

    To make prediction “easier”, increase regime separation (|mu_bull - mu_bear|)
    and persistence (smaller transition probabilities).
    """
    rng = np.random.default_rng(seed)

    # ---- transition matrix ----
    transmat = np.array([
        [1.0 - p_bull_to_bear, p_bull_to_bear],
        [p_bear_to_bull,       1.0 - p_bear_to_bull],
    ], dtype=float)
    startprob = np.asarray(startprob, float)

    # ---- build HMM with fixed params (no re-init) ----
    model = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        init_params="",           # do NOT override our params
        random_state=seed,
    )
    # We define component 0 = bull, 1 = bear (by construction).
    model.means_  = np.array([[mu_bull], [mu_bear]], dtype=float)
    model.covars_ = np.array([[sigma_bull**2], [sigma_bear**2]], dtype=float)
    model.transmat_  = transmat
    model.startprob_ = startprob

    # ---- sample ----
    observations, hidden_states = model.sample(n_samples)  # (T,1), (T,)
    returns = observations.ravel().astype(np.float32)

    # ---- date index ----
    dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)

    # ---- map hidden state -> Bull=1, Bear=0 robustly ----
    # (Just in case, infer by the higher mean instead of trusting index)
    means = model.means_.ravel()
    bull_id = int(np.argmax(means))
    state_true = (hidden_states == bull_id).astype(int)  # 1=bull, 0=bear

    # ---- base frame ----
    df = pd.DataFrame(
        {
            "equity_premium": returns,           # continuous “y”
            "state_true": state_true.astype(int) # latent label at time t
        },
        index=dates,
    )

    # optional engineered features from *past* info only
    if make_features:
        # absolute return often helps separate high-vol bear regimes
        df["abs_ret"] = df["equity_premium"].abs()

        # rolling vols / means (NaN in first windows)
        for w in vol_windows:
            df[f"rv{w}"] = df["equity_premium"].rolling(w, min_periods=w).std()
        for w in mean_windows:
            df[f"ma{w}"] = df["equity_premium"].rolling(w, min_periods=w).mean()


    df["timestamp"] = df.index
    return df, hidden_states, model
