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

def calc_equity_premium(df, uselog = True):
    """Calculate the equity premium."""
    # compute the equity premium (Goyal and Welch, 2002)
    # equity premium (or market premium) = return on the stock market (Rm(t)) - return on a short-term risk free treasury bill Rf(t)
    # more in detail: paper relies on the well-known (value-weighted CRSP index) return on the stock market and the 3-month risk-free treasury bill (called Rf(t) and obtained from Ibbotson)
    # This translates to ret (return w/ dividends (CRSP calc)) and Rfree (riskfree return)
    # to predict only data from 1926 onwards is used
    df['equity_premium'] = np.log1p(df['ret']) - np.log1p(df['Rfree'])
    return df

def drop_na_after_1926(df):
    df = df.copy()
    start = pd.Timestamp('1926-01-01')
    idx = df.index
    before = idx < start
    after  = (idx >= start) & df['equity_premium'].notna()
    return df.loc[before | after]

def prepare_data(file_path="../../Data/GoyalAndWelch.xlsx"):
    """Load and prepare the data."""
    df = load_data(file_path)
    df = format_date(df)
    df = calc_equity_premium(df)
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


def label_states(df_diff, variable='M1WO', n_states=2):
    returns = df_diff[variable].dropna().values.reshape(-1, 1)

    model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=1000, random_state = 42) #full does not matter
    model.fit(returns)

    trans_mat = model.transmat_
    print("Transition matrix (rows sum to 1):")
    print(trans_mat)
    hidden_states = model.predict(returns)
    r = returns.reshape(-1)
    state_means = np.array([r[hidden_states == i].mean() for i in range(model.n_components)])
    state_stds  = np.array([r[hidden_states == i].std()  for i in range(model.n_components)])

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


def create_classification_data(file_path="../../Data/GoyalAndWelch.xlsx"):
    df = pd.read_csv("../../Data/MSCI_World_Data.csv")
    df_diff = difference_over_variables(df)
    df_diff['timestamp'] = pd.to_datetime(df_diff['timestamp'])

    df_states = label_states(df_diff)
    return df_states
