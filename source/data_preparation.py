import pandas as pd
import numpy as np


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