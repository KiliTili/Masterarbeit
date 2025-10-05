import pandas as pd
import numpy as np


def load_data(file_path="../../Data/GoyalAndWelch.xlsx"):
    """Load data from an Excel file."""
    return pd.read_excel(file_path, sheet_name="Monthly")

def format_date(df):
    """Format the date column to datetime."""
    df['date'] = pd.to_datetime(df['yyyymm'], format='%Y%m', errors='coerce')
    df.drop(columns=['yyyymm'], inplace=True)
    return df

def calc_equity_premium(df):
    """Calculate the equity premium."""
    # compute the equity premium (Goyal and Welch, 2002)
    # equity premium (or market premium) = return on the stock market (Rm(t)) - return on a short-term risk free treasury bill Rf(t)
    # more in detail: paper relies on the well-known (value-weighted CRSP index) return on the stock market and the 3-month risk-free treasury bill (called Rf(t) and obtained from Ibbotson)
    # This translates to ret (return w/ dividends (CRSP calc)) and Rfree (riskfree return)
    # to predict only data from 1926 onwards is used
    df['equity_premium'] = df['ret'] - df['Rfree']
    return df