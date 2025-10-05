import pandas as pd
import statsmodels.api as sm
def compare_to_paper(df, variables= "d/p"):
    """ takes the dataframe and compares the results to the paper """
    df['equity_premium'] = df['ret'] - df['Rfree']

    df['dp_lag'] = df[variables].shift(1)

    df = df.dropna(subset=['equity_premium', 'dp_lag'])
    df = df[df['date'] >= '1926-01-01']

    X = sm.add_constant(df['dp_lag'])
    y = df['equity_premium']

    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':12})  # Newey–West (12 lags)
    print(model.summary())

    r2_is = model.rsquared

    df['mean_ep'] = df['equity_premium'].expanding().mean().shift(1)

    df['pred_dp'] = model.params['const'] + model.params['dp_lag'] * df['dp_lag']

    sse_model = ((df['equity_premium'] - df['pred_dp'])**2).sum()
    sse_mean = ((df['equity_premium'] - df['mean_ep'])**2).sum()
    r2_oos = 1 - sse_model/sse_mean
    print(f"In-sample R²: {r2_is:.4f}, Out-of-sample R²: {r2_oos:.4f}")
    # the coefficient value of 0.2068 is the same as the one in the paper with 0.2