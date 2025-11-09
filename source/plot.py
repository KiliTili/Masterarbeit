
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_series(df, timestamp_col='timestamp', figsize=(14, 12)):
    """
    Plots all numeric columns in the DataFrame over time.
    
    Parameters:
        df (pd.DataFrame): DataFrame with time series data.
        timestamp_col (str): Name of the timestamp column.
        figsize (tuple): Figure size.
    """
    # Ensure timestamp is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Select only numeric columns (skip object/text columns)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Create subplots
    n_cols = 2
    n_rows = (len(numeric_cols) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = axes.flatten()

    # Plot each variable
    for i, col in enumerate(numeric_cols):
        axes[i].plot(df[timestamp_col], df[col], label=col)
        axes[i].set_title(col)
        axes[i].set_xlabel('Timestamp')
        axes[i].set_ylabel('Value')
        axes[i].legend(loc='upper left', fontsize='small')

    # Remove empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

plot_all_series(df)