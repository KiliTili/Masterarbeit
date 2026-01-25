import pandas as pd
import numpy as np




import os
import pandas as pd

def save_oos_results(
    dates,
    y_true,
    y_pred,
    model_name,
    y_lower=None,
    y_upper=None,
    filepath="../../Data/ModelData/oos_predictions.csv",
    mode="append",  # "append" or "overwrite"
):
    # Make model name file/column safe
    safe_name = (
        model_name.replace("(", "")
                  .replace(")", "")
                  .replace("/", "_")
                  .replace(" ", "_")
    )

    # Build new DataFrame for this model
    data = {
        "date": pd.to_datetime(dates),
        f"y_true_{safe_name}": y_true,
        f"y_pred_{safe_name}": y_pred,
    }

    if y_lower is not None and y_upper is not None:
        data[f"y_lower_{safe_name}"] = y_lower
        data[f"y_upper_{safe_name}"] = y_upper

    df_new = pd.DataFrame(data).set_index("date")

    # --- Case 1: overwrite OR file doesn't exist ---
    if mode == "overwrite" or not os.path.exists(filepath):
        df_new.to_csv(filepath)
        print(f"Saved new file: {filepath}")
        return

    # --- Case 2: append columns to existing file ---
    df_existing = pd.read_csv(filepath, parse_dates=["date"]).set_index("date")

    # Join on date index
    df_combined = df_existing.join(df_new, how="outer")

    df_combined.to_csv(filepath)
    print(f"Appended model results to: {filepath}")
