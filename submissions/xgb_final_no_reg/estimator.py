from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from xgboost.sklearn import XGBRegressor

# dataset website: https://dev.meteostat.net/bulk/hourly.html#endpoints
# dataset 2020 https://bulk.meteostat.net/v2/hourly/2020/07156.csv.gz
# dataset 2021 https://bulk.meteostat.net/v2/hourly/2021/07156.csv.gz
# license https://dev.meteostat.net/terms.html#license

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext.sort_values("date"), on="date"
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    num_features = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'pres']

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name", "season"]

    rest_cols = ['holiday', 'weekend', 'is_night', 'lockdown1', 'lockdown2']

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
            ("numf", StandardScaler(), num_features),
            ("rem", 'passthrough', rest_cols)
        ]
    )

    regressor = XGBRegressor(
        colsample_bytree=0.4, eval_metric='rmse',
        gamma=0.3, learning_rate=0.1,
        max_cat_threshold=64, max_cat_to_onehot=4,
        max_depth=9, min_child_weight=3,
        n_estimators=200, random_state=0
    )

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe
