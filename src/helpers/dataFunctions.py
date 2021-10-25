from datetime import datetime
import helpers.values
import numpy as np
import scipy as sp

from sklearn.preprocessing import StandardScaler


def remove_columns_and_nan(data):
    keys_list = list(helpers.values.features_new.keys())
    for _, value in helpers.values.features_new.items():
        if value.count() > 0:
            for index in value:
                keys_list.append(value[index])

    filtered_data = data[keys_list]

    # Concate columns that are of the same test
    for key, value in helpers.values.features_new.items():
        if value.count() > 0:
            # Assumes that missing values have a value of NaN
            filtered_data[key] = np.where(
                np.isnan(filtered_data[key]), filtered_data[value]
            )

            # Remove column
            filtered_data.drop(columns=[value], inplace=True)

    return filtered_data


def calc_diff_from_mean(df):
    """
        Preforms standeridization of the Dataframe and then adds a column of
        the difference for the the row from the overall mean.

        Type of difference is cdist.
    """
    # Remove the patient_id column from calculation
    df_dropped = df.drop(columns=["patient_id"])

    # Standerdize the Data and Calculate Mean
    pd_mean = np.nanmean(a=df_dropped, axis=0)
    pd_mean = pd_mean.reshape(1, -1)

    df_dropped["diff_for_mean"] = df_dropped.apply(
        lambda row: sp.spatial.distance.cdist(pd_mean, [row])[0][0], axis=1
    )

    df["diff_for_mean"] = np.absolute(df_dropped["diff_for_mean"])
    # df["diff_for_mean"] = df_dropped["diff_for_mean"]

    print("diff_for_mean column:")
    print(df["diff_for_mean"])

    return df


def calc_admission_and_time_to_admission(data, days_to_admission=30):
    data["is_admission"] = np.where(
        data["outcome"] != "death"
        and data["date"] + datetime.timedelta(days=days_to_admission)
        >= data["outcome_date"]
        and data["date"] <= data["outcome_date"]
    )

    return data


def create_rf_data(data, days_to_admission=30):
    # Calc days to admission
    data = calc_admission_and_time_to_admission(data, days_to_admission)

    # Calc the mean test score for patients
    df_for_mean = data.drop(
        columns=["gender", "date", "outcome", "outcome_date", "is_valid"]
    )
    df_for_mean = calc_diff_from_mean(data)
    df_for_mean = df_for_mean.filter(["patient_id", "diff_for_mean"])

    # Standerdize data except for the eid
    data.loc[:, data.columns != "patient_id"] = StandardScaler().fit_transform(
        data.loc[:, data.columns != "patient_id"]
    )

    data = data.merge(df_for_mean, on="patient_id", how="left")

    # Split data into data and outcome
    x = data.drop(columns=["patient_id", "outcome", "outcome_date", "is_admission"])
    y = data["is_admission"]

    return x, y

