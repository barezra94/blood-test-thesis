import numpy as np
import scipy as sp
import pandas as pd
from datetime import datetime, timedelta

import matplotlib.pyplot as plt


def add_invalid_column(data, features_list):
    # If at least one test is not in the valid value range then value of column will be 0 (False)
    data["allTestValid"] = 1

    for key in features_list:
        # Gets data from dictonary
        min_value = features_list[key][0]
        max_value = features_list[key][1]

        # Change values of "allTestValid" according to test values in range
        data.loc[(data[key] > max_value) | (data[key] < min_value), "allTestValid"] = 0
        df = data.loc[(data[key] > max_value) | (data[key] < min_value)]
        print(key, df.shape)

    return data


# TODO: Maybe add a parameter here for different types of distances
def calc_diff_from_mean(df):
    """
        Preforms standeridization of the Dataframe and then adds a column of
        the difference for the the row from the overall mean.

        Type of difference is cdist.
    """
    # Remove the eid column from calculation
    df_dropped = df.drop(columns=["eid"])

    # Standerdize the Data and Calculate Mean
    pd_mean = np.nanmean(a=df_dropped, axis=0)
    pd_mean = pd_mean.reshape(1, -1)

    df_dropped["diff_for_mean"] = df_dropped.apply(
        lambda row: sp.spatial.distance.cdist(pd_mean, [row])[0][0], axis=1
    )

    df["diff_for_mean"] = np.absolute(df_dropped["diff_for_mean"])
    # df["diff_for_mean"] = df_dropped["diff_for_mean"]

    return df


def remove_columns_and_nan(data, features_list=False, illness=False):
    # Drop rows that have NaN values in them
    # Drop the Oestradiol (pmol/L) and Rheumatoid factor (IU/ml)
    # because the have too many NaN values

    # for x in features_list:
    #     print(x + " :", data[x].isnull().sum())

    if features_list == False:
        data = data.drop(columns=["Oestradiol (pmol/L)", "Rheumatoid factor (IU/ml)"])
    elif type(illness) is not list:
        keys_list = list(features_list.keys())
        keys_list.append(illness)
        data = data[keys_list]
    else:
        keys_list = list(features_list.keys())
        keys_list.extend(illness)
        data = data[keys_list]

    data = data.dropna()

    return data


def calc_num_of_illnesses(data):
    df = pd.read_csv("blood-test-thesis/docs/new_data.csv")

    cols_to_sum = df.columns[1:214]
    df["# of Illnesses"] = df[cols_to_sum].notna().sum(axis=1)

    # Drop all Columns except for eid and # of Illnesses
    df = df.filter(["eid", "# of Illnesses"], axis=1)
    df = df.rename(columns={"eid": "FID"})

    # Merge data
    data = data.merge(df, on="FID", how="left")

    return data


def calc_death_and_time_of_death(data):
    df = pd.read_csv("blood-test-thesis/docs/death_data.csv")

    df = df.filter(["eid", "40007-0.0"])  # Keeps only the subject id and age at death

    # Merge data
    data = data.merge(df, on="eid", how="left")

    # Subjects that have not died will have a value of -1 for further calculations
    data["40007-0.0"] = data["40007-0.0"].fillna(-1)

    data["years_to_death"] = data["40007-0.0"] - data["visit_age"]

    # Patients that have not died will get the max value
    max_years = data["years_to_death"].max()
    data.loc[(data["years_to_death"] < 0, "years_to_death")] = max_years + 1

    # Change the value of the death event
    data.loc[(data["40007-0.0"] != -1, "did_die")] = 1
    data.loc[(data["40007-0.0"] == -1, "did_die")] = 0

    data = data.drop(columns=["40007-0.0"])

    return data


def filter_at_least_2(filterEnabled=False):
    df_uk_53 = pd.read_csv("blood-test-thesis/docs/data_with_53.csv")

    if filterEnabled:
        df1 = df_uk_53[["eid", "53-0.0"]]
        df2 = df_uk_53[["eid", "53-1.0"]]
        df3 = df_uk_53[["eid", "53-2.0"]]
        df4 = df_uk_53[["eid", "53-3.0"]]

        # Remove NaN
        df1 = df1.dropna()
        df2 = df2.dropna()
        df3 = df3.dropna()
        df4 = df4.dropna()

        df12 = df1.index.intersection(df2.index)
        df13 = df1.index.intersection(df3.index)
        df14 = df1.index.intersection(df4.index)

        df_index_final = df12.union(df13)
        df_index_final = df_index_final.union(df14)

        result = df_uk_53.iloc[df_index_final]

        return result[["eid", "53-0.0"]]

    else:
        return df_uk_53[["eid", "53-0.0"]]


def calc_admission_and_time_to_admission(
    data, days_to_admission, include_diagnosis=False
):
    df_admission = pd.read_csv("blood-test-thesis/docs/hesin.txt", delimiter="\t",)
    df_diag = pd.read_csv("blood-test-thesis/docs/hesin_diag.txt", delimiter="\t",)

    df_admission = df_admission.drop_duplicates(subset="eid")
    df_diag = df_diag.drop_duplicates(subset="eid")

    df_uk_filtered = filter_at_least_2(filterEnabled=True)

    df_join = pd.merge(df_uk_filtered, df_admission, on="eid", how="left")
    if include_diagnosis:
        df_join = pd.merge(df_join, df_diag, on="eid", how="left")

    df_final = df_join[["eid", "53-0.0", "admidate"]]
    df_final = df_final.dropna()

    df_final["admidate"] = df_final["admidate"].apply(
        lambda x: datetime.strptime(x, "%d/%m/%Y")
    )
    df_final["53-0.0"] = df_final["53-0.0"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d")
    )
    df_final["diff"] = df_final["53-0.0"] - df_final["admidate"]

    # Keep only subjects that admission was after registration
    df = df_final.loc[df_final["diff"] <= timedelta(days=0)]

    data = data.merge(df[["eid", "diff"]], on="eid", how="left")

    data["diff"] = np.abs(data["diff"])
    data["diff"] = data["diff"].apply(lambda x: x.days)
    data["diff"] = data["diff"].fillna(-1)

    # Change the value of the admission event
    data["is_admission"] = 0

    if days_to_admission is np.nan:
        data.loc[data["diff"] >= 0, "is_admission"] = 1
    else:
        data.loc[
            (data["diff"] <= days_to_admission) & (data["diff"] >= 0), "is_admission"
        ] = 1
        data.loc[(data["diff"] > days_to_admission) | (data["diff"] < 0), "diff"] = (
            days_to_admission + 1
        )

    # Patients that were not admitted will get the max value
    # data.loc[(data["diff"] < 0, "diff")] =

    return data


def FTest(data):
    admission = data[data["is_admission"] > 0]
    not_admission = data[data["is_admission"] < 0]

    admission = admission["diff_for_mean"]
    not_admission = not_admission["diff_for_mean"]

    f = np.var(admission, ddof=1) / np.var(
        not_admission, ddof=1
    )  # calculate F test statistic
    dfn = admission.size - 1  # define degrees of freedom numerator
    dfd = not_admission.size - 1  # define degrees of freedom denominator
    p = 1 - sp.stats.f.cdf(f, dfn, dfd)  # find p-value of F test statistic
    return f, p


def TTest(data):
    admission = data[data["is_admission"] > 0]
    not_admission = data[data["is_admission"] < 0]

    admission_age = admission["visit_age"]
    not_admission_age = not_admission["visit_age"]

    stat, p = sp.stats.ttest_ind(admission_age, not_admission_age)

    return stat, p


def KSTest(data, field):
    admission = data[data["is_admission"] > 0]
    not_admission = data[data["is_admission"] < 0]

    admission = admission[field]
    not_admission = not_admission[field]

    stat, p = sp.stats.ks_2samp(admission, not_admission)

    all_data = data[field].values.reshape(1, -1)[0]
    admission = admission.values.reshape(1, -1)[0]
    not_admission = not_admission.values.reshape(1, -1)[0]

    plt.figure(figsize=(10, 5))
    # plt.plot(all_data, admission, label="admission")
    # plt.plot(all_data, not_admission, label="non admission")

    for val, p1, p2 in zip(all_data, admission, not_admission):
        plt.plot([val, val], [p1, p2], color="green", alpha=0.2)

    plt.legend()
    plt.ylabel("F(x)")
    plt.xlabel("x")
    plt.title("KS Goodness of Fit for 2 Samples")

    plt.show()

    return stat, p

