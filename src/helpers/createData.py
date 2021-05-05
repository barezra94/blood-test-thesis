import pandas as pd
from ..helpers import values


def __filter_uk_data():
    df_uk = pd.read_csv("../../docs/ukbb_new_tests.csv")
    df_uk_added_data = pd.read_csv("../../docs/ICD10_ukbb_new.csv")

    print("Before", df_uk.shape)

    # Create DataSet to return
    wanted_columns = list(values.features.values())
    wanted_columns.extend(["eid"])
    df_uk = df_uk.loc[:, wanted_columns]

    # Drop rows that have NaN values in them
    df_uk = df_uk.dropna()

    print(df_uk.shape)

    # Create a DataSet for UK data that has values from both files
    df_uk = df_uk.merge(df_uk_added_data, on="eid")

    print(df_uk.columns)

    df_uk["sex_f31_0_0"].replace("Male", 1, inplace=True)
    df_uk["sex_f31_0_0"].replace("Female", 2, inplace=True)

    return df_uk


def __filter_il_data():
    df_il = pd.read_csv("../../docs/blood_test_il.csv")
    df_uk_added_data = pd.read_csv("../../docs/ICD10_ukbb_new.csv")

    # Filter dataSet values
    wanted_columns = list(values.features.keys())
    wanted_columns.extend(["hospital_patient_id"])

    df_il = df_il.loc[:, wanted_columns]

    # Keep only the latest test of the patient
    df_il = df_il.drop_duplicates(subset="hospital_patient_id", keep="last")
    # Drop rows that have NaN in them
    df_il = df_il.dropna()

    for illness in df_uk_added_data.columns[2:]:
        df_il[illness] = 3

    return df_il


def __merge_df(df_uk, df_il):

    # Change column names to match df_uk
    all_columns = {
        "hospital_patient_id": "eid",
    }
    all_columns.update(values.features)
    df_il = df_il.rename(columns=all_columns)

    df_uk = df_uk.drop(columns=["IID", "FID"])
    df_il = df_il.drop(columns=["FID"])

    # Merge datasets
    df_uk = df_uk.append(df_il)
    df_uk = df_uk.reset_index(drop=True)

    df_uk["D50*"] = df_uk["D500"]

    df_uk.loc[
        (df_uk["D500"] == 2)
        | (df_uk["D501"] == 2)
        | (df_uk["D508"] == 2)
        | (df_uk["D509"] == 2),
        "D50*",
    ] = 2

    df_uk = df_uk.drop(columns=["D500", "D501", "D508", "D509"])

    return df_uk


def create_data():
    """
        Creates the basic data for use
        Combines the UK dataset and the IL dataset into one
    """

    filtered_uk = __filter_uk_data()
    filtered_il = __filter_il_data()
    merged_df = __merge_df(df_uk=filtered_uk, df_il=filtered_il)

    return merged_df
