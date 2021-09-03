import pandas as pd
import numpy as np
import helpers.values, helpers.functions

from sklearn.preprocessing import StandardScaler


def __filter_uk_data():
    df_uk = pd.read_csv("blood-test-thesis/docs/ukbb_new_tests.csv")
    df_uk_added_data = pd.read_csv("blood-test-thesis/docs/ICD10_ukbb_new.csv")
    df_uk_added_data_2 = pd.read_csv(
        "blood-test-thesis/docs/alzehimer_parkinson_asthma.csv"
    )

    df_uk_added_data_2 = df_uk_added_data_2.drop(
        columns=["sex", "visit_date", "visit_age"]
    )

    df_uk_added_data = df_uk_added_data.rename(columns={"FID": "eid"})

    # Create a DataSet for UK data that has values from both files
    df_uk = df_uk.merge(df_uk_added_data, on="eid", how="left")
    df_uk = df_uk.merge(df_uk_added_data_2, on="eid", how="left")

    df_uk["sex"].replace("Male", 1, inplace=True)
    df_uk["sex"].replace("Female", 2, inplace=True)

    df_uk["K760"] = df_uk["K760"] - 1

    # Create new columns for similar diseases
    df_uk["D50*"] = 0

    df_uk.loc[
        (df_uk["D500"] == 2)
        | (df_uk["D501"] == 2)
        | (df_uk["D508"] == 2)
        | (df_uk["D509"] == 2)
        | (df_uk["D630"] == 2)
        | (df_uk["D631"] == 2)
        | (df_uk["D638"] == 2)
        | (df_uk["D640"] == 2)
        | (df_uk["D641"] == 2)
        | (df_uk["D648"] == 2)
        | (df_uk["D642"] == 2)
        | (df_uk["D643"] == 2)
        | (df_uk["D644"] == 2),
        "D50*",
    ] = 1

    df_uk["D70*"] = 0

    df_uk.loc[
        (df_uk["D70"] == 2)
        | (df_uk["D700"] == 2)
        | (df_uk["D701"] == 2)
        | (df_uk["D702"] == 2)
        | (df_uk["D703"] == 2)
        | (df_uk["D704"] == 2)
        | (df_uk["D708"] == 2)
        | (df_uk["D709"] == 2),
        "D70*",
    ] = 1

    df_uk["alzheimer"] = 0
    df_uk.loc[(df_uk["Source of alzheimer's disease report"].notna(), "alzheimer")] = 1

    df_uk["parkinsonism"] = 0
    df_uk.loc[
        (df_uk["Source of all cause parkinsonism report"].notna(), "parkinsonism")
    ] = 1

    df_uk["asthma"] = 0
    df_uk.loc[(df_uk["Source of asthma report"].notna(), "asthma")] = 1

    df_uk = df_uk.drop(
        columns=[
            "D500",
            "D501",
            "D508",
            "D509",
            "D630",
            "D631",
            "D638",
            "D640",
            "D641",
            "D642",
            "D643",
            "D644",
            "D648",
            "D509",
            "D70",
            "D700",
            "D701",
            "D702",
            "D703",
            "D704",
            "D708",
            "D709",
            "visit_date",
            "Source of alzheimer's disease report",
            "Source of all cause parkinsonism report",
            "Source of asthma report",
        ]
    )
    print("After Dropping Phenotype Columns: ", df_uk.shape)

    # return df_uk[df_uk["sex"] == 1], df_uk[df_uk["sex"] == 2], df_uk
    return df_uk


def __filter_il_data():
    df_il = pd.read_csv("../../docs/blood_test_il.csv")
    df_uk_added_data = pd.read_csv("../../docs/ICD10_ukbb_new.csv")

    # Filter dataSet values
    wanted_columns = list(helpers.values.features.keys())
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
    all_columns.update(helpers.values.features)
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


def create_rf_data(
    is_death=False, is_admission=False, days_to_admission=30,
):
    """
        Get the data from the UKBB, with additional data according to choice.
        Splits the data into data  and outcome.
    """
    data = __filter_uk_data()
    df_for_mean = data.drop(
        columns=[
            "parkinsonism",
            "alzheimer",
            "asthma",
            "K760",
            "D50*",
            "D70*",
            "visit_age",
            "sex",
        ]
    )

    if not is_death and not is_admission:
        data = helpers.functions.calc_num_of_illnesses(data)

        data = helpers.functions.remove_columns_and_nan(
            data,
            helpers.values.features_values_female_partly,
            [
                "parkinsonism",
                "alzheimer",
                "asthma",
                "K760",
                "D50*",
                "D70*",
                "# of Illnesses",
                "visit_age",
                "sex",
                # "diff_for_mean",
            ],
        )

        x = data.drop(
            columns=[
                "parkinsonism",
                "alzheimer",
                "asthma",
                "K760",
                "D50*",
                "D70*",
                "# of Illnesses",
            ]
        )

        y = np.where((data["# of Illnesses"] > 0), 1, 0)

        return x, y

    elif is_death and not is_admission:
        data = helpers.functions.calc_death_and_time_of_death(data)

        data = helpers.functions.remove_columns_and_nan(
            data,
            helpers.values.features_values_female_partly,
            [
                "parkinsonism",
                "alzheimer",
                "asthma",
                "K760",
                "D50*",
                "D70*",
                "years_to_death",
                "visit_age",
                "sex",
                # "diff_for_mean",
            ],
        )

        x = data.drop(
            columns=[
                "parkinsonism",
                "alzheimer",
                "asthma",
                "K760",
                "D50*",
                "D70*",
                "years_to_death",
            ]
        )

        y = np.where((data["years_to_death"] < 0), 0, 1)

        return x, y

    elif not is_death and is_admission:
        data = __filter_uk_data()
        data = helpers.functions.remove_columns_and_nan(
            data,
            helpers.values.features_values_female_partly,
            [
                "parkinsonism",
                "alzheimer",
                "asthma",
                "K760",
                "D50*",
                "D70*",
                "visit_age",
                "sex",
                "eid"
                # "diff_for_mean",
            ],
        )

        data = helpers.functions.calc_admission_and_time_to_admission(
            data, days_to_admission
        )

        # Standerdize data except for the eid
        data.loc[:, data.columns != "eid"] = StandardScaler().fit_transform(
            data.loc[:, data.columns != "eid"]
        )

        df_for_mean = data.drop(
            # columns=["visit_age", "sex", "diff", "is_admission"]
            columns=[
                "visit_age",
                "parkinsonism",
                "alzheimer",
                "asthma",
                "K760",
                "D50*",
                "D70*",
                "diff",
                "sex",
                "is_admission",
            ]
        )

        df_for_mean = helpers.functions.calc_diff_from_mean(df_for_mean)

        df_for_mean = df_for_mean.filter(["eid", "diff_for_mean"])
        data = data.merge(df_for_mean, on="eid", how="left")

        data = helpers.functions.remove_columns_and_nan(
            data,
            helpers.values.features_values_female_partly,
            [
                "parkinsonism",
                "alzheimer",
                "asthma",
                "K760",
                "D50*",
                "D70*",
                "diff",
                "visit_age",
                "sex",
                "diff_for_mean",
                "eid",
            ],
        )

        x = data.drop(
            columns=[
                "parkinsonism",
                "alzheimer",
                "asthma",
                "K760",
                "D50*",
                "D70*",
                "diff",
                "eid",
            ]
        )

        # x = data[["sex", "diff_for_mean", "visit_age"]]
        y = np.where((data["diff"] > 0), 0, 1)

        return x, y

    else:
        raise Exception("No valid argument was passed.")


def create_survivability_data(
    is_death=True,
    is_admission=False,
    return_full=False,
    is_partial=True,
    days_to_admission=np.nan,
    filterByGender=False,
):
    """
        Get the data for the survivability models. 
        is_death - should a time to death column be added. Default True.
        is_admission - should a time to admission column be added. Defualt False.
        return_full - should the dataframe that is returned include the full list of tests. Defualt False.
        is_partial - calc using partial lab test features. Uses female partial tests

        Returns a dataframe containing - age, gender, diff from mean and the specified columns

        
    """

    df = __filter_uk_data()

    # Remove values from the dataset
    if is_partial:
        df = helpers.functions.remove_columns_and_nan(
            df,
            helpers.values.features_values_female_partly,
            ["visit_age", "sex", "eid"],
        )
    if filterByGender:
        # Filters the dataset to contain only female patients
        print(df.shape)
        df = df.loc[df["sex"] == 2]
        df = df.drop(columns=["sex"])
        print(df.shape)
    if is_death:
        df_death_values = helpers.functions.calc_death_and_time_of_death(df)

        # Standerdize data except for the eid
        df_death_values.loc[
            :, df_death_values.columns != "eid"
        ] = StandardScaler().fit_transform(
            df_death_values.loc[:, df_death_values.columns != "eid"]
        )

        df_for_mean = df_death_values.drop(
            columns=["visit_age", "sex", "years_to_death", "did_die"]
        )

        df_for_mean = helpers.functions.calc_diff_from_mean(df_for_mean)

        df_for_mean = df_for_mean.filter(["eid", "diff_for_mean"])
        df_death_values = df_death_values.merge(df_for_mean, on="eid", how="left")

        if return_full:
            return df_death_values
        else:
            return df_death_values[
                ["did_die", "diff_for_mean", "visit_age", "sex", "years_to_death"]
            ]

    elif is_admission:
        df_admission_values = helpers.functions.calc_admission_and_time_to_admission(
            df, days_to_admission
        )

        # Standerdize data except for the eid
        df_admission_values.loc[
            :, df_admission_values.columns != "eid"
        ] = StandardScaler().fit_transform(
            df_admission_values.loc[:, df_admission_values.columns != "eid"]
        )

        df_for_mean = df_admission_values.drop(
            # columns=["visit_age", "sex", "diff", "is_admission"]
            columns=["visit_age", "diff", "is_admission"]
        )

        df_for_mean = helpers.functions.calc_diff_from_mean(df_for_mean)

        df_for_mean = df_for_mean.filter(["eid", "diff_for_mean"])
        df_admission_values = df_admission_values.merge(
            df_for_mean, on="eid", how="left"
        )

        if return_full:
            return df_admission_values
        else:
            return df_admission_values[
                ["is_admission", "diff_for_mean", "visit_age", "diff",]
                # ["is_admission", "diff_for_mean", "visit_age", "sex", "diff",]
            ]

    else:
        raise Exception("No valid argument was passed.")
