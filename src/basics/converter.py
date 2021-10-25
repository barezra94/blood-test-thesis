from datetime import datetime
import pandas as pd

"""
  Converter .py for UKBB data to MS dataframe for further checking

  Dataframe should look like the following:
    Patient Data:
    - Patient ID
    - DOB
    - gender
    - last update date

    Tests Data:
    - Patient ID
    - Date Taken
    - Test Name
    - Min Range
    - Max Range
    - unit
    - is valid

    Outcome Data:
    - Patient ID
    - date
    - event type
"""


def convert_from(ukbb_dataframe):
    pass


"""
Converter .py for MS data to UKBB dataframe for further checking

  Dataframe should look like the following:
    
"""


def convert_to(patients_df, tests_df, outcome_df, days_delta=30):
    df = pd.DataFrame(columns=["patient_id", "gender", "date"])

    for _, row in patients_df.iterrows():
        # Get all relevent patient data from the tests
        patient_tests = tests_df.loc[patients_df["patient_id"] == row["patient_id"]]

        # Group by dates to add as single rows
        testing_dates = patient_tests["date_taken"].unique()

        for date in testing_dates:
            tests = patient_tests.loc[patient_tests["date_taken"] == date]
            test_date_df = tests[["test", "value", "is_valid"]]
            test_date_df = test_date_df.transpose()

            outcome = outcome_df.iloc[
                outcome_df["patient_id"] == row["patient_id"]
                and (
                    date - datetime.timedelta(days=days_delta)
                    < outcome_df["date"]
                    < date + datetime.timedelta(days=days_delta)
                )
            ]

            test_date_df.append(
                {
                    "patient_id": row["patient_id"],
                    "gender": row["gender"],
                    "date": date,
                    "outcome": outcome[0],
                    "outcome_date": outcome_df["date"],
                }
            )

            df.append(test_date_df, ignore_index=True)

    print("Converted data to new Dataframe with %s rows" % df.shape[0])
    return df

