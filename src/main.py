import sys

import helpers.createData as cd
import helpers.dataFunctions as hdf
import ModelsMethods.survivability as surv
import ModelsMethods.rf_binary as rfb
import helpers.functions as fn
import basics.converter as converter

import pickle

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    precision_recall_curve,
    plot_roc_curve,
    confusion_matrix,
    plot_confusion_matrix,
)


def run_survivability():
    # df_survivability = cd.create_survivability_data()
    # df_partial = df_survivability.dropna()

    # surv.calc_survivability(
    #     df_partial, "years_to_death", "did_die", "CoxPHFitter_death-summary_partial.csv"
    # )

    df_survivability = cd.create_survivability_data(
        is_death=False, is_admission=True, days_to_admission=30, filterByGender=True
    )
    df_partial = df_survivability.dropna()

    surv.calc_survivability(
        df_partial,
        "diff",
        "is_admission",
        "CoxPHFitter_admission_30_female-summary_partial-filtered-subjects.csv",
        "Survivability Admissions Female - 30 Days",
    )

    # df_survivability = cd.create_survivability_data(
    #     is_death=False, is_admission=True, days_to_admission=365, filterByGender=True
    # )
    # df_partial = df_survivability.dropna()

    # surv.calc_survivability(
    #     df_partial,
    #     "diff",
    #     "is_admission",
    #     "CoxPHFitter_admission_365_all-summary_partial-filtered-subjects.csv",
    #     "Survivability Admissions - 365 Days",
    # )

    # df_survivability = cd.create_survivability_data(
    #     is_death=False, is_admission=True, days_to_admission=30, filterByGender=False
    # )
    # df_partial = df_survivability.dropna()

    # surv.calc_survivability(
    #     df_partial,
    #     "diff",
    #     "is_admission",
    #     "CoxPHFitter_admission_30_all-summary_partial-filtered-subjects.csv",
    #     "Survivability Admissions - 30 Days",
    # )

    # df_survivability = cd.create_survivability_data(
    #     is_death=False, is_admission=True, days_to_admission=365, filterByGender=False
    # )
    # df_partial = df_survivability.dropna()

    # surv.calc_survivability(
    #     df_partial,
    #     "diff",
    #     "is_admission",
    #     "CoxPHFitter_admission_365_female-summary_partial-filtered-subjects.csv",
    #     "Survivability Admissions Female - 365 Days",
    # )


def admission_propability_tests(days_to_admission, file_name):
    df_survivability = cd.create_survivability_data(
        is_death=False,
        is_admission=True,
        days_to_admission=days_to_admission,
        filterByGender=False,
    )

    df_partial = df_survivability.dropna()

    f, pf = fn.FTest(df_partial)

    stat, pt = fn.TTest(df_partial)

    ksstat1, p_value1 = fn.KSTest(df_partial, "diff_for_mean")
    ksstat2, p_value2 = fn.KSTest(df_partial, "visit_age")

    df_save = pd.DataFrame(
        columns=[
            "F-test",
            "F-test p-value",
            "t-test",
            "t-test p-value",
            "ks-test-diff",
            "ks-test-diff p-value",
            "ks-test-age",
            "ks-test-age p-value",
        ]
    )

    df_save.loc[0] = [f, pf, stat, pt, ksstat1, p_value1, ksstat2, p_value2]

    df_save.to_csv("{}.csv".format(file_name))


def run_binary_rf(type):
    data_x, data_y = cd.create_rf_data(is_admission=True, days_to_admission=30)

    X_train, X_test, y_train, y_test = rfb.split_train_test(data_x, data_y, type=type)

    # X_train = np.loadtxt(
    #     "blood-test-thesis/docs/results/X_train_{}.csv".format(type), delimiter=",",
    # )
    # y_train = np.loadtxt(
    #     "blood-test-thesis/docs/results/y_train_{}.csv".format(type), delimiter=",",
    # )
    # X_test = np.loadtxt(
    #     "blood-test-thesis/docs/results/X_test_{}.csv".format(type), delimiter=",",
    # )
    # y_test = np.loadtxt(
    #     "blood-test-thesis/docs/results/y_test_{}.csv".format(type), delimiter=",",
    # )

    model = rfb.CreateRandomForestClassifierModel(X_train, X_test, y_train, y_test)

    # Save model to pickle file
    pickle.dump(model, open("{}.sav".format(type), "wb"))


def rf_plot(filename):
    loaded_model = pickle.load(open("{}.sav".format(filename), "rb"))
    X_test = np.loadtxt(
        "blood-test-thesis/docs/results/X_test_{}.csv".format(filename), delimiter=",",
    )
    y_test = np.loadtxt(
        "blood-test-thesis/docs/results/y_test_{}.csv".format(filename), delimiter=",",
    )

    y_pred = loaded_model.predict(X_test)

    ax = plt.gca()
    rfc_disp = plot_roc_curve(loaded_model, X_test, y_test, ax=ax)
    # plot_confusion_matrix(loaded_model, X_test, y_test, ax=ax)

    # value = roc_auc_score(y_test, y_pred)

    # fpr, tpr, _ = roc_curve(y_test, y_pred)
    # label = "ROC AUC: " + str(value)
    # sns.lineplot(fpr, tpr, label=label)

    plt.show()


def run_rf(X_train, y_train, test_data, days_to_admission=30):
    # Split test data to data and outcome
    test_data = hdf.calc_admission_and_time_to_admission(test_data, days_to_admission)
    X_test = test_data.drop(columns=["patient_id", "outcome", "outcome_date"])
    y_test = test_data["is_admission"]

    model = rfb.CreateRandomForestClassifierModel(X_train, X_test, y_train, y_test)

    # Save model to pickle file
    pickle.dump(model, open("{}.sav".format(type), "wb"))

    y_pred = model.predict(X_test)

    value = roc_auc_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    # label = "ROC AUC: " + str(value)
    # sns.lineplot(fpr, tpr, label=label)

    ax = plt.gca()
    rfc_disp = plot_roc_curve(model, X_test, y_test, ax=ax)


if __name__ == "__main__":
    train_df, test_df = [], []

    if len(sys.argv) == 0:
        print("Expected at least two argument. Please add data file locations.")
        sys.exit()
    elif len(sys.argv) == 3:
        print("Fetching Data...")
        patients_df = pd.read_csv(sys.argv[0])
        tests_df = pd.read_csv(sys.argv[1])
        outcome_df = pd.read_csv(sys.argv[2])

        print("Converting to DataFrame...")
        df = converter.convert_to(patients_df, tests_df, outcome_df)

        print("Number of Rows after converstion: %s" % df.shape[0])
        print("Spliting to train and test files..")
        train_df, test_df = train_test_split(df, test_size=0.3)
        print("Saving files..")
        train_df.save_to_file("merged_data_train.csv")
        test_df.save_to_file("merged_data_test.csv")
    else:
        print("Opening train and test files")
        train_df = pd.read_csv(sys.argv[0])
        test_df = pd.read_csv(sys.argv[1])

        # Get data ready for RF
        print("Getting data ready for Random Forest with default admission value of 30")
        x_30, y_30 = hdf.create_rf_data(train_df)

        run_rf(x_30, y_30, test_df)

        print(
            "Getting data ready for Random Forest with default admission value of 365"
        )
        x_365, y_365 = hdf.create_rf_data(train_df, days_to_admission=365)

        run_rf(x_365, y_365, test_df, days_to_admission=365)

