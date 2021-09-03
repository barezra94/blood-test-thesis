import helpers.createData as cd
import ModelsMethods.survivability as surv
import ModelsMethods.rf_binary as rfb
import helpers.functions as fn

import pickle

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

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

    X_train = np.loadtxt(
        "blood-test-thesis/docs/results/X_train_{}.csv".format(type), delimiter=",",
    )
    y_train = np.loadtxt(
        "blood-test-thesis/docs/results/y_train_{}.csv".format(type), delimiter=",",
    )
    X_test = np.loadtxt(
        "blood-test-thesis/docs/results/X_test_{}.csv".format(type), delimiter=",",
    )
    y_test = np.loadtxt(
        "blood-test-thesis/docs/results/y_test_{}.csv".format(type), delimiter=",",
    )

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


if __name__ == "__main__":
    # run_survivability()
    run_binary_rf(type="binary_admission_diff-from-mean_all_columns_30")
    rf_plot("binary_admission_diff-from-mean_all_columns_30")
    # rfb.LoadFromFile()
    # admission_propability_tests(days_to_admission=365, file_name="all-365-1")
