import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter


def calc_survivability(
    data, duration_col, event_col, file_name, title="Survivability Death"
):
    cph = CoxPHFitter()
    cph.fit(
        data,
        duration_col=duration_col,
        event_col=event_col,
        show_progress=True,
        # step_size=0.1,
    )

    cph.summary.to_csv(file_name)
    cph.print_summary()

    kmf = KaplanMeierFitter()
    T = data[duration_col]
    E = data[event_col]
    kmf.fit(T, E)

    plt.figure()

    kmf.survival_function_
    kmf.cumulative_density_
    kmf.plot_survival_function()

    plt.title(title)
    plt.show()
