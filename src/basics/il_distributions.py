import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import math

df = pd.read_csv("docs/blood_test_il.csv")
dataSet = []

# TODO: Change this to female for continuity of the research
blood_normals_male = {
    "RBC": (4.7, 6.1),
    "BASO_percent": (0, 0.5),
    "BASO_NO": (0, 100),
    "EOS_precent": (0, 7),
    "EOS_NO": (0, 800),
    "HCT": (40.7, 50.3),
    "HGB": (13.8, 17.2),
    "LYM_precent": (18, 45),
    "LYM": (800, 5000),
    "MCH": (23, 31),
    "MCHC": (32, 36),
    "MCV": (80, 95),
    "MONO_precent": (1, 10),
    "MONO": (400, 1000),
    "MPV": (7, 11),
    "NEU": (1800, 8300),
    "NEU_precent": (45, 75),
    "PLT": (150, 400),
    "RDW": (11, 15),
    "RETICUL_precent": (0.5, 2.0),
    "WBC": (4500, 10000),
}

# Filter out fields by gender and duplicate. Defualt value if not sent - male
def set_filtered_data(gender=1):
    global df
    df = df[df["gender"] == gender]
    df = df.drop_duplicates(subset="hospital_patient_id", keep="last")

    for col in df.columns[3:]:
        dataSet.append({"label": col, "value": col})


# Plots the ZScore of the two values that were sent
def z_score_outlier(valueName1, valueName2, sd=4):  # sd - The standard deviation
    # Filter out the fields that where not sent and remove null values
    dataFrame = df[[valueName1, valueName2]]
    dataFrame.replace("", np.nan, inplace=True)
    dataFrame.dropna(inplace=True)

    # Calculate the Standard Score
    dataFrame = (dataFrame - dataFrame.mean()) / dataFrame.std()

    x_out = dataFrame[abs(dataFrame[valueName1]) > sd]
    y_out = dataFrame[abs(dataFrame[valueName2]) > sd]
    x_y_out = dataFrame[
        (abs(dataFrame[valueName1]) > sd) & (abs(dataFrame[valueName2]) > sd)
    ]
    print(len(x_out), len(y_out), len(x_y_out))

    trace0 = go.Scattergl(
        x=dataFrame[valueName1],
        y=dataFrame[valueName2],
        mode="markers",
        name="Both Normal",
    )
    trace1 = go.Scattergl(
        x=x_out[valueName1],
        y=x_out[valueName2],
        mode="markers",
        name=valueName1 + " Not Normal",
    )
    trace2 = go.Scattergl(
        x=y_out[valueName1],
        y=y_out[valueName2],
        mode="markers",
        name=valueName2 + " Not Normal",
    )
    trace3 = go.Scattergl(
        x=x_y_out[valueName1],
        y=x_y_out[valueName2],
        mode="markers",
        name="Both Not Normal",
    )

    return [trace0, trace1, trace2, trace3]


def abnormal_outliner(valueName1, valueName2):
    dataFrame = df[[valueName1, valueName2]]
    dataFrame.replace("", np.nan, inplace=True)
    dataFrame.dropna(inplace=True)

    valueName1_out = dataFrame[
        (dataFrame[valueName1] < blood_normals_male[valueName1][0])
        | (dataFrame[valueName1] > blood_normals_male[valueName1][1])
    ]
    valueName2_out = dataFrame[
        (dataFrame[valueName2] < blood_normals_male[valueName2][0])
        | (dataFrame[valueName2] > blood_normals_male[valueName2][1])
    ]
    valueName1_valueName2_out = dataFrame[
        (
            (dataFrame[valueName2] < blood_normals_male[valueName2][0])
            | (dataFrame[valueName2] > blood_normals_male[valueName2][1])
        )
        & (
            (dataFrame[valueName1] < blood_normals_male[valueName1][0])
            | (dataFrame[valueName1] > blood_normals_male[valueName1][1])
        )
    ]

    print(
        len(valueName1_out),
        len(valueName2_out),
        len(valueName1_valueName2_out),
        len(dataFrame),
    )

    trace0 = go.Scattergl(
        x=dataFrame[valueName1],
        y=dataFrame[valueName2],
        mode="markers",
        name="Both Normal",
        marker=dict(color="#28AD28"),
    )
    trace1 = go.Scattergl(
        x=valueName1_out[valueName1],
        y=valueName1_out[valueName2],
        mode="markers",
        name=valueName1 + " Not Normal",
        marker=dict(color="#16DED1"),
    )
    trace2 = go.Scattergl(
        x=valueName2_out[valueName1],
        y=valueName2_out[valueName2],
        mode="markers",
        name=valueName2 + " Not Normal",
        marker=dict(color="#B510EA"),
    )
    trace3 = go.Scattergl(
        x=valueName1_valueName2_out[valueName1],
        y=valueName1_valueName2_out[valueName2],
        mode="markers",
        name="Both Not Normal",
        marker=dict(color="#CD1616"),
    )

    return [trace0, trace1, trace2, trace3]


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Dropdown(id="valueName1", options=dataSet, value="RBC",),
                dcc.Dropdown(id="valueName2", options=dataSet, value="WBC",),
                dcc.RadioItems(
                    id="test-type",
                    options=[
                        {"label": i, "value": i}
                        for i in ["Z-Score", "Abnormal Outliner"]
                    ],
                    value="Z-Score",
                    labelStyle={"display": "inline-block"},
                ),
            ],
            style={"width": "49%", "display": "inline-block"},
        ),
        html.Div([dcc.Graph(id="graph"),]),
    ]
)


@app.callback(
    dash.dependencies.Output("graph", "figure"),
    [
        dash.dependencies.Input("test-type", "value"),
        dash.dependencies.Input("valueName1", "value"),
        dash.dependencies.Input("valueName2", "value"),
    ],
)
def change_graph_data(test_type, value1, value2):
    data = []

    if test_type == "Z-Score":
        data = z_score_outlier(value1, value2)
    else:
        data = abnormal_outliner(value1, value2)

    return {
        "data": data,
    }


if __name__ == "__main__":
    set_filtered_data()
    app.run_server(debug=True)
