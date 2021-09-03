from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from ..helpers import values


def mean_diff(data, pca=True, drop_columns=["K760", "D50*"]):
    data = data.drop(columns=drop_columns)

    if pca:
        pca_data = PCA(n_components=len(values.features))
        data = pca_data.fit_transform(data)

    mean = data.mean(axis=0)

    return mean


def data_prep(data, test_size=0.3, random_state=13, fatty_liver=True):
    """ Splits the data into train and test, according to the test size sent. 
    Will return the results for Fatty Liver or Iron.
    Returns Background data for CPCA calculations.  """

    dataFrame = data[data["K760"] != 3]

    x = dataFrame.drop(columns=["K760", "D50*"])
    y1 = dataFrame["K760"]
    y2 = dataFrame["D50*"]

    df = data.loc[:, data.columns.difference(["K760", "D50*"])]

    background = df[(data["K760"] == 3) | (data["D50*"] == 3)]
    background = background.values

    if fatty_liver:
        X_train, X_test, y_train, y_test = train_test_split(
            x, y1, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test, background

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            x, y1, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test, background
