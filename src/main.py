import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn import svm
import google_trends.dl_google_trends as dl_google_trends


def plot_time_series(df, x, y):
    fig = px.line(df, x=x, y=y)
    fig.update_layout(yaxis_range=[0, 100])
    fig.show()


def feature_ext(dataframe, labels_df):
    df = pd.DataFrame()
    for col in dataframe.columns:
        if col != "date":
            data_col = dataframe[col] / 100
            class_n = labels_df.loc[col]["class"]
            d = {
                "mean": data_col.mean(),
                "max_zscore": ((max(data_col) - data_col.mean()) / (data_col.std())),
                "autocorr": data_col.autocorr(lag=53),
                "topic": col,
                "class_n": class_n,
            }

            i = [0]
            df_tmp = pd.DataFrame(data=d, index=i)
            df = pd.concat([df, df_tmp], ignore_index=True, sort=False)

    return df


def drop_rows_with_nan(df):
    print(f"Before drop: {df.shape}")
    df = df.dropna(axis=0, how="any")
    print(f"After drop: {df.shape}")
    return df


def remove_minus_one_label(features):
    print(f"Before drop: {features.shape}")
    minus = features[features["class_n"] == -1]
    features = features[features["class_n"] != -1]
    print(f"After drop: {features.shape}")
    return features, minus


def create_svm_model(features):
    X = features[["max_zscore", "autocorr", "mean"]]
    Y = features["class_n"]
    clf = svm.SVC(kernel="linear").fit(X, Y)
    return clf


def predic_with_svm_model(clf, features):
    X = features[["max_zscore", "autocorr", "mean"]]
    features["svm_label"] = clf.predict(X)
    return features


def plot_data(features, labels_df=""):
    if labels_df == "":
        fig = px.scatter_3d(
            features,
            y="max_zscore",
            x="autocorr",
            z="mean",
            text="topic",
        )
    else:
        fig = px.scatter_3d(
            features,
            y="max_zscore",
            x="autocorr",
            z="mean",
            text="topic",
            color=labels_df,
        )
    fig.show()


def plot_svm_nth_separation_plane(features, svm_model, plane_number):
    # plot features
    fig = px.scatter_3d(
        features,
        x="max_zscore",
        y="autocorr",
        z="mean",
        text="topic",
        color="svm_label",
    )
    # plot nth separation plane
    x_min, x_max = (
        features["max_zscore"].min() - 1,
        features["max_zscore"].max() + 1,
    )
    y_min, y_max = (
        features["autocorr"].min() - 1,
        features["autocorr"].max() + 1,
    )
    x, y = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    z = (
        lambda x, y: (
            -svm_model.intercept_[plane_number]
            - svm_model.coef_[plane_number][0] * x
            - svm_model.coef_[plane_number][1] * y
        )
        / svm_model.coef_[plane_number][2]
    )
    fig.add_traces(
        go.Surface(
            x=x,
            y=y,
            z=z(x, y),
            showscale=False,
            opacity=0.5,
            colorscale="Blues",
            name="SVM separation plane",
        )
    )
    fig.show()


if __name__ == "__main__":
    train_df = pd.read_csv("data/gtrends.csv")
    labels_df = pd.read_csv("data/labels.csv", index_col=0)
    features = feature_ext(train_df, labels_df)

    # important time series
    # plot_time_series(train_df, "date", "long coat")
    # plot_time_series(train_df, "date", "cloth")
    # plot_time_series(train_df, "date", "cady")
    # plot_time_series(train_df, "date", "hron")

    # plot_data(features, "")

    features = drop_rows_with_nan(features)

    # plot_data(features, "class_n")

    features, minus = remove_minus_one_label(features)

    # plot_data(features, "")

    plot_data(features, "class_n")

    svm_model = create_svm_model(features)
    minus = predic_with_svm_model(svm_model, minus)
    features = predic_with_svm_model(svm_model, features)
    # plot_data(minus, "svm_label")

    # for i in range(0, 6):
    #    plot_svm_nth_separation_plane(minus, svm_model, i)

    #for i in range(0, 6):
    #    plot_svm_nth_separation_plane(features, svm_model, i)
