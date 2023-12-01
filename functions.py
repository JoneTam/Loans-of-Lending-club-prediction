import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from math import ceil
import scipy.stats as stats
from sklearn.metrics import (
    confusion_matrix,
)
from sklearn.base import BaseEstimator, TransformerMixin
import math
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from sklearn.base import BaseEstimator, TransformerMixin


def categorical_to_binary(df: pd.DataFrame, value1: str, value0: str) -> pd.DataFrame:
    """Replace "yes" with 1 and "no" with 0"""
    df1 = df.applymap(lambda x: 1 if x == value1 else (0 if x == value0 else x))
    return df1


def count_unique_values(dataframe1: pd.DataFrame) -> pd.DataFrame:
    """function takes dataframe1 as input and calculates the number of unique values in each column."""
    unique_value_counts = [
        {"Column name": column, "Unique Value Count": dataframe1[column].nunique()}
        for column in dataframe1.columns
    ]
    dataframe2 = pd.DataFrame(unique_value_counts)
    return dataframe2


def unique_values(df: pd.DataFrame) -> dict:
    """Lists all unique values in columns"""
    unique_values_dict = {}

    for column in df.columns:
        unique_values_dict[column] = df[column].unique().tolist()
    return unique_values_dict


def copy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Takes pd.DataFrame as an input and returns it's copy"""
    return df.copy()


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Takes pd.DataFrame as an input and drops duplicates"""
    return df.drop_duplicates()


def drop_rows_if_nan(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df[df[column].notnull()]


def get_year_month(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Takes pd.DataFrame column with date, turns it into datetime dtype and
    extracts year and month into two new columns.
    params:df: pd.DataFrame;
           columns: str - title of the column which has date
    returns: pd.DataFrame with 2 new columns"""

    df[column] = pd.to_datetime(df[column])
    df["Year"] = pd.DatetimeIndex(df[column]).year
    df["Month"] = pd.DatetimeIndex(df[column]).month
    return df


def date_split_to_Y_M(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Splits date format like "Aug 2015" into two columns for year and month"""
    for column1 in columns:
        df[f"Month_{column1}"] = df[column1].str[:3]
        df[f"Year_{column1}"] = df[column1].str[4:8].astype(float)
        df[f"Month_{column1}"] = df[f"Month_{column1}"].replace(
            {
                "Jan": 1,
                "Feb": 2,
                "Mar": 3,
                "Apr": 4,
                "May": 5,
                "Jun": 6,
                "Jul": 7,
                "Aug": 8,
                "Sep": 9,
                "Oct": 10,
                "Nov": 11,
                "Dec": 12,
            }
        )
    return df


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Takes Data Frame and list of columns.
    Drops columns listed in the list.
    Returns new dataframe
    """

    df = df.drop(columns=columns)
    return df


def replace_what_with_what(
    df: pd.DataFrame, column1: str, to_replace: str, with_what: str
) -> pd.DataFrame:
    """Takes Data Frame and name of column.
    Replace smth with smth and tranform into float.
    Returns new dataframe
    """
    df[column1] = df[column1].str.replace(to_replace, with_what).astype(float)

    return df


def employment_Length_to_numeric(df: pd.DataFrame, column1) -> pd.DataFrame:
    """Convert employment length categories to numeric values in a specified column of a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column1 (str): The column containing employment length categories to be converted.

    Returns:
    - pd.DataFrame: DataFrame with the specified column converted to numeric values."""

    df[column1] = (
        df[column1]
        .replace(
            {
                "< 1 year": "0",
                "1 year": "1",
                "2 years": "2",
                "3 years": "3",
                "4 years": "4",
                "5 years": "5",
                "6 years": "6",
                "7 years": "7",
                "8 years": "8",
                "9 years": "9",
                "10+ years": "10",
            }
        )
        .astype(float)
    )
    return df


def filter_column_range_0_to_100(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Filter a DataFrame based on a specified column's values within the range of 0 to 100.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): The column to filter.

    Returns:
    - pd.DataFrame: DataFrame with rows filtered based on the specified column's values within the range of 0 to 100.
    """

    df = df[(df[column] >= 0) & (df[column] <= 100)]
    return df


def insert_status(df: pd.DataFrame, number: int) -> pd.DataFrame:
    """Takes pd.DataFrame and inserts new column, named 'status' with certain set int value for target values"""
    df["status"] = number
    return df


def calc_risk(df: pd.DataFrame, column1: str, column2: str) -> pd.DataFrame:
    """Calculates risk lika a mean of "fico_range_low" and 'fico_range_high'"""
    df["risk_score"] = (df[column1] + df[column2]) / 2
    return df


def months_sin_cos(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Transform specified columns in a DataFrame by adding sine and cosine components of the months.
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns containing month values to be transformed.
    Returns: pd.DataFrame: DataFrame with added cosine and sine components for each specified column.
    """
    for col in columns:
        df["cos_" + col] = np.cos(2 * math.pi * df[col] / df[col].max())
        df["sin_" + col] = np.sin(2 * math.pi * df[col] / df[col].max())
    return df


def select_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Select specified columns from a DataFrame.
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - cols (list): List of column names to be selected.
    Returns: pd.DataFrame: DataFrame with only the selected columns."""
    df = df[cols]
    return df


def rename_columns(df: pd.DataFrame, names: dict) -> pd.DataFrame:
    """Takes as an input pd.DataFrame and a dict with old and new column names. Renames columns"""
    df.rename(names, axis=1, inplace=True)
    return df


def log_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Takes as an input pd.DataFrame certain column and returns dataframe with new column of log_feature"""
    df["log_" + feature] = np.log(df[feature] + 0.0001)
    return df


# PLOTTING


def my_countplot(df: pd.DataFrame, feature1: str, feature2: str, title: str) -> None:
    """Plots the count plot of df, setting feature1 as name of X axis,
    feature2 as name of 'hue' parameter"""

    ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x=feature1, hue=feature2, data=df)
    ax.set_title(title)
    ax.set(ylabel="")
    ax.set(xlabel=feature1)
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])


def my_proportionsplot(
    df: pd.DataFrame, feature1: str, feature2: str, title: str
) -> None:
    """Plots the proportion plot of df, setting feature1 as name of X axis,
    feature2 as name of 'hue' parameter"""
    plt.figure(figsize=(15, 5))
    sns.histplot(
        data=df,
        x=feature1,
        hue=feature2,
        multiple="fill",
        stat="proportion",
        discrete=True,
        shrink=0.5,
    ).set(title=title)


def my_plots(
    df: pd.DataFrame, feature1: str, feature2: str, title1: str, title2: str
) -> None:
    """Plots count plot and proportion plots of categorical features per target variable.
    params: df: usable pd.DataFrame;
    feature1: str - name of the categorical column, which values to plot;
    feature2: str - name of the target feature;"""
    f, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.countplot(
        x=feature1,
        hue=feature2,
        data=df,
        # order=df[feature1].value_counts().index,
        ax=ax[0],
    )
    ax[0].set_title(title1)
    sns.histplot(
        data=df,
        x=feature1,
        hue=feature2,
        multiple="fill",
        stat="proportion",
        discrete=True,
        shrink=0.8,
        ax=ax[1],
    )
    ax[1].set_title(title2)
    plt.close(2)
    plt.show()


def plot_kde(df: pd.DataFrame, feature1: str, feature2: str) -> None:
    """Plots KDE plot distribution of numerical feature with categorical "hue".
    params: df: usable pd.DataFrame;
            feature1: str - name of the numerical column, which values to plot;
            feature2: str - name of the categorical binary (1 or 0) column used as a "hue";
    """
    plt.figure(figsize=(15, 5))
    plt.title(f"KDE Plot: {feature1} vs. {feature2}", fontsize=30, fontweight="bold")
    ax = sns.kdeplot(
        df[df[feature2] == 1][feature1],
        color="red",
        label=f"was {feature2}",
        lw=2,
        legend=True,
    )
    ax1 = sns.kdeplot(
        df[df[feature2] == 0][feature1],
        color="blue",
        label=f"not {feature2}",
        lw=2,
        legend=True,
    )
    legend = ax.legend(loc="upper right")
    ax.yaxis.grid(True)
    sns.despine(right=True, left=True)
    plt.tight_layout()


def compare_two_samples(df1: pd.DataFrame, df2: pd.DataFrame, feature1: str) -> None:
    """Compare the distributions of a specific feature between two samples using Kernel Density Estimation (KDE) plots.
    Parameters:
    - df1 (pd.DataFrame): First sample DataFrame.
    - df2 (pd.DataFrame): Second sample DataFrame.
    - feature1 (str): The feature to compare between the two samples."""

    plt.figure(figsize=(15, 5))
    plt.title(
        f"distributions of {feature1} of first sample and second sample",
        fontsize=30,
        fontweight="bold",
    )
    ax = sns.kdeplot(
        data=df1[feature1],
        color="red",
        label=f"first sample {feature1}",
        lw=2,
        legend=True,
    )
    ax1 = sns.kdeplot(
        data=df2[feature1],
        color="blue",
        label=f"second sample {feature1}",
        lw=2,
        legend=True,
    )
    legend = ax.legend(loc="upper right")
    ax.yaxis.grid(True)
    sns.despine(right=True, left=True)
    plt.tight_layout()


# STATISTICAL INFERENCE


def power_test_one_tail(proportion1: float, proportion2: float) -> None:
    """Prints the needed sample size to avoid p-hacking for one tail test.
    params: proportion1: float - proportion of positive target values in first group of interest;
            proportion2: float - proportion of positive target values in second group of interest;
    """
    effect_size = sms.proportion_effectsize(proportion1, proportion2)
    required_n = sms.NormalIndPower().solve_power(
        effect_size, power=0.8, alpha=0.05, ratio=1
    )
    required_n = ceil(required_n)
    print(f" Required sample size:{required_n}")


def calc_pi_t_test_proportions(df_emp: pd.DataFrame) -> None:
    """Calculates pi value using t test for difference in proportions
    params: df_emp: DataFrame
    preparation of df:
        df_emp = df.groupby("feature of interest")[["target value"]].agg(["sum", "count"])
        df_emp = df_emp.droplevel(0, axis=1).reset_index()
        df_emp["proportion"] = df_emp["sum"] / df_emp["count"]
        df_emp"""
    p_both = (df_emp.iloc[0]["sum"] + df_emp.iloc[1]["sum"]) / (
        df_emp.iloc[0]["count"] + df_emp.iloc[1]["count"]
    )
    va = p_both * (1 - p_both)
    se = np.sqrt(va * (1 / df_emp.iloc[0]["count"] + 1 / df_emp.iloc[1]["count"]))
    test_stat = (df_emp.iloc[0]["proportion"] - df_emp.iloc[1]["proportion"]) / se
    pvalue = stats.norm.sf(abs(test_stat))
    print(f"Pi value for diff in proportions using t test:{pvalue}")


def calc_confid_intervals(df_emp: pd.DataFrame) -> None:
    """Calculates confidens intervals for difference in proportions
        using Confidence level of 95%, significant level alpha = 0.05
    params: df_emp: DataFrame
    preparation of df:
        df_emp = df.groupby("feature of interest")[["target value"]].agg(["sum", "count"])
        df_emp = df_emp.droplevel(0, axis=1).reset_index()
        df_emp["proportion"] = df_emp["sum"] / df_emp["count"]
        df_emp"""
    # SE0
    p0 = df_emp.iloc[0]["proportion"]
    n0 = df_emp.iloc[0]["sum"]  # Total number of purchases
    st_err0 = p0 * (1 - p0) / n0

    # SE1
    p1 = df_emp.iloc[1]["proportion"]
    n1 = df_emp.iloc[1]["sum"]  # Total number of purchases
    st_err1 = p1 * (1 - p1) / n1

    # sqrt(SE0+SE1) = Standar error
    se = np.sqrt(st_err0 + st_err1)

    # First interval value
    intv_1 = (p0 - p1) - 1.96 * se
    # Second interval value
    intv_0 = (p0 - p1) + 1.96 * se
    print(f"Confidence interval is   {intv_1.round(2)} - {intv_0.round(2)}")


# MODELING


def replace_rare_categorical_values(df: pd.DataFrame, threshold=0.3):
    """Replaces rare categorical values in a DataFrame with a common label.
    Parameters:
    - df (pd.DataFrame): Input DataFrame containing categorical columns.
    - threshold (float, optional): Threshold for considering values as rare. Defaults to 0.3.
    Returns: pd.DataFrame: DataFrame with rare values replaced."""
    result_df = df.copy()
    for column in df.select_dtypes(include=["object"]):
        value_counts = df[column].value_counts()
        threshold_count = threshold * value_counts.iloc[0]

        rare_values = value_counts[value_counts < threshold_count].index

        result_df[column] = result_df[column].apply(
            lambda x: "Rare feature" if x in rare_values and not pd.isna(x) else x
        )

    return result_df


def get_risk_times_dti(df: pd.DataFrame):
    df["risk_times_dti"] = df["risk_score"] * df["debt_to_income_ratio"]
    return df.drop(["risk_score", "debt_to_income_ratio"], axis=1)


class PrinterTransformer(BaseEstimator, TransformerMixin):
    """Transformer class for printing the input data.
    Parameters: None
    Methods:
    - fit(X, y=None): Fit method for compatibility with scikit-learn pipeline. Returns self.
    - transform(X): Transform method to print the input data and return it unchanged."""

    def __init__(
        self,
    ):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X)
        return X


class ArrayNamer(BaseEstimator, TransformerMixin):
    """
    Transformer for naming columns of an array.

    This transformer takes an input array and assigns column names to it.
    Additionally it infers tha data types.
    This transformer is useful when the pipeline contains sklearn
    ColumnTransformer objects.

    Parameters:
    col_names : list of column names to assign to the array.

    Methods:
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X)
        Transform the input array by assigning column names.

    Returns:
    X : DataFrame with column names assigned to the input array.
    """

    def __init__(self, col_names):
        """
        Initialize the ArrayNamer.

        Parameters:
            col_names: A list of column names to assign to the array.
        """
        self.col_names = col_names

    def fit(self, X, y=None):
        """
        Fit the transformer to the input data.
        Parameters:
            X : array Input data.
        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Transform the input array by assigning column names.
        Parameters:
            X : array Input data.
        Returns:
            X : DataFrame DataFrame with column names assigned to the input array.
        """
        X_df = pd.DataFrame(X, columns=self.col_names).infer_objects()
        return X_df


def confusion_matrix_normalized(
    y_val: pd.DataFrame, pred_y: pd.DataFrame, labels: list
) -> None:
    """Plots normalized confusion matrix.
    :param: y_val: pd.DataFrame with features;
            pred_y: pd.DataFrame dependent variable;
            labels: matrix labels
    """
    cm = confusion_matrix(y_val, pred_y)
    # Normalise
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cmn, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show(block=False)


def big_confusion_matrix_normalized(
    y_val: pd.DataFrame, pred_y: pd.DataFrame, labels: list
) -> None:
    """Plots normalized confusion matrix.
    :param: y_val: pd.DataFrame with features;
            pred_y: pd.DataFrame dependent variable;
            labels: matrix labels
    """
    cm = confusion_matrix(y_val, pred_y)
    # Normalise
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(25, 25))
    sns.heatmap(cmn, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show(block=False)
