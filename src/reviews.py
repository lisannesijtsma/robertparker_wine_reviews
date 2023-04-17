""" Load, explore, prepare and get insights in the reviews """

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
from loguru import logger
from pathlib import Path
from pydantic import BaseSettings
from scipy.stats import chi2
from settings import settings, logger

IMAGES = settings.images_dir
RATING = settings.rating_col


def concat_data(dir: Path) -> pd.DataFrame:
    """Return a concatenated DataFrame of all csv files in the raw data directory"""
    data = []
    for file in os.listdir(dir):
        if file.endswith(".csv"):
            logger.info(f"Reading file {file}")
            temp = pd.read_csv(dir / file, delimiter=",", index_col=0)
            data.append(temp)
    df = pd.concat(data, axis=0)
    logger.info(f"Total size of DataFrame: {df.shape}")
    return df


def write_plot(plot: plt, name: str) -> None:
    """Save a plot with given name to the images directory"""
    plot.savefig(f"{IMAGES}/{name}.pdf")


def get_countplot(df: pd.DataFrame, cols: list[str]) -> None:
    """Get a countplot (in subplot) of given columns and save"""
    plot_rows = int(len(cols) / 2)

    fig, axs = plt.subplots(nrows=plot_rows, ncols=2, figsize=(30, 10))
    fig.suptitle("Count number of occurences in dataset", fontsize=18, y=0.95)
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.3)

    for feature, ax in zip(cols, axs.ravel()):
        ax = sns.countplot(
            data=df, y=feature, order=df[feature].value_counts().index, ax=ax
        )
        ax.bar_label(ax.containers[0])
        ax.set_title(f"Number of reviews per {feature}")
        ax.set_xlabel(f"Number of reviews")
        ax.set_ylabel(f"{feature}")
    write_plot(plt, "data_exploration_categorical_occurences")


def data_exploration(df: pd.DataFrame) -> None:
    """Exploration counts in the data before cleaning"""
    plot_features = ["sweetness", "type", "color", "reviewed_by"]
    get_countplot(df, plot_features)


def clean_missing_values(df: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
    """Clean missing values based on given method"""
    logger.info(f"Current number of rows in DataFrame: {len(df.index)}")
    logger.info(
        f"Number of rows in the {col} column with missing values: {df[col].isnull().sum()}"
    )
    if method == "drop":
        df.dropna(axis=0, subset=[col], inplace=True)
    elif method == "fill":
        df[col].fillna(value=0, inplace=True)
    logger.info(f"Number of rows in DataFrame: {len(df.index)}")
    return df


def clean_rating(rating: pd.Series) -> pd.Series:
    """Clean the rating and return in a DataFrame Series"""
    # Some ratings consists of multiple ratings in one, for example: (86 - 88)+
    # Take the average of the ratings in the "tuple" (actually it's a string)
    if rating.startswith("("):
        numbers_rating = (float, re.findall(r"\d+", rating))[1]
        numbers_rating = [float(x) for x in numbers_rating]
        average = sum(numbers_rating) / len(numbers_rating)
        logger.info(f"Replaced {rating} with {average}")  # Write to log for auditting purposes
        rating = str(average)
    rating = re.sub(r"[^\d\.]", "", rating)
    if (
        rating == ""
    ):  # Return None for new empty ratings after removing non-digit characters (for example if rating="?")
        return None
    return float(rating)


def get_location_data(df: pd.DataFrame, location_vars: list[str]) -> pd.DataFrame:
    """Split location data and return with new location variables in DataFrame"""
    location_splitted = df["from_location"].str.split(",", expand=True)
    for x in location_splitted:
        df[location_vars[x]] = location_splitted[x]
    return df


def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich DataFrame with new columns constructed from existing columns"""
    df["length_content"] = df["content"].str.len()
    df = clean_missing_values(df, "length_content", "fill")

    regex = re.compile("^(\d{4})")
    df["vintage"] = df["title"].str.extract(regex)
    df = clean_missing_values(df, "vintage", "drop")
    df["vintage"] = df["vintage"].astype("int")

    regex = re.compile(r"\•(.+)")
    df["name"] = df["title"].str.extract(regex)

    location_vars = [
        "country",
        "region",
        "sub_region",
        "appellation",
        "sub_appellation",
    ]
    df = get_location_data(df, location_vars)
    return df


def drop_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Remove for further process unnecessary column"""
    logger.info(f'Column "{col}" dropped.')
    df = df.drop([col], axis=1)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the data and return a cleaned DataFrame"""
    columns_to_remove = ["issue_date", "source"]
    for col in columns_to_remove:
        df = drop_column(df, col)

    columns = df.loc[:, df.columns != "content"].columns
    for col in columns:
        df = clean_missing_values(df, col, "drop")

    df["rating_cleaned"] = df["rating"].apply(clean_rating)
    df = clean_missing_values(df, "rating_cleaned", "drop")
    df = enrich_data(df)

    columns_to_remove = ["title", "rating", "from_location"]
    for col in columns_to_remove:
        df = drop_column(df, col)
    df["color"].replace(to_replace="Ros", value="Rose", inplace=True)
    df.rename(columns={"reviewed_by": "reviewer"}, inplace=True)
    df.rename(columns={"rating_cleaned": "rating"}, inplace=True)
    return df


def describe_data(df: pd.DataFrame) -> None:
    """Describe the cleaned DataFrame"""
    logger.info(f"Columns with number of missing values after data preparation: \n{df.isna().sum()}")
    logger.info(f"Describe the float variables: \n{df.describe()}")
    for i in df.columns:
        logger.info(f"Number of unique values in {i} column: {df[i].nunique()}")


def check_normal_dist(df: pd.DataFrame) -> None:
    """Check and test the normal distribution by histogram and lilliefors test"""
    plt.figure(figsize=(10, 4))
    ax = sns.histplot(data=df, x=RATING, stat="count", binwidth=1)
    ax.set_title(f"Histogram of Rating")
    write_plot(plt, "histogram_rating")

    plt.figure(figsize=(10, 4))
    ax = sns.boxplot(data=df, y=RATING)
    ax.set_title(f"Boxplot of ratings")
    write_plot(plt, "boxplot_rating")
    
    values = np.array(df[RATING])
    result = sm.stats.diagnostic.lilliefors(values, dist="norm", pvalmethod="table")

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["Test statistic", "P-value"]),
                cells=dict(values=[result[0], result[1]]),
            )
        ]
    )
    fig.write_image(rf"{IMAGES}/test_normalization_results.pdf")


def create_boxplots(df: pd.DataFrame, var: str) -> None:
    """Create boxplots image for given variable"""
    plt.figure(figsize=(20, 10))
    ax = sns.boxplot(data=df, x=var, y=RATING)
    ax.set_title(f"Rating by {var}")
    name = f"boxplot_{var}"
    write_plot(plt, name)


def kruskal_wallis_test(test_df, num_field, cat_field, alpha):
    """Execute kruskal wallis test and return the teststatistics"""
    """Based on: https://medium.com/@indkulwardana8/application-on-kruskal-wallis-test-26c92af132f3"""
    k = test_df[cat_field].nunique()
    N = test_df[cat_field].count()

    df = test_df.copy()

    df["rank"] = df[num_field]
    df["rank"] = df["rank"].rank(ascending=True)

    table1 = df.groupby(cat_field).sum()
    table2 = df.groupby(cat_field).count()

    sub_component = 0
    for i in range(0, k):
        sub_component = (table1.iloc[i, 1] ** 2 / table2.iloc[i, 1]) + sub_component

    test_statistic = ((12 / (N * (N + 1))) * sub_component) - 3 * (N + 1)
    degrees_of_freedom = k - 1
    a = alpha / 100
    chi_critical_value = round(chi2.isf(q=a, df=degrees_of_freedom), 2)
    p_value = chi2.sf(test_statistic, degrees_of_freedom, loc=0, scale=1)
    return (round(test_statistic, 2), chi_critical_value, p_value, a)


def get_test_results(df: pd.DataFrame, var: str) -> dict():
    """Get test results and return in a dictionary"""
    test_df = df[[RATING, var]]
    results = kruskal_wallis_test(test_df, num_field=RATING, cat_field=var, alpha=5)
    new_result = {
        "Variable": var,
        "Test Statistic": results[0],
        "Chi critical value": results[1],
        "P-Value": results[2],
        "Alpha": results[3],
    }
    return new_result


def get_statistics(df: pd.DataFrame) -> None:
    """Get all different statistics and save to images"""
    check_normal_dist(df)

    test_results = []
    vars = [
        "color",
        "type",
        "reviewer",
        "sweetness",
        "producer",
        "variety",
        "vintage",
        "appellation",
        "country",
    ]
    for var in vars:
        create_boxplots(df, var)
        result = get_test_results(df, var)
        test_results.append(result)

    df_test_results = pd.DataFrame(test_results)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=df_test_results.columns),
                cells=dict(
                    values=[
                        df_test_results["Variable"],
                        df_test_results["Test Statistic"],
                        df_test_results["Chi critical value"],
                        df_test_results["P-Value"],
                        df_test_results["Alpha"],
                    ]
                ),
            )
        ]
    )
    fig.write_image(rf"{IMAGES}/test_results.pdf")

    plt.figure(figsize=(10,5))
    ax = sns.lineplot(data=df, x='vintage', y=RATING, estimator='median', markers=True, dashes=True, err_style='band')
    ax.set_title(f"Median rating over the years")
    ax.set(xticks=df['vintage'].values)
    write_plot(plt, "median_rating_over_the_years")


def get_insights(presets: BaseSettings) -> None:
    """Get insights in the reviews and save cleaned dataset"""
    raw_data_dir: Path = presets.raw_data_dir
    cleaned_data_file: Path = presets.cleaned_data_file

    df = concat_data(raw_data_dir)
    data_exploration(df)
    df = prepare_data(df)
    describe_data(df)
    get_statistics(df)

    logger.info(f"Writing cleaned datafile to {cleaned_data_file}")
    df.to_csv(cleaned_data_file)
