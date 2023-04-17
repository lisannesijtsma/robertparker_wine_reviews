""" Dashboard showing insights in Robert Parker wine reviews """

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from datetime import datetime
from pathlib import Path
from settings import settings

data_file = settings.cleaned_data_file
RATING = settings.rating_col

st.set_page_config(
    page_title="Robert Parkers wine review insights",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://google.com",
        "About": 'Robert Parker wine review insights is made by Lisanne Sijtsma. For more information, documentation and GitHub Repository see "Get Help"',
    },
)
st.title("Robert Parker wine reviews dashboard")
st.text(f"Last updated: {datetime.now(): %A %B %d, %Y %H:%M}")


#### LOAD DATA
# Caching docs: https://docs.streamlit.io/library/get-started/create-an-app
@st.cache_data
def load_data(nrows: int, file: Path) -> pd.DataFrame:
    """Load data and return in a DataFrame"""
    data = pd.read_csv(file, delimiter=",", index_col=0, nrows=nrows)
    return data


data_load_state = st.text("Loading data...")
data = load_data(20000, data_file)
data_load_state.text("Data successfully loaded!")
st.write("---")


#### SIDEBAR FILTERS
def filter_dataset(
    subset: pd.DataFrame, filter: str, filter_values: list
) -> pd.DataFrame:
    """Return filtered DataFrame based on given filter"""
    return subset[subset[filter].isin(filter_values)]


def create_multiselect_filter(filter: str, subset: pd.DataFrame) -> pd.DataFrame:
    """Create a multiselect and a checkbox to select all values"""
    container = st.sidebar.container()
    all = st.sidebar.checkbox(f"Select all values of {filter}")
    if all:
        filter_values = container.multiselect(
            f"Select the {filter}",
            options=pd.unique(subset[filter]),
            default=pd.unique(subset[filter]),
        )
    else:
        filter_values = container.multiselect(
            f"Select the {filter}",
            options=pd.unique(subset[filter]),
            default=subset[filter].head(1),
        )
    return filter_dataset(subset, filter, filter_values)

st.sidebar.title(f"Filters")

rows_filter = st.sidebar.slider(
    "Select the number of (random) reviews:",
    min_value=100,
    max_value=len(data.index),
    value=10000,
    step=100,
)
subset = data.sample(n=rows_filter)

vintage_filter = st.sidebar.slider(
    "Select a range for the vintage:",
    min_value=int(subset.vintage.min()),
    max_value=int(subset["vintage"].max()),
    value=(1990, 2010),
    step=1,
)
filtered_subset = subset[
    (subset["vintage"] >= vintage_filter[0]) & (subset["vintage"] <= vintage_filter[1])
]

rating_filter = st.sidebar.slider(
    "Select a range for the rating:",
    min_value=int(filtered_subset[RATING].min()),
    max_value=int(filtered_subset[RATING].max()),
    value=(0, 100),
    step=1,
)
filtered_subset = filtered_subset[
    (filtered_subset[RATING] >= rating_filter[0])
    & (filtered_subset[RATING] <= rating_filter[1])
]

container_filters = ["country", "reviewer", "color", "type", "sweetness"]
for filter in container_filters:
    filtered_subset = create_multiselect_filter(filter, filtered_subset)



#### CHARTS
fig = plt.figure(figsize=(22, 10))
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.subheader("Number of reviews by vintage")
    sns.set_style("white")
    sns.countplot(data=filtered_subset, y="vintage")
    st.pyplot(fig=fig, clear_figure=True)

with fig_col2:
    st.subheader("Number of reviews by reviewer")
    sns.set_style("white")
    sns.countplot(data=filtered_subset, y="reviewer")
    st.pyplot(fig=fig, clear_figure=True)

st.subheader("Median rating over the years by reviewer")
fig = plt.figure(figsize=(35, 10))
sns.set_style("white")
ax = sns.lineplot(
    data=filtered_subset,
    x="vintage",
    y=RATING,
    hue="reviewer",
    estimator="median",
    markers=True,
    dashes=True,
)
ax.set(xticks=filtered_subset["vintage"].values)
st.pyplot(fig=fig, clear_figure=True)

st.write("---")
boxplot_filter = st.selectbox(f"Select a variable specifically for the boxplot",
                                     options=["country", "reviewer", "color", "type", "sweetness"])

st.subheader(f"Boxplot of ratings by {boxplot_filter}")
fig = plt.figure(figsize=(20,10))
sns.boxplot(data=subset, x=boxplot_filter, y=RATING)
st.pyplot(fig=fig, clear_figure=True)
