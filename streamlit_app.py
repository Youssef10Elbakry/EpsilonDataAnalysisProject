import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="FIFA Players Data Analysis",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ FIFA Players Data Analysis")
st.markdown(
    "Explore the FIFA players dataset, data quality, player ages, "
    "ratings, potential, wages, nationalities, and positions."
)

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    return data


default_path = Path(__file__).parent / "fifa_players.csv"


try:
    if default_path.exists():
        original_df = load_data(str(default_path))
    else:
        st.warning(
            "Place `fifa_players.csv` in the same folder as `app.py`, "
            "or upload the CSV using the sidebar."
        )
        st.stop()
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

df = original_df.copy()

required_columns = [
    "name", "age", "overall_rating", "potential",
    "positions", "nationality", "wage_euro"
]
missing_required = [column for column in required_columns if column not in df.columns]

if missing_required:
    st.error(
        "The dataset is missing required column(s): "
        + ", ".join(missing_required)
    )
    st.write("Columns found in your CSV:", df.columns.tolist())
    st.stop()

numeric_columns = [
    "age", "overall_rating", "potential", "wage_euro", "value_euro",
    "height_cm", "weight_kgs"
]
for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

st.sidebar.header("Player Filters")

nationality_options = sorted(df["nationality"].dropna().astype(str).unique())
selected_nationalities = st.sidebar.multiselect(
    "Nationality (leave empty for all)",
    options=nationality_options
)

position_series = (
    df["positions"].dropna().astype(str)
    .str.split(",")
    .explode()
    .str.strip()
)
position_options = sorted(position_series[position_series.ne("")].unique())
selected_positions = st.sidebar.multiselect(
    "Position (leave empty for all)",
    options=position_options
)

valid_ages = df["age"].dropna()
if not valid_ages.empty:
    age_min = int(valid_ages.min())
    age_max = int(valid_ages.max())
    age_range = st.sidebar.slider(
        "Age range",
        min_value=age_min,
        max_value=max(age_min, age_max),
        value=(age_min, max(age_min, age_max))
    )
else:
    age_range = None

valid_ratings = df["overall_rating"].dropna()
if not valid_ratings.empty:
    rating_min = int(valid_ratings.min())
    rating_max = int(valid_ratings.max())
    rating_range = st.sidebar.slider(
        "Overall rating range",
        min_value=rating_min,
        max_value=max(rating_min, rating_max),
        value=(rating_min, max(rating_min, rating_max))
    )
else:
    rating_range = None

filtered_df = df.copy()

if selected_nationalities:
    filtered_df = filtered_df[
        filtered_df["nationality"].astype(str).isin(selected_nationalities)
    ]

if selected_positions:
    selected_position_set = set(selected_positions)
    position_mask = filtered_df["positions"].apply(
        lambda value: bool(
            selected_position_set.intersection(
                part.strip() for part in str(value).split(",")
            )
        ) if pd.notna(value) else False
    )
    filtered_df = filtered_df[position_mask]

if age_range is not None:
    filtered_df = filtered_df[
        filtered_df["age"].between(age_range[0], age_range[1])
    ]

if rating_range is not None:
    filtered_df = filtered_df[
        filtered_df["overall_rating"].between(rating_range[0], rating_range[1])
    ]

st.header("1. Data Exploration")

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Rows", f"{len(original_df):,}")
metric2.metric("Columns", f"{len(original_df.columns):,}")
metric3.metric("Duplicate rows", f"{original_df.duplicated().sum():,}")
metric4.metric("Filtered players", f"{len(filtered_df):,}")

with st.expander("Preview the original dataset"):
    st.dataframe(original_df.head(20), use_container_width=True)

with st.expander("Column information (equivalent to df.info())"):
    info_df = pd.DataFrame({
        "column": original_df.columns,
        "non_null_count": [original_df[col].notna().sum() for col in original_df.columns],
        "missing_count": [original_df[col].isna().sum() for col in original_df.columns],
        "data_type": [str(original_df[col].dtype) for col in original_df.columns],
    })
    st.dataframe(info_df, use_container_width=True)

with st.expander("Descriptive statistics (df.describe().round(2))"):
    st.dataframe(original_df.describe().round(2), use_container_width=True)

with st.expander("Missing values before cleaning"):
    missing_before = original_df.isna().sum().rename("missing_values").to_frame()
    missing_before = missing_before.sort_values("missing_values", ascending=False)
    st.dataframe(missing_before, use_container_width=True)

st.header("2. Data Cleaning")

columns_to_drop = [
    "national_team", "national_rating", "national_team_position",
    "national_jersey_number", "birth_date", "release_clause_euro"
]
existing_drop_columns = [col for col in columns_to_drop if col in df.columns]
clean_df = df.drop(columns=existing_drop_columns, errors="ignore").copy()

for column in ["value_euro", "wage_euro"]:
    if column in clean_df.columns:
        mean_value = clean_df[column].mean()
        if pd.notna(mean_value):
            clean_df[column] = clean_df[column].fillna(mean_value)

st.write("Columns removed, when present:", existing_drop_columns or "None")
st.write("The notebook's missing-value treatment fills missing `value_euro` and `wage_euro` values with their column means.")

clean_col1, clean_col2 = st.columns(2)
with clean_col1:
    st.subheader("Missing values after cleaning")
    missing_after = clean_df.isna().sum().rename("missing_values").to_frame()
    st.dataframe(
        missing_after.sort_values("missing_values", ascending=False),
        use_container_width=True
    )
with clean_col2:
    st.subheader("Cleaned data preview")
    st.dataframe(clean_df.head(10), use_container_width=True)

with st.expander("Descriptive statistics after cleaning"):
    st.dataframe(clean_df.describe().round(2), use_container_width=True)

analysis_df = clean_df.copy()

if selected_nationalities:
    analysis_df = analysis_df[
        analysis_df["nationality"].astype(str).isin(selected_nationalities)
    ]
if selected_positions:
    selected_position_set = set(selected_positions)
    position_mask = analysis_df["positions"].apply(
        lambda value: bool(
            selected_position_set.intersection(
                part.strip() for part in str(value).split(",")
            )
        ) if pd.notna(value) else False
    )
    analysis_df = analysis_df[position_mask]
if age_range is not None:
    analysis_df = analysis_df[analysis_df["age"].between(*age_range)]
if rating_range is not None:
    analysis_df = analysis_df[
        analysis_df["overall_rating"].between(*rating_range)
    ]

if analysis_df.empty:
    st.warning("No players match the selected filters. Change the filters in the sidebar.")
    st.stop()

st.header("3. Age Analysis")
age_col1, age_col2 = st.columns(2)

with age_col1:
    st.subheader("Age descriptive statistics")
    st.dataframe(analysis_df["age"].describe().to_frame("age").round(2))

with age_col2:
    st.subheader("Player count by age")
    age_counts = analysis_df["age"].value_counts().sort_index()
    st.dataframe(age_counts.rename("number_of_players").to_frame())

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(analysis_df["age"].dropna(), bins=range(15, 46), edgecolor="black")
ax.set_title("Distribution of Player Ages")
ax.set_xlabel("Age")
ax.set_ylabel("Number of Players")
st.pyplot(fig)
plt.close(fig)

st.header("4. Overall Rating Analysis")
rating_col1, rating_col2 = st.columns(2)

with rating_col1:
    st.subheader("Overall rating descriptive statistics")
    st.dataframe(
        analysis_df["overall_rating"].describe().to_frame("overall_rating").round(2)
    )

with rating_col2:
    st.subheader("Player count by overall rating")
    rating_counts = analysis_df["overall_rating"].value_counts().sort_index()
    st.dataframe(rating_counts.rename("number_of_players").to_frame())

fig, ax = plt.subplots(figsize=(10, 5))
ratings = analysis_df["overall_rating"].dropna()
if not ratings.empty:
    ax.hist(
        ratings,
        bins=range(int(ratings.min()), int(ratings.max()) + 2),
        edgecolor="black"
    )
ax.set_title("Distribution of Player Overall Ratings")
ax.set_xlabel("Overall Rating")
ax.set_ylabel("Number of Players")
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3))
ax.boxplot(analysis_df["overall_rating"].dropna(), vert=False)
ax.set_title("Box Plot of Player Overall Ratings")
ax.set_xlabel("Overall Rating")
st.pyplot(fig)
plt.close(fig)

st.header("5. Position Analysis")

position_combinations = (
    analysis_df["positions"].dropna().astype(str)
    .value_counts()
    .rename_axis("positions")
    .rename("number_of_players")
    .to_frame()
)
st.subheader("Most common listed position combinations")
st.dataframe(position_combinations.head(20), use_container_width=True)

df_positions = analysis_df.assign(
    positions=analysis_df["positions"].astype("string").str.split(",")
).explode("positions")
df_positions["positions"] = df_positions["positions"].astype("string").str.strip()
df_positions = df_positions[df_positions["positions"].notna() & df_positions["positions"].ne("")]

position_stats = (
    df_positions.groupby("positions")["overall_rating"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
    .rename(columns={"mean": "average_rating", "count": "number_of_players"})
)
st.subheader("Average overall rating by individual position")
st.dataframe(position_stats.round(2), use_container_width=True)

if not position_stats.empty:
    fig, ax = plt.subplots(figsize=(10, max(5, len(position_stats) * 0.3)))
    position_stats["average_rating"].sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Average Overall Rating by Position")
    ax.set_xlabel("Average Overall Rating")
    ax.set_ylabel("Position")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.header("6. Overall Rating vs. Potential")
rating_potential_corr = analysis_df[["overall_rating", "potential"]].corr()
st.subheader("Correlation")
st.dataframe(rating_potential_corr.round(3), use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 5))
scatter_data = analysis_df.dropna(subset=["overall_rating", "potential"])
ax.scatter(
    scatter_data["overall_rating"],
    scatter_data["potential"],
    alpha=0.5
)
ax.set_title("Overall Rating vs. Potential")
ax.set_xlabel("Overall Rating")
ax.set_ylabel("Potential")
st.pyplot(fig)
plt.close(fig)

analysis_df["potential_gap"] = (
    analysis_df["potential"] - analysis_df["overall_rating"]
)
st.subheader("Top 10 players with the largest potential gap")
potential_gap_table = (
    analysis_df[["name", "overall_rating", "potential", "potential_gap"]]
    .sort_values("potential_gap", ascending=False)
    .head(10)
)
st.dataframe(potential_gap_table, use_container_width=True)

st.header("7. Wage vs. Overall Rating")
wage_rating_corr = analysis_df[["wage_euro", "overall_rating"]].corr()
st.subheader("Correlation")
st.dataframe(wage_rating_corr.round(3), use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 5))
wage_data = analysis_df.dropna(subset=["overall_rating", "wage_euro"])
ax.scatter(
    wage_data["overall_rating"],
    wage_data["wage_euro"],
    alpha=0.5
)
ax.set_title("Wage vs. Overall Rating")
ax.set_xlabel("Overall Rating")
ax.set_ylabel("Weekly Wage (€)")
st.pyplot(fig)
plt.close(fig)

st.header("8. Correlation Heatmap")
numeric_df = analysis_df.select_dtypes(include="number")
corr_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    annot=False,
    linewidths=0.5,
    ax=ax
)
ax.set_title("Correlation Heatmap of FIFA Player Attributes")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


st.header("9. Nationality Analysis")

average_rating = (
    analysis_df.groupby("nationality")["overall_rating"]
    .mean()
    .sort_values(ascending=False)
)
top_10 = average_rating.head(10)

st.subheader("Top 10 nationalities by average player rating")
st.dataframe(
    top_10.rename("average_rating").to_frame().round(2),
    use_container_width=True
)

fig, ax = plt.subplots(figsize=(10, 6))
top_10.sort_values().plot(kind="barh", ax=ax)
ax.set_title("Top 10 Nationalities by Average Player Rating")
ax.set_xlabel("Average Overall Rating")
ax.set_ylabel("Nationality")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

nationality_stats = (
    analysis_df.groupby("nationality")["overall_rating"]
    .agg(["mean", "count"])
    .query("count > 100")
    .sort_values("mean", ascending=False)
    .rename(columns={
        "mean": "average_rating",
        "count": "number_of_players"
    })
)

st.subheader("Nationality statistics (more than 100 players)")
st.dataframe(nationality_stats.round(2), use_container_width=True)

top_10_cleaned = nationality_stats.head(10)
st.subheader("Top 10 nationalities with more than 100 players")
st.dataframe(top_10_cleaned.round(2), use_container_width=True)

if not top_10_cleaned.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10_cleaned["average_rating"].sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top 10 Nationalities by Average Player Rating")
    ax.set_xlabel("Average Overall Rating")
    ax.set_ylabel("Nationality")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


st.header("10. Highest-Paid Players")
highest_paid = (
    analysis_df.sort_values("wage_euro", ascending=False)
    [["name", "nationality", "overall_rating", "wage_euro"]]
    .head(10)
)
st.dataframe(highest_paid, use_container_width=True)

highest_paid_chart = highest_paid.sort_values("wage_euro")
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(highest_paid_chart["name"], highest_paid_chart["wage_euro"])
ax.set_title("Top 10 Highest-Paid FIFA Players")
ax.set_xlabel("Wage (€)")
ax.set_ylabel("Player")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


st.header("11. Players with the Highest Potential")
highest_potential = (
    analysis_df.sort_values("potential", ascending=False)
    [["name", "nationality", "overall_rating", "potential"]]
    .head(10)
)
st.dataframe(highest_potential, use_container_width=True)

highest_potential_chart = highest_potential.sort_values("potential")
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    highest_potential_chart["name"],
    highest_potential_chart["potential"]
)
ax.set_title("Top 10 Players by Potential")
ax.set_xlabel("Potential Rating")
ax.set_ylabel("Player")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

