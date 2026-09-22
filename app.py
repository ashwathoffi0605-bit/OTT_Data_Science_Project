import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================

st.set_page_config(
    page_title="OTT Performance & Cinema Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme Custom Styling
st.markdown("""
<style>
    /* Dark Theme Global Styling */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sleek Metric Card Container */
    .metric-card {
        background: linear-gradient(135deg, #1A1F2C 0%, #121620 100%);
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #E50914;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-delta {
        font-size: 0.8rem;
        color: #00D26A;
        margin-top: 4px;
    }

    /* Platform Pills */
    .pill {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .pill-netflix { background-color: #E50914; color: white; }
    .pill-prime { background-color: #00A8E1; color: white; }
    .pill-hulu { background-color: #1CE783; color: #0E1117; }
    .pill-disney { background-color: #113CCF; color: white; }

    /* Prediction Result Cards */
    .pred-card-high {
        background: linear-gradient(135deg, rgba(0, 210, 106, 0.15) 0%, rgba(20, 83, 45, 0.3) 100%);
        border: 1px solid #00D26A;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
    }
    .pred-card-low {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(127, 29, 29, 0.3) 100%);
        border: 1px solid #EF4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
    }

    /* Dataframe container padding */
    .stDataFrame {
        border: 1px solid #2D3748;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for custom metric card
def render_metric_card(label, value, delta=None):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# Matplotlib dark styling helper
def set_dark_plot_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#161E2E',
        'axes.facecolor': '#161E2E',
        'axes.edgecolor': '#2D3748',
        'axes.labelcolor': '#E2E8F0',
        'xtick.color': '#94A3B8',
        'ytick.color': '#94A3B8',
        'grid.color': '#2D3748',
        'grid.alpha': 0.5,
        'font.sans-serif': 'DejaVu Sans'
    })


# ==============================================================================
# 2. ROBUST PATH RESOLUTION & DATA PIPELINE
# ==============================================================================

CURRENT_DIR = Path(__file__).resolve().parent

# Check potential roots (supports execution from repository root or dashboard/ directory)
if (CURRENT_DIR / "data").exists():
    PROJECT_ROOT = CURRENT_DIR
elif (CURRENT_DIR.parent / "data").exists():
    PROJECT_ROOT = CURRENT_DIR.parent
else:
    PROJECT_ROOT = CURRENT_DIR

OTT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "ott_cleaned.csv"
OTT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "MoviesOnStreamingPlatforms.csv"
INDIAN_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "indian movies.csv"


@st.cache_data(show_spinner=False)
def load_ott_data():
    """Loads and sanitizes the OTT streaming catalog dataset."""
    df = None
    if OTT_PROCESSED_PATH.exists():
        df = pd.read_csv(OTT_PROCESSED_PATH)
    elif OTT_RAW_PATH.exists():
        df = pd.read_csv(OTT_RAW_PATH)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        # Clean Rotten Tomatoes
        df["Rotten Tomatoes"] = (
            df["Rotten Tomatoes"]
            .astype(str)
            .str.replace("/100", "", regex=False)
            .str.strip()
        )
        df["Rotten Tomatoes"] = pd.to_numeric(df["Rotten Tomatoes"], errors="coerce")
        df["Rotten Tomatoes"] = df["Rotten Tomatoes"].fillna(df["Rotten Tomatoes"].median())
        df["Age"] = df["Age"].fillna("Not Rated")
    else:
        st.error(
            f"❌ Required OTT dataset missing!\n\n"
            f"Looked for either:\n"
            f"- `{OTT_PROCESSED_PATH}`\n"
            f"- `{OTT_RAW_PATH}`\n\n"
            f"Please verify the file exists in your project directory."
        )
        st.stop()

    # Ensure required target column exists
    if "High_Rated" not in df.columns:
        df["High_Rated"] = (df["Rotten Tomatoes"] >= 70).astype(int)

    # Standardize types and fill missing categorical values
    df["Age"] = df["Age"].fillna("Not Rated").astype(str).str.strip()
    df["Type"] = pd.to_numeric(df["Type"], errors="coerce").fillna(0).astype(int)
    for p in ["Netflix", "Hulu", "Prime Video", "Disney+"]:
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors="coerce").fillna(0).astype(int)

    return df


@st.cache_data(show_spinner=False)
def load_indian_data():
    """Loads and sanitizes the Indian Cinema dataset."""
    if not INDIAN_RAW_PATH.exists():
        st.error(
            f"❌ Required Indian Movies dataset missing!\n\n"
            f"Looked for: `{INDIAN_RAW_PATH}`\n\n"
            f"Please ensure `indian movies.csv` is present under `data/raw/`."
        )
        st.stop()

    df = pd.read_csv(INDIAN_RAW_PATH)

    # Rating to numeric (handles '-' and text)
    df["Rating(10)"] = pd.to_numeric(
        df["Rating(10)"].replace("-", np.nan),
        errors="coerce"
    )

    # Votes: strip commas and dashes
    df["Votes"] = (
        df["Votes"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("-", np.nan)
    )
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

    # Clean release Year: extract 4-digit numeric pattern (handles '2018 Video' etc.)
    df["Year"] = (
        df["Year"]
        .astype(str)
        .str.extract(r"(\d{4})")[0]
    )
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Standardize Language: Title case, strip whitespace, handle missing
    df["Language"] = (
        df["Language"]
        .astype(str)
        .str.strip()
        .str.capitalize()
        .replace({"-": "Unknown", "Nan": "Unknown", "": "Unknown"})
    )

    # Standardize Genre: handle missing
    df["Genre"] = (
        df["Genre"]
        .astype(str)
        .str.strip()
        .replace({"-": "Unknown", "Nan": "Unknown", "": "Unknown"})
    )

    return df


# Load data
with st.spinner("Loading datasets..."):
    ott_df = load_ott_data()
    indian_df = load_indian_data()


# ==============================================================================
# 3. MACHINE LEARNING MODEL PIPELINE
# ==============================================================================

@st.cache_resource(show_spinner=False)
def build_ml_pipeline(df):
    """Trains and caches the Random Forest rating classification pipeline."""
    features = ["Year", "Age", "Netflix", "Hulu", "Prime Video", "Disney+", "Type"]
    X = df[features].copy()
    y = df["High_Rated"].copy()

    categorical_cols = ["Age", "Type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return pipeline, accuracy, X_train.columns.tolist()

ml_pipeline, model_acc, feature_names = build_ml_pipeline(ott_df)


# ==============================================================================
# 4. SIDEBAR NAVIGATION & GLOBAL FILTERS
# ==============================================================================

st.sidebar.markdown("## 🎬 OTT Analytics Hub")
st.sidebar.caption("Data Science & Cinema Intelligence Dashboard")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Content Analytics",
        "Indian Movies 🇮🇳",
        "Tamil Movies 🎬",
        "ML Prediction 🤖",
        "About"
    ],
    index=0
)

st.sidebar.markdown("---")

# Platform filter for Dashboard and Content Analytics
selected_platform = "All"
if page in ["Dashboard", "Content Analytics"]:
    st.sidebar.markdown("### 🎯 Platform Filter")
    selected_platform = st.sidebar.selectbox(
        "Filter OTT Platform",
        ["All", "Netflix", "Hulu", "Prime Video", "Disney+"],
        index=0
    )
    st.sidebar.caption("Applies to OTT catalog analyses.")

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Academic Project**\n\n"
    "Built with Python, Streamlit, Pandas, Matplotlib, Seaborn, and Scikit-Learn."
)

# Apply OTT platform filter
if selected_platform == "All":
    filtered_ott_df = ott_df.copy()
else:
    filtered_ott_df = ott_df[ott_df[selected_platform] == 1].copy()


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================

if page == "Dashboard":
    st.title("🎬 OTT Performance Dashboard")
    st.markdown(
        f"Real-time overview of global streaming catalog data. "
        f"Currently viewing: **{selected_platform}** catalog."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # 1. KPIs
    total_titles = len(filtered_ott_df)
    avg_rating = filtered_ott_df["Rotten Tomatoes"].mean()
    latest_year = int(filtered_ott_df["Year"].max()) if not filtered_ott_df.empty else 0
    movies_count = int((filtered_ott_df["Type"] == 0).sum())
    shows_count = int((filtered_ott_df["Type"] == 1).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Titles", f"{total_titles:,}", f"{selected_platform} Scope")
    with c2:
        render_metric_card("Avg Rotten Tomatoes", f"{avg_rating:.1f}%", "Out of 100")
    with c3:
        render_metric_card("Latest Release Year", f"{latest_year}", "Most recent")
    with c4:
        render_metric_card("Movies / TV Shows", f"{movies_count} / {shows_count}", "Type ratio")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Visual Charts Row
    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        st.subheader("📊 Platform-Wise Content Volume")
        platforms = ["Netflix", "Hulu", "Prime Video", "Disney+"]
        counts = [ott_df[p].sum() for p in platforms]
        platform_colors = ["#E50914", "#1CE783", "#00A8E1", "#113CCF"]

        set_dark_plot_style()
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bars = ax.bar(platforms, counts, color=platform_colors, width=0.55, edgecolor="#2D3748")
        ax.set_ylabel("Number of Titles", fontsize=11, labelpad=8)
        ax.set_title("Total Content Count Across Major OTT Providers", fontsize=12, pad=12, weight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height):,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, weight='bold', color="#FAFAFA")
        st.pyplot(fig, clear_figure=True)

    with col_chart2:
        st.subheader("⭐ High vs Standard Content")
        high_rated_count = (filtered_ott_df["High_Rated"] == 1).sum()
        lower_rated_count = (filtered_ott_df["High_Rated"] == 0).sum()

        set_dark_plot_style()
        fig_pie, ax_pie = plt.subplots(figsize=(5, 4.2))
        wedges, texts, autotexts = ax_pie.pie(
            [high_rated_count, lower_rated_count],
            labels=["High Rated (≥70%)", "Standard (<70%)"],
            autopct="%1.1f%%",
            startangle=140,
            colors=["#00D26A", "#E50914"],
            explode=(0.06, 0),
            textprops={'color': '#FAFAFA', 'weight': 'bold', 'fontsize': 10}
        )
        for at in autotexts:
            at.set_color('#0E1117')
            at.set_weight('bold')
        ax_pie.set_title(f"Rating Split ({selected_platform})", fontsize=12, pad=12, weight='bold')
        st.pyplot(fig_pie, clear_figure=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Content Table Explorer
    st.subheader("🎞️ OTT Catalog Explorer")
    tab_top, tab_recent = st.tabs(["⭐ Top Rated Titles", "📅 Most Recent Releases"])

    display_cols = ["Title", "Year", "Age", "Rotten Tomatoes", "Netflix", "Hulu", "Prime Video", "Disney+"]

    with tab_top:
        top_df = (
            filtered_ott_df[display_cols]
            .sort_values(by="Rotten Tomatoes", ascending=False)
            .head(15)
        )
        st.dataframe(top_df, width="stretch", hide_index=True)

    with tab_recent:
        recent_df = (
            filtered_ott_df[display_cols]
            .sort_values(by=["Year", "Rotten Tomatoes"], ascending=[False, False])
            .head(15)
        )
        st.dataframe(recent_df, width="stretch", hide_index=True)


# ==============================================================================
# PAGE 2: CONTENT ANALYTICS
# ==============================================================================

elif page == "Content Analytics":
    st.title("📈 OTT Content Analytics")
    st.markdown(
        f"Deep dive into rating distributions, release patterns, and demographic tags. "
        f"Platform scope: **{selected_platform}**"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("⭐ Rotten Tomatoes Rating Distribution")
        set_dark_plot_style()
        fig_dist, ax_dist = plt.subplots(figsize=(7, 4.2))
        sns.histplot(
            filtered_ott_df["Rotten Tomatoes"],
            kde=True,
            color="#38BDF8",
            bins=20,
            ax=ax_dist,
            edgecolor="#1E293B"
        )
        median_val = filtered_ott_df["Rotten Tomatoes"].median()
        ax_dist.axvline(median_val, color="#E50914", linestyle="--", linewidth=2, label=f"Median: {median_val:.1f}")
        ax_dist.axvline(70, color="#00D26A", linestyle=":", linewidth=2, label="High-Rated Mark (70%)")
        ax_dist.set_xlabel("Rotten Tomatoes Rating (%)", fontsize=10, labelpad=8)
        ax_dist.set_ylabel("Title Count", fontsize=10, labelpad=8)
        ax_dist.legend(loc="upper left")
        ax_dist.grid(axis='y', linestyle='--', alpha=0.3)
        st.pyplot(fig_dist, clear_figure=True)

    with c2:
        st.subheader("📅 Content Production by Release Year")
        year_counts = (
            filtered_ott_df[filtered_ott_df["Year"] >= 1970]["Year"]
            .value_counts()
            .sort_index()
        )
        set_dark_plot_style()
        fig_year, ax_year = plt.subplots(figsize=(7, 4.2))
        ax_year.plot(year_counts.index, year_counts.values, color="#F59E0B", linewidth=2.5, marker="o", markersize=3)
        ax_year.fill_between(year_counts.index, year_counts.values, color="#F59E0B", alpha=0.2)
        ax_year.set_xlabel("Release Year", fontsize=10, labelpad=8)
        ax_year.set_ylabel("Number of Releases", fontsize=10, labelpad=8)
        ax_year.grid(linestyle='--', alpha=0.3)
        st.pyplot(fig_year, clear_figure=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("🎯 Content Age Rating Breakdown")
        age_counts = filtered_ott_df["Age"].value_counts()
        set_dark_plot_style()
        fig_age, ax_age = plt.subplots(figsize=(7, 4.2))
        sns.barplot(
            x=age_counts.index,
            y=age_counts.values,
            palette="Blues_r",
            ax=ax_age,
            edgecolor="#2D3748"
        )
        ax_age.set_xlabel("Age Classification", fontsize=10, labelpad=8)
        ax_age.set_ylabel("Title Count", fontsize=10, labelpad=8)
        ax_age.grid(axis='y', linestyle='--', alpha=0.3)
        for idx, val in enumerate(age_counts.values):
            ax_age.annotate(f'{val:,}', xy=(idx, val), xytext=(0, 4), textcoords="offset points",
                            ha='center', fontsize=9, color="#FAFAFA", weight="bold")
        st.pyplot(fig_age, clear_figure=True)

    with c4:
        st.subheader("🏆 Platform Quality & Volume Comparison")
        platforms = ["Netflix", "Hulu", "Prime Video", "Disney+"]
        stats = []
        for p in platforms:
            p_df = ott_df[ott_df[p] == 1]
            stats.append({
                "Platform": p,
                "Catalog Size": len(p_df),
                "Average Rating": p_df["Rotten Tomatoes"].mean(),
                "High Rated Share (%)": (p_df["High_Rated"] == 1).mean() * 100
            })
        platform_stats_df = pd.DataFrame(stats)

        set_dark_plot_style()
        fig_comp, ax_comp1 = plt.subplots(figsize=(7, 4.2))
        ax_comp2 = ax_comp1.twinx()

        x = np.arange(len(platforms))
        w = 0.35
        rects1 = ax_comp1.bar(x - w/2, platform_stats_df["Catalog Size"], w, label="Catalog Size", color="#38BDF8", edgecolor="#2D3748")
        rects2 = ax_comp2.bar(x + w/2, platform_stats_df["Average Rating"], w, label="Avg Rating (%)", color="#E50914", edgecolor="#2D3748")

        ax_comp1.set_xticks(x)
        ax_comp1.set_xticklabels(platforms)
        ax_comp1.set_ylabel("Total Titles", color="#38BDF8", fontsize=10)
        ax_comp2.set_ylabel("Average Rotten Tomatoes", color="#E50914", fontsize=10)
        ax_comp1.grid(axis='y', linestyle='--', alpha=0.3)
        st.pyplot(fig_comp, clear_figure=True)


# ==============================================================================
# PAGE 3: INDIAN MOVIES 🇮🇳
# ==============================================================================

elif page == "Indian Movies 🇮🇳":
    st.title("🇮🇳 Indian Cinema Analytics")
    st.markdown(
        "Historical and linguistic analysis of Indian cinema spanning 50,000+ films (1913 - 2024)."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Indian Movie Filters
    lang_list = sorted([l for l in indian_df["Language"].unique() if l != "Unknown"])
    selected_language = st.selectbox(
        "🌐 Filter by Language",
        ["All"] + lang_list,
        index=0
    )

    if selected_language == "All":
        filtered_indian_df = indian_df.copy()
    else:
        filtered_indian_df = indian_df[indian_df["Language"] == selected_language].copy()

    # 2. Metric Cards
    total_ind_movies = len(filtered_indian_df)
    avg_ind_rating = filtered_indian_df["Rating(10)"].mean()
    total_ind_votes = filtered_indian_df["Votes"].sum()
    num_languages = filtered_indian_df["Language"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Movies", f"{total_ind_movies:,}", f"{selected_language} Scope")
    with c2:
        val_rating = f"{avg_ind_rating:.2f} / 10" if pd.notna(avg_ind_rating) else "N/A"
        render_metric_card("Average Rating", val_rating, "User Score")
    with c3:
        val_votes = f"{int(total_ind_votes):,}" if pd.notna(total_ind_votes) else "0"
        render_metric_card("Total Votes", val_votes, "Audience Engagement")
    with c4:
        render_metric_card("Languages", f"{num_languages}", "Linguistic Diversity")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Charts
    col_ind1, col_ind2 = st.columns(2)

    with col_ind1:
        st.subheader("🌐 Top Languages by Production Volume")
        top_languages = (
            indian_df["Language"]
            .value_counts()
            .head(12)
        )
        set_dark_plot_style()
        fig_lang, ax_lang = plt.subplots(figsize=(7, 4.5))
        sns.barplot(
            x=top_languages.values,
            y=top_languages.index,
            palette="mako",
            ax=ax_lang,
            edgecolor="#2D3748"
        )
        ax_lang.set_xlabel("Number of Movies", fontsize=10, labelpad=8)
        ax_lang.grid(axis='x', linestyle='--', alpha=0.3)
        for idx, val in enumerate(top_languages.values):
            ax_lang.annotate(f'{val:,}', xy=(val, idx), xytext=(5, 0), textcoords="offset points",
                            va='center', fontsize=9, color="#FAFAFA")
        st.pyplot(fig_lang, clear_figure=True)

    with col_ind2:
        st.subheader("🎭 Popular Movie Genres")
        # Extract and clean genres (handling combinations)
        genres_clean = (
            filtered_indian_df["Genre"]
            .replace("Unknown", np.nan)
            .dropna()
            .str.split(",")
            .explode()
            .str.strip()
        )
        top_genres = genres_clean.value_counts().head(12)

        set_dark_plot_style()
        fig_genre, ax_genre = plt.subplots(figsize=(7, 4.5))
        sns.barplot(
            x=top_genres.values,
            y=top_genres.index,
            palette="viridis",
            ax=ax_genre,
            edgecolor="#2D3748"
        )
        ax_genre.set_xlabel("Frequency in Films", fontsize=10, labelpad=8)
        ax_genre.grid(axis='x', linestyle='--', alpha=0.3)
        for idx, val in enumerate(top_genres.values):
            ax_genre.annotate(f'{val:,}', xy=(val, idx), xytext=(5, 0), textcoords="offset points",
                            va='center', fontsize=9, color="#FAFAFA")
        st.pyplot(fig_genre, clear_figure=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Release Timeline
    st.subheader("📅 Indian Movie Releases Over Time")
    yearly_ind = (
        filtered_indian_df[filtered_indian_df["Year"] >= 1950]["Year"]
        .value_counts()
        .sort_index()
    )
    set_dark_plot_style()
    fig_ind_time, ax_ind_time = plt.subplots(figsize=(12, 3.8))
    ax_ind_time.plot(yearly_ind.index, yearly_ind.values, color="#10B981", linewidth=2.5)
    ax_ind_time.fill_between(yearly_ind.index, yearly_ind.values, color="#10B981", alpha=0.15)
    ax_ind_time.set_xlabel("Release Year", fontsize=10, labelpad=8)
    ax_ind_time.set_ylabel("Films Produced", fontsize=10, labelpad=8)
    ax_ind_time.grid(linestyle='--', alpha=0.3)
    st.pyplot(fig_ind_time, clear_figure=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Top Rated Indian Movies Table
    st.subheader("⭐ Top Rated Indian Movies Leaderboard")
    c_flt1, c_flt2 = st.columns([2, 2])
    with c_flt1:
        min_votes_ind = st.slider("Minimum Votes Filter (Avoid Low-Vote Outliers)", 0, 5000, 250, step=50)

    top_ind_df = (
        filtered_indian_df[filtered_indian_df["Votes"] >= min_votes_ind]
        .dropna(subset=["Rating(10)"])
        .sort_values(by=["Rating(10)", "Votes"], ascending=[False, False])
        .head(20)[["Movie Name", "Year", "Rating(10)", "Votes", "Genre", "Language"]]
    )
    if top_ind_df.empty:
        st.warning("No titles found with the chosen filter criteria. Try lowering the minimum votes slider.")
    else:
        st.dataframe(top_ind_df, width="stretch", hide_index=True)


# ==============================================================================
# PAGE 4: TAMIL MOVIES 🎬
# ==============================================================================

elif page == "Tamil Movies 🎬":
    st.title("🎬 Tamil Cinema (Kollywood) Analytics")
    st.markdown(
        "Dedicated exploration of Tamil cinema, highlighting 6,000+ films, legendary releases, and genre trends."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    tamil_df = indian_df[indian_df["Language"].str.lower() == "tamil"].copy()

    # 1. Tamil Cinema KPIs
    tamil_count = len(tamil_df)
    tamil_avg_rating = tamil_df["Rating(10)"].mean()
    tamil_total_votes = tamil_df["Votes"].sum()
    tamil_rated_films = tamil_df["Rating(10)"].dropna().count()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Tamil Films", f"{tamil_count:,}", "Kollywood Archive")
    with c2:
        val_tm_rating = f"{tamil_avg_rating:.2f} / 10" if pd.notna(tamil_avg_rating) else "N/A"
        render_metric_card("Average Rating", val_tm_rating, "User Score")
    with c3:
        val_tm_votes = f"{int(tamil_total_votes):,}" if pd.notna(tamil_total_votes) else "0"
        render_metric_card("Total Votes Cast", val_tm_votes, "Audience Engagement")
    with c4:
        render_metric_card("Rated Titles", f"{tamil_rated_films:,}", f"{(tamil_rated_films/tamil_count)*100:.1f}% catalog")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Visual Trends
    col_tm1, col_tm2 = st.columns(2)

    with col_tm1:
        st.subheader("📅 Tamil Movies Produced by Year")
        tm_year_counts = (
            tamil_df[tamil_df["Year"] >= 1950]["Year"]
            .value_counts()
            .sort_index()
        )
        set_dark_plot_style()
        fig_tm_year, ax_tm_year = plt.subplots(figsize=(7, 4.2))
        ax_tm_year.plot(tm_year_counts.index, tm_year_counts.values, color="#E50914", linewidth=2.5, marker="o", markersize=3)
        ax_tm_year.fill_between(tm_year_counts.index, tm_year_counts.values, color="#E50914", alpha=0.18)
        ax_tm_year.set_xlabel("Release Year", fontsize=10, labelpad=8)
        ax_tm_year.set_ylabel("Number of Movies", fontsize=10, labelpad=8)
        ax_tm_year.grid(linestyle='--', alpha=0.3)
        st.pyplot(fig_tm_year, clear_figure=True)

    with col_tm2:
        st.subheader("🎭 Popular Tamil Genres")
        tm_genres = (
            tamil_df["Genre"]
            .replace("Unknown", np.nan)
            .dropna()
            .str.split(",")
            .explode()
            .str.strip()
        )
        top_tm_genres = tm_genres.value_counts().head(10)
        set_dark_plot_style()
        fig_tm_genre, ax_tm_genre = plt.subplots(figsize=(7, 4.2))
        sns.barplot(
            x=top_tm_genres.values,
            y=top_tm_genres.index,
            palette="rocket",
            ax=ax_tm_genre,
            edgecolor="#2D3748"
        )
        ax_tm_genre.set_xlabel("Number of Titles", fontsize=10, labelpad=8)
        ax_tm_genre.grid(axis='x', linestyle='--', alpha=0.3)
        for idx, val in enumerate(top_tm_genres.values):
            ax_tm_genre.annotate(f'{val:,}', xy=(val, idx), xytext=(5, 0), textcoords="offset points",
                                 va='center', fontsize=9, color="#FAFAFA")
        st.pyplot(fig_tm_genre, clear_figure=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Interactive Search & Filters
    st.subheader("🔍 Kollywood Movie Search & Explorer")
    c_s1, c_s2 = st.columns([2, 1])
    with c_s1:
        search_query = st.text_input("🔎 Search by Tamil Movie Title", placeholder="e.g., Nayagan, Baasha, Vikram, Anbe Sivam...")
    with c_s2:
        min_tm_votes = st.slider("Min Votes (Quality Gate)", 0, 5000, 100, step=50)

    # Apply search filter
    filtered_tamil = tamil_df[tamil_df["Votes"] >= min_tm_votes].copy()
    if search_query.strip():
        filtered_tamil = filtered_tamil[
            filtered_tamil["Movie Name"].str.contains(search_query.strip(), case=False, na=False)
        ]

    st.markdown("#### ⭐ Top Rated Tamil Movies")
    top_tamil_df = (
        filtered_tamil
        .dropna(subset=["Rating(10)"])
        .sort_values(by=["Rating(10)", "Votes"], ascending=[False, False])
        .head(25)[["Movie Name", "Year", "Rating(10)", "Votes", "Genre"]]
    )

    if top_tamil_df.empty:
        st.info("No Tamil movies found matching your search term and vote filter.")
    else:
        st.dataframe(top_tamil_df, width="stretch", hide_index=True)


# ==============================================================================
# PAGE 5: ML PREDICTION 🤖
# ==============================================================================

elif page == "ML Prediction 🤖":
    st.title("🤖 ML Rating Classification")
    st.markdown(
        "Predict whether an OTT title is likely to be **High Rated** (Rotten Tomatoes ≥ 70%) "
        "using a Scikit-Learn **Random Forest Classifier** trained on streaming platform metadata."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Model Performance Architecture Card
    with st.expander("ℹ️ Model Architecture & Evaluation Metrics", expanded=True):
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("**Algorithm:** `RandomForestClassifier`")
            st.caption("100 Decision Trees with balanced class weights.")
        with m2:
            st.markdown(f"**Test Accuracy:** `{model_acc:.2%}`")
            st.caption("Evaluated on stratified 20% test holdout.")
        with m3:
            st.markdown("**Target Threshold:** `Rotten Tomatoes ≥ 70`")
            st.caption("Binary classification: High Rated (1) vs Standard (0).")

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("🎛️ Input Movie Features")

    # Form Controls
    col_inp1, col_inp2 = st.columns(2)

    with col_inp1:
        input_year = st.slider("📅 Release Year", min_value=1920, max_value=2030, value=2024, step=1)
        input_age = st.selectbox(
            "🔞 Age Rating Classification",
            ["Not Rated", "all", "7+", "13+", "16+", "18+"],
            index=3
        )
        input_type = st.radio(
            "🎞️ Content Format",
            options=[0, 1],
            format_func=lambda x: "Movie (Feature Film)" if x == 0 else "TV Show (Series)",
            horizontal=True
        )

    with col_inp2:
        st.markdown("**🌐 Streaming Platform Availability**")
        st.caption("Select all platforms where this title will be distributed:")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            input_netflix = 1 if st.checkbox("Netflix", value=True) else 0
            input_hulu = 1 if st.checkbox("Hulu", value=False) else 0
        with c_p2:
            input_prime = 1 if st.checkbox("Prime Video", value=False) else 0
            input_disney = 1 if st.checkbox("Disney+", value=False) else 0

    st.markdown("<br>", unsafe_allow_html=True)

    # Predict button
    if st.button("🔮 Run Machine Learning Prediction", type="primary", use_container_width=True):
        input_sample = pd.DataFrame([{
            "Year": input_year,
            "Age": input_age,
            "Netflix": input_netflix,
            "Hulu": input_hulu,
            "Prime Video": input_prime,
            "Disney+": input_disney,
            "Type": input_type
        }])

        prediction = ml_pipeline.predict(input_sample)[0]
        probabilities = ml_pipeline.predict_proba(input_sample)[0]
        high_prob = probabilities[1] * 100
        low_prob = probabilities[0] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="pred-card-high">
                <h2 style="color: #00D26A; margin: 0;">🎉 High Rated Title Predicted!</h2>
                <h4 style="color: #FAFAFA; margin-top: 8px;">Confidence Score: {high_prob:.1f}%</h4>
                <p style="color: #94A3B8; max-width: 600px; margin: 8px auto 0 auto;">
                    The model predicts that this content exhibits characteristics of titles receiving a 
                    <b>Rotten Tomatoes score of 70% or higher</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-card-low">
                <h2 style="color: #EF4444; margin: 0;">📉 Standard / Lower Rated Predicted</h2>
                <h4 style="color: #FAFAFA; margin-top: 8px;">Confidence Score: {low_prob:.1f}%</h4>
                <p style="color: #94A3B8; max-width: 600px; margin: 8px auto 0 auto;">
                    The model predicts that this title is more likely to score <b>below 70%</b> on Rotten Tomatoes.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            "⚠️ **Academic Disclaimer**: This prediction is generated by a Scikit-Learn Random Forest Classifier "
            "trained on streaming catalog attributes. It reflects statistical probabilities based on historical "
            "platform distribution patterns and does not guarantee audience or critical reception."
        )


# ==============================================================================
# PAGE 6: ABOUT
# ==============================================================================

elif page == "About":
    st.title("ℹ️ About OTT Performance Analytics")
    st.markdown("### End-to-End Data Science Project & Cinema Intelligence")
    st.markdown("<br>", unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([3, 2])

    with col_a1:
        st.markdown("""
        #### 🎯 Project Overview
        **OTT Performance Analytics** is a comprehensive Data Science system designed to inspect, analyze, and predict 
        content trends across major Over-The-Top (OTT) streaming providers, combined with in-depth analytics 
        of Indian and Tamil cinema archives.

        #### 🔬 Data Science Workflow
        1. **Data Ingestion**: Multi-source collection covering streaming platforms (`Netflix`, `Hulu`, `Prime Video`, `Disney+`) and 50,000+ Indian cinema titles.
        2. **Data Sanitization**: Handling missing ratings, cleaning Rotten Tomatoes string percentages, sanitizing vote counts with commas, and standardizing linguistic taxonomies.
        3. **Exploratory Data Analysis (EDA)**: Rating distributions, production trends, demographic classifications, and regional industry comparisons.
        4. **Machine Learning**: Random Forest classification with ColumnTransformer pipelines for rating tier prediction.
        5. **Interactive UI**: Fully responsive Streamlit web dashboard styled with an immersive dark theme suitable for college presentations.
        """)

    with col_a2:
        st.markdown("""
        #### 🛠️ Tech Stack
        - **Language:** Python 3.10+
        - **Dashboard Framework:** Streamlit
        - **Data Manipulation:** Pandas, NumPy
        - **Data Visualization:** Matplotlib, Seaborn
        - **Machine Learning:** Scikit-Learn (RandomForest, ColumnTransformer, Pipeline)
        - **Deployment Target:** Streamlit Community Cloud

        #### 📁 Verified Project Datasets
        - `data/processed/ott_cleaned.csv` (9,515 titles)
        - `data/raw/MoviesOnStreamingPlatforms.csv` (9,515 titles)
        - `data/raw/indian movies.csv` (50,602 titles)
        """)

    st.markdown("---")
    st.success("🎓 Suitable for Academic Project Demonstrations & Industry Portfolios.")
