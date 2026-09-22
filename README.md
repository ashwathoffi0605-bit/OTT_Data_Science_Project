# 🎬 OTT Performance Analytics & Cinema Intelligence Hub

A complete, presentation-ready Data Science web application and interactive dark-themed dashboard built with **Python**, **Streamlit**, **Pandas**, **Matplotlib**, **Seaborn**, and **Scikit-Learn**.

Analyzes streaming catalog trends across major OTT platforms (**Netflix**, **Hulu**, **Prime Video**, **Disney+**), explores historical Indian and Tamil cinema archives (**50,000+ films**), and classifies content rating tiers using an integrated Machine Learning pipeline.

---

## 📌 Features & Navigation

The dashboard offers a modern, responsive dark UI accessible via the sidebar:

1. **📊 Dashboard**
   - Summary KPIs: Total Titles, Average Rotten Tomatoes score, Latest Release Year, and Movie vs. TV Show split.
   - Global OTT platform filter (All, Netflix, Hulu, Prime Video, Disney+).
   - Platform content volume comparison and rating tier breakdown.
   - Interactive content tables for Top Rated Titles and Recent Releases.

2. **📈 Content Analytics**
   - Rotten Tomatoes score distribution (histogram + KDE with median annotations).
   - Content production volume by release year across decades.
   - Age rating classification breakdown (`18+`, `13+`, `7+`, `all`, `16+`, `Not Rated`).
   - Platform quality vs. catalog size comparison.

3. **🇮🇳 Indian Movies**
   - Deep-dive into 50,000+ Indian movies spanning 1913–2024.
   - Language selector across 20+ Indian languages (Hindi, Malayalam, Tamil, Telugu, Bengali, Kannada, Marathi, Punjabi, etc.).
   - Top languages by production volume and most popular genres.
   - Historical release timeline and top-rated leaderboard with minimum-vote thresholds.

4. **🎬 Tamil Movies**
   - Dedicated Kollywood analytics covering 6,000+ Tamil films.
   - Production trends over the decades and top Tamil genres.
   - Real-time movie title search bar (e.g., *Nayagan*, *Baasha*, *Vikram*, *Anbe Sivam*).
   - Top-rated Tamil movies leaderboard with customizable vote filters.

5. **🤖 ML Prediction**
   - Scikit-Learn `RandomForestClassifier` inside a `Pipeline` with `ColumnTransformer` and `OneHotEncoder`.
   - **Target**: `High_Rated` (1 if Rotten Tomatoes ≥ 70%, 0 otherwise).
   - **Features**: Release Year, Age Classification, Netflix, Hulu, Prime Video, Disney+, Content Type.
   - Interactive form controls for simulated or upcoming movie releases.
   - Instant prediction with confidence score and statistical disclaimer.

6. **ℹ️ About**
   - Full data science workflow documentation, architecture overview, dataset schema, and presentation notes.

---

## 🗂️ Project Structure

```text
OTT_Data_Science_Project/
├── .streamlit/
│   └── config.toml               # Modern dark theme configuration
├── data/
│   ├── raw/
│   │   ├── MoviesOnStreamingPlatforms.csv  # Raw OTT streaming dataset (9,515 titles)
│   │   └── indian movies.csv               # Comprehensive Indian cinema dataset (50,602 titles)
│   └── processed/
│       └── ott_cleaned.csv                 # Cleaned OTT dataset with Rotten Tomatoes scores
├── dashboard/
│   └── app.py                    # Streamlit dashboard application (subfolder copy)
├── notebooks/                    # Jupyter notebooks for exploratory research
├── src/
│   ├── data_check.py             # Data sanity and cleaning script
│   ├── eda.py                    # Exploratory Data Analysis helper script
│   └── ml_model.py               # Standalone ML training & evaluation script
├── app.py                        # Root Streamlit entry point (for Cloud deployment)
├── requirements.txt              # Production Python dependencies
└── README.md                     # Project documentation & presentation guide
```

---

## 🛠️ Technology Stack

- **Frontend & UI:** [Streamlit](https://streamlit.io/) (Dark Mode, Custom CSS, Glassmorphic Metric Cards)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Data Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (`RandomForestClassifier`, `ColumnTransformer`, `Pipeline`, `OneHotEncoder`)

---

## 🚀 Running Locally

### 1. Clone or Open the Repository
```bash
cd OTT_Data_Science_Project
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
The application will open automatically in your default browser at `http://localhost:8501`.

*(You can also run `streamlit run dashboard/app.py` directly; robust path resolution ensures seamless operation from any directory.)*

---

## ☁️ Streamlit Community Cloud Deployment

This repository is pre-configured for one-click deployment on **Streamlit Community Cloud**:

1. Push this repository to **GitHub**.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**.
4. Select your repository, branch (`main`), and set the main file path to:
   ```text
   app.py
   ```
5. Click **"Deploy"**! Streamlit will automatically install dependencies from `requirements.txt` and run the dashboard.

---

## 🎓 Academic Presentation Notes

- **Data Robustness:** All file paths use `pathlib.Path` with automatic root detection. Missing values are gracefully coerced and sanitized without application crashes.
- **Model Explainability:** The Random Forest classifier handles high-cardinality categorical features using stratified train-test splits and outputs calibrated class probabilities.
- **Real-World Relevance:** Combines global streaming intelligence with regional linguistic cinema analysis, demonstrating full-stack Data Science capability from raw data ingestion to interactive deployment.
