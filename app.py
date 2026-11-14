
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ExoMatch | Exoplanet Explorer",
    page_icon="🪐",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = PROJECT_ROOT / "data" / "cleaned_planets.csv"
CONFIRMED_FILE = PROJECT_ROOT / "data" / "confirmed_planets.csv"
IMAGE_FILE = PROJECT_ROOT / "assets" / "exoplanet_types.png"


# =========================================================
# 2. DARK BLUE THEME
# =========================================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #101827;
    color: #e2e8f0;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headings */
h1 {
    color: #60a5fa !important;
    font-size: 32px !important;
    font-weight: 700 !important;
}

h2 {
    color: #93c5fd !important;
    font-size: 25px !important;
}

h3 {
    color: #bfdbfe !important;
    font-size: 20px !important;
}

/* Normal text */
p, li, label, .stMarkdown {
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0b1220;
    border-right: 1px solid #26364f;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    color: #60a5fa !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #1e293b !important;
    border: 1px solid #334e78 !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    min-height: 88px !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.15);
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
    font-size: 13px !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-size: 27px !important;
    font-weight: 700 !important;
}

/* Metric delta */
[data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

/* Buttons */
.stButton > button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 8px !important;
    padding: 7px 14px !important;
    font-weight: 600 !important;
    transition: 0.2s;
}

.stButton > button:hover {
    background-color: #1d4ed8 !important;
    border-color: #93c5fd !important;
}

/* Download buttons */
.stDownloadButton > button {
    background-color: #1d4ed8 !important;
    color: white !important;
    border-radius: 8px !important;
    border: 1px solid #60a5fa !important;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stDateInput input {
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #475569 !important;
    border-radius: 7px !important;
}

/* Select boxes and multiselect */
[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
    border-color: #475569 !important;
}

/* Selectbox text */
[data-baseweb="select"] span {
    color: #f1f5f9 !important;
}

/* Checkbox */
[data-testid="stCheckbox"] label {
    color: #e2e8f0 !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

/* Tables */
[data-testid="stTable"] {
    background-color: #1e293b !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #1e293b !important;
    border: 1px solid #334e78 !important;
    border-radius: 10px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #cbd5e1 !important;
}

/* Dividers */
hr {
    border-color: #334155 !important;
}

/* Planet cards */
.planet-card {
    background-color: #1e293b;
    border: 1px solid #334e78;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
}

/* Captions */
.stCaption, small {
    color: #94a3b8 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Links */
a {
    color: #60a5fa !important;
}

</style>
""", unsafe_allow_html=True)


# Dark chart theme
px.defaults.template = "plotly_dark"


# =========================================================
# 3. LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    if DATA_FILE.exists():
        file_path = DATA_FILE

    elif CONFIRMED_FILE.exists():
        file_path = CONFIRMED_FILE

    else:
        return pd.DataFrame()

    data = pd.read_csv(file_path)

    if "pl_name" not in data.columns:
        return pd.DataFrame()

    data = data.drop_duplicates(subset=["pl_name"])

    numeric_columns = [
        "disc_year",
        "pl_rade",
        "pl_bmasse",
        "pl_orbper",
        "pl_eqt",
        "st_teff"
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data.reset_index(drop=True)


df = load_data()


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def get_value(row, column):

    if column not in row.index:
        return "Not available"

    value = row[column]

    if pd.isna(value):
        return "Not available"

    return value


def format_number(value, decimals=2):

    if pd.isna(value):
        return "Not available"

    return f"{value:,.{decimals}f}"


def get_planet_type(radius):

    if pd.isna(radius):
        return "Unknown"

    if radius < 1.25:
        return "Small / Terrestrial"

    elif radius < 2:
        return "Super-Earth"

    elif radius < 4:
        return "Mini-Neptune"

    elif radius < 10:
        return "Neptune-like"

    else:
        return "Gas Giant"


def get_planet_image(planet_name):

    # Official NASA artist-concept images.
    # Exoplanets have not been photographed individually, so these are
    # illustrations used to represent different kinds of worlds.
    images = [
        "https://images-assets.nasa.gov/image/PIA23689/PIA23689~medium.jpg",
        "https://assets.science.nasa.gov/dynamicimage/assets/science/psd/photojournal/pia/pia23/pia23690/PIA23690.jpg",
        "https://assets.science.nasa.gov/dynamicimage/assets/science/psd/photojournal/pia/pia26/pia26601/PIA26601.jpg",
        "https://assets.science.nasa.gov/dynamicimage/assets/science/psd/photojournal/pia/pia21/pia21751/PIA21751.jpg?crop=faces,focalpoint&fit=clip&h=900&w=1200",
        "https://assets.science.nasa.gov/dynamicimage/assets/science/psd/photojournal/pia/pia09/pia09200/PIA09200.jpg?crop=faces,focalpoint&fit=clip&h=900&w=1200"
    ]

    # Give each planet name a stable image instead of repeating
    # the same image for every row position.
    image_index = sum(ord(character) for character in str(planet_name)) % len(images)

    return images[image_index]


def create_download_button(data, filename, label, key):

    csv_data = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        key=key
    )


# =========================================================
# 5. MACHINE LEARNING RECOMMENDATION FUNCTIONS
# =========================================================

FEATURES = [
    "pl_rade",
    "pl_bmasse",
    "pl_orbper",
    "pl_eqt",
    "st_teff"
]


@st.cache_data
def prepare_recommendation_data(data):

    available_features = [
        column for column in FEATURES
        if column in data.columns
    ]

    if len(data) < 2 or not available_features:
        return None

    numeric_data = data[available_features].copy()

    available_features = [
        column for column in available_features
        if numeric_data[column].notna().any()
    ]

    if not available_features:
        return None

    numeric_data = numeric_data[available_features]

    imputer = SimpleImputer(strategy="median")
    imputed_data = imputer.fit_transform(numeric_data)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    number_of_clusters = min(5, len(data))

    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    cluster_labels = model.fit_predict(scaled_data)

    return {
        "features": available_features,
        "scaled_data": scaled_data,
        "clusters": cluster_labels
    }


def recommend_similar_planets(
    selected_planet,
    data,
    model_data
):

    if model_data is None:
        return pd.DataFrame()

    selected_index = data.index[
        data["pl_name"] == selected_planet
    ].tolist()

    if not selected_index:
        return pd.DataFrame()

    selected_index = selected_index[0]

    scaled_data = model_data["scaled_data"]
    clusters = model_data["clusters"]

    selected_cluster = clusters[selected_index]
    selected_vector = scaled_data[selected_index]

    distances = np.linalg.norm(
        scaled_data - selected_vector,
        axis=1
    )

    results = data.copy()

    results["_distance"] = distances
    results["_cluster"] = clusters

    results = results[
        (results["_cluster"] == selected_cluster)
        & (results["pl_name"] != selected_planet)
    ]

    results = results.sort_values("_distance")

    return results.head(5)


# =========================================================
# 6. SESSION STATE
# =========================================================

if "favourites" not in st.session_state:
    st.session_state.favourites = []


# =========================================================
# 7. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🪐 ExoMatch")

st.sidebar.caption(
    "AI-Powered Exoplanet Discovery and Recommendation System"
)

pages = [
    "Home",
    "Explore Planets",
    "Planet Gallery",
    "Planet Details",
    "AI Recommendations",
    "Compare Planets",
    "Planet Statistics",
    "Discovery Insights",
    "About ExoMatch"
]

selected_page = st.sidebar.selectbox(
    "Navigate",
    pages
)

st.sidebar.divider()

st.sidebar.metric(
    "Total Planets",
    len(df)
)

st.sidebar.metric(
    "My Favourites",
    len(st.session_state.favourites)
)

st.sidebar.divider()

st.sidebar.caption(
    "Explore Beyond Our Solar System ✨"
)


# =========================================================
# 8. CHECK DATA
# =========================================================

if df.empty:

    st.error(
        "Planet dataset not found or is empty. "
        "Please check your data folder."
    )

    st.info(
        "Make sure cleaned_planets.csv is inside "
        "the ExoMatch/data folder."
    )

    st.stop()


# =========================================================
# PAGE 1: HOME
# =========================================================

if selected_page == "Home":

    st.title("🪐 Welcome to ExoMatch")

    st.subheader(
        "Explore the fascinating world beyond our Solar System"
    )

    st.write("""
    ExoMatch is an AI-powered exoplanet exploration platform
    that helps users explore confirmed exoplanets, understand
    their properties, compare planets, and discover planets
    with similar characteristics.
    """)

    if IMAGE_FILE.exists():

        st.image(
            str(IMAGE_FILE),
            use_container_width=True
        )

    else:

        st.info(
            "Add exoplanet_types.png to the assets folder "
            "to display the home page illustration."
        )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Planets", len(df))

    if "hostname" in df.columns:
        col2.metric("Host Stars", df["hostname"].nunique())
    else:
        col2.metric("Host Stars", "N/A")

    if "discoverymethod" in df.columns:
        col3.metric(
            "Discovery Methods",
            df["discoverymethod"].nunique()
        )
    else:
        col3.metric("Discovery Methods", "N/A")

    if "disc_year" in df.columns:
        col4.metric(
            "Discovery Years",
            df["disc_year"].nunique()
        )
    else:
        col4.metric("Discovery Years", "N/A")

    st.divider()

    st.subheader("🌍 Recently Listed Planets")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.info(
        "Use the sidebar to explore the different features of ExoMatch."
    )


# =========================================================
# PAGE 2: EXPLORE PLANETS
# =========================================================

elif selected_page == "Explore Planets":

    st.title("🔭 Explore Exoplanets")

    st.write(
        "Search, filter, and explore the confirmed exoplanet dataset."
    )

    filtered_df = df.copy()

    search_text = st.text_input(
        "Search by planet name"
    )

    if search_text:

        filtered_df = filtered_df[
            filtered_df["pl_name"].astype(str).str.contains(
                search_text,
                case=False,
                na=False
            )
        ]

    if "discoverymethod" in filtered_df.columns:

        methods = sorted(
            filtered_df["discoverymethod"]
            .dropna()
            .astype(str)
            .unique()
        )

        selected_methods = st.multiselect(
            "Filter by discovery method",
            methods
        )

        if selected_methods:

            filtered_df = filtered_df[
                filtered_df["discoverymethod"]
                .astype(str)
                .isin(selected_methods)
            ]

    if "disc_year" in filtered_df.columns:

        years = filtered_df["disc_year"].dropna()

        if not years.empty:

            min_year = int(years.min())
            max_year = int(years.max())

            selected_years = st.slider(
                "Select discovery year range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )

            filtered_df = filtered_df[
                filtered_df["disc_year"].between(
                    selected_years[0],
                    selected_years[1]
                )
            ]

    st.subheader("Planet Dataset")

    st.write(f"Showing {len(filtered_df)} planets.")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    create_download_button(
        filtered_df,
        "exomatch_planets.csv",
        "📥 Download Filtered Planet Data",
        "download_explore"
    )

    st.divider()

    st.subheader("📊 Discoveries by Year")

    if "disc_year" in filtered_df.columns:

        yearly_data = (
            filtered_df["disc_year"]
            .dropna()
            .value_counts()
            .sort_index()
            .reset_index()
        )

        yearly_data.columns = [
            "Year",
            "Number of Discoveries"
        ]

        if not yearly_data.empty:

            fig = px.bar(
                yearly_data,
                x="Year",
                y="Number of Discoveries",
                title="Exoplanet Discoveries by Year"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("No discovery-year data available.")


# =========================================================
# PAGE 3: PLANET GALLERY
# =========================================================

elif selected_page == "Planet Gallery":

    st.title("🪐 Planet Gallery")

    st.write(
        "Browse planets and save your favourite discoveries."
    )

    gallery_df = df.copy()

    gallery_search = st.text_input(
        "Search planets",
        key="gallery_search"
    )

    if gallery_search:

        gallery_df = gallery_df[
            gallery_df["pl_name"].astype(str).str.contains(
                gallery_search,
                case=False,
                na=False
            )
        ]

    if "pl_rade" in gallery_df.columns:

        gallery_df["Planet Type"] = (
            gallery_df["pl_rade"].apply(get_planet_type)
        )

        planet_types = sorted(
            gallery_df["Planet Type"].unique()
        )

        selected_type = st.selectbox(
            "Filter by approximate planet type",
            ["All Types"] + planet_types
        )

        if selected_type != "All Types":

            gallery_df = gallery_df[
                gallery_df["Planet Type"] == selected_type
            ]

    show_favourites = st.checkbox(
        "⭐ Show only my favourite planets"
    )

    if show_favourites:

        gallery_df = gallery_df[
            gallery_df["pl_name"].isin(
                st.session_state.favourites
            )
        ]

    st.write(f"Showing {len(gallery_df)} planets.")

    if st.button("🎲 Discover a Random Planet"):

        random_planet = df.sample(1).iloc[0]

        st.success(
            f"Your random discovery is: {random_planet['pl_name']}"
        )

    st.divider()

    for start in range(0, len(gallery_df), 3):

        columns = st.columns(3)

        for position, column in enumerate(columns):

            index = start + position

            if index >= len(gallery_df):
                break

            planet = gallery_df.iloc[index]
            planet_name = planet["pl_name"]

            with column:

                st.markdown(
                    '<div class="planet-card">',
                    unsafe_allow_html=True
                )

                st.image(
                    get_planet_image(planet_name),
                    use_container_width=True
                )

                st.subheader(planet_name)

                if "pl_rade" in planet.index:

                    st.write(
                        "Radius:",
                        format_number(planet["pl_rade"]),
                        "Earth radii"
                    )

                if "discoverymethod" in planet.index:

                    st.write(
                        "Method:",
                        get_value(planet, "discoverymethod")
                    )

                st.write(
                    "Type:",
                    get_planet_type(
                        planet.get("pl_rade", np.nan)
                    )
                )

                is_favourite = (
                    planet_name in st.session_state.favourites
                )

                if is_favourite:

                    if st.button(
                        "💔 Remove Favourite",
                        key=f"remove_{planet_name}"
                    ):

                        st.session_state.favourites.remove(
                            planet_name
                        )

                        st.rerun()

                else:

                    if st.button(
                        "⭐ Add to Favourites",
                        key=f"add_{planet_name}"
                    ):

                        st.session_state.favourites.append(
                            planet_name
                        )

                        st.rerun()

                with st.expander("View Details"):

                    st.write(
                        "Host Star:",
                        get_value(planet, "hostname")
                    )

                    st.write(
                        "Discovery Year:",
                        get_value(planet, "disc_year")
                    )

                    st.write(
                        "Mass:",
                        format_number(
                            planet.get("pl_bmasse", np.nan)
                        ),
                        "Earth masses"
                    )

                    st.write(
                        "Orbital Period:",
                        format_number(
                            planet.get("pl_orbper", np.nan)
                        ),
                        "days"
                    )

                    st.write(
                        "Equilibrium Temperature:",
                        format_number(
                            planet.get("pl_eqt", np.nan)
                        ),
                        "K"
                    )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

    st.divider()

    st.subheader("⭐ My Favourite Planets")

    st.write(
        f"Total favourites: {len(st.session_state.favourites)}"
    )

    if st.session_state.favourites:

        for favourite in st.session_state.favourites:
            st.write(f"⭐ {favourite}")

        favourite_df = df[
            df["pl_name"].isin(
                st.session_state.favourites
            )
        ]

        create_download_button(
            favourite_df,
            "exomatch_favourites.csv",
            "📥 Download My Favourite Planets",
            "download_favourites"
        )

        if st.button("🗑️ Clear All Favourites"):

            st.session_state.favourites = []

            st.rerun()

    else:

        st.info(
            "You haven't added any favourite planets yet."
        )

    st.caption(
        "Gallery images are illustrative, not actual photographs "
        "of individual exoplanets. Planet types are approximate "
        "classifications based on radius."
    )


# =========================================================
# PAGE 4: PLANET DETAILS
# =========================================================

elif selected_page == "Planet Details":

    st.title("🔍 Planet Details")

    planet_names = sorted(
        df["pl_name"].dropna().astype(str).unique()
    )

    selected_planet = st.selectbox(
        "Select an exoplanet",
        planet_names
    )

    planet = df[
        df["pl_name"] == selected_planet
    ].iloc[0]

    st.subheader(selected_planet)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Planet Radius",
        format_number(planet.get("pl_rade", np.nan)),
        "Earth radii"
    )

    col2.metric(
        "Planet Mass",
        format_number(planet.get("pl_bmasse", np.nan)),
        "Earth masses"
    )

    col3.metric(
        "Orbital Period",
        format_number(planet.get("pl_orbper", np.nan)),
        "Days"
    )

    col4, col5 = st.columns(2)

    col4.metric(
        "Equilibrium Temperature",
        format_number(planet.get("pl_eqt", np.nan)),
        "Kelvin"
    )

    col5.metric(
        "Host Star Temperature",
        format_number(planet.get("st_teff", np.nan)),
        "Kelvin"
    )

    st.divider()

    st.subheader("Planet Information")

    details = {
        "Planet Name": get_value(planet, "pl_name"),
        "Host Star": get_value(planet, "hostname"),
        "Discovery Method": get_value(planet, "discoverymethod"),
        "Discovery Year": get_value(planet, "disc_year"),
        "Approximate Planet Type": get_planet_type(
            planet.get("pl_rade", np.nan)
        )
    }

    details_df = pd.DataFrame(
        list(details.items()),
        columns=["Property", "Value"]
    )

    st.table(details_df)


# =========================================================
# PAGE 5: AI RECOMMENDATIONS
# =========================================================

elif selected_page == "AI Recommendations":

    st.title("🤖 AI Planet Recommendations")

    st.write("""
    This feature uses machine learning to group exoplanets
    based on their available physical characteristics.

    It then finds planets that are close to the selected
    planet in the scaled feature space.
    """)

    model_data = prepare_recommendation_data(df)

    if model_data is None:

        st.warning(
            "Not enough usable numerical data to generate recommendations."
        )

    else:

        st.success(
            "Machine learning clustering model is ready."
        )

        st.write(
            "Features used:",
            ", ".join(model_data["features"])
        )

        planet_names = sorted(
            df["pl_name"].dropna().astype(str).unique()
        )

        selected_planet = st.selectbox(
            "Choose a planet",
            planet_names
        )

        if st.button("✨ Find Similar Planets"):

            recommendations = recommend_similar_planets(
                selected_planet,
                df,
                model_data
            )

            if recommendations.empty:

                st.info(
                    "No similar planets were found in the same cluster."
                )

            else:

                st.subheader(
                    f"Planets similar to {selected_planet}"
                )

                display_columns = [
                    column for column in [
                        "pl_name",
                        "hostname",
                        "pl_rade",
                        "pl_bmasse",
                        "pl_orbper",
                        "pl_eqt",
                        "discoverymethod"
                    ]
                    if column in recommendations.columns
                ]

                st.dataframe(
                    recommendations[display_columns],
                    use_container_width=True
                )

                st.caption(
                    "Recommendations are based on numerical similarity "
                    "in the available dataset. Similarity does not prove "
                    "habitability or the presence of life."
                )


# =========================================================
# PAGE 6: COMPARE PLANETS
# =========================================================

elif selected_page == "Compare Planets":

    st.title("⚖️ Compare Exoplanets")

    st.write(
        "Select two planets to compare their available properties."
    )

    planet_names = sorted(
        df["pl_name"].dropna().astype(str).unique()
    )

    if len(planet_names) < 2:

        st.warning(
            "At least two planets are required for comparison."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            planet_one = st.selectbox(
                "Select first planet",
                planet_names,
                key="compare_one"
            )

        with col2:

            planet_two = st.selectbox(
                "Select second planet",
                planet_names,
                index=1,
                key="compare_two"
            )

        if planet_one == planet_two:

            st.warning(
                "Please select two different planets."
            )

        else:

            first = df[
                df["pl_name"] == planet_one
            ].iloc[0]

            second = df[
                df["pl_name"] == planet_two
            ].iloc[0]

            comparison_columns = [
                "pl_rade",
                "pl_bmasse",
                "pl_orbper",
                "pl_eqt",
                "st_teff"
            ]

            comparison_data = []

            for feature in comparison_columns:

                if feature in df.columns:

                    comparison_data.append({
                        "Property": feature,
                        planet_one: first[feature],
                        planet_two: second[feature]
                    })

            comparison_df = pd.DataFrame(
                comparison_data
            )

            st.subheader("Comparison Table")

            st.dataframe(
                comparison_df,
                use_container_width=True
            )

            st.subheader("Visual Comparison")

            numeric_comparison = pd.DataFrame({
                "Property": [
                    "Radius",
                    "Mass",
                    "Orbital Period",
                    "Equilibrium Temperature",
                    "Host Star Temperature"
                ],
                planet_one: [
                    first.get("pl_rade", np.nan),
                    first.get("pl_bmasse", np.nan),
                    first.get("pl_orbper", np.nan),
                    first.get("pl_eqt", np.nan),
                    first.get("st_teff", np.nan)
                ],
                planet_two: [
                    second.get("pl_rade", np.nan),
                    second.get("pl_bmasse", np.nan),
                    second.get("pl_orbper", np.nan),
                    second.get("pl_eqt", np.nan),
                    second.get("st_teff", np.nan)
                ]
            })

            chart_df = numeric_comparison.melt(
                id_vars="Property",
                var_name="Planet",
                value_name="Value"
            )

            chart_df = chart_df.dropna()

            if not chart_df.empty:

                fig = px.bar(
                    chart_df,
                    x="Property",
                    y="Value",
                    color="Planet",
                    barmode="group",
                    title="Planet Property Comparison"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "No numerical values are available for this comparison."
                )


# =========================================================
# PAGE 7: PLANET STATISTICS
# =========================================================

elif selected_page == "Planet Statistics":

    st.title("📊 Planet Statistics")

    st.write(
        "Explore statistical patterns in the exoplanet dataset."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Planets", len(df))

    if "pl_rade" in df.columns:

        col2.metric(
            "Average Radius",
            format_number(df["pl_rade"].mean())
        )

    else:
        col2.metric("Average Radius", "N/A")

    if "pl_bmasse" in df.columns:

        col3.metric(
            "Average Mass",
            format_number(df["pl_bmasse"].mean())
        )

    else:
        col3.metric("Average Mass", "N/A")

    st.divider()

    if "disc_year" in df.columns:

        st.subheader("Discoveries by Year")

        yearly_data = (
            df["disc_year"]
            .dropna()
            .value_counts()
            .sort_index()
            .reset_index()
        )

        yearly_data.columns = [
            "Year",
            "Discoveries"
        ]

        if not yearly_data.empty:

            fig = px.line(
                yearly_data,
                x="Year",
                y="Discoveries",
                markers=True,
                title="Exoplanet Discoveries Over Time"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    if "discoverymethod" in df.columns:

        st.subheader("Discovery Methods")

        method_data = (
            df["discoverymethod"]
            .value_counts()
            .reset_index()
        )

        method_data.columns = [
            "Method",
            "Count"
        ]

        if not method_data.empty:

            fig = px.pie(
                method_data,
                names="Method",
                values="Count",
                title="Distribution of Discovery Methods"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    if "pl_rade" in df.columns:

        st.subheader("Planet Radius Distribution")

        radius_data = df.dropna(
            subset=["pl_rade"]
        )

        if not radius_data.empty:

            fig = px.histogram(
                radius_data,
                x="pl_rade",
                nbins=30,
                title="Distribution of Planet Radius"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Approximate Planet Types")

        type_data = df.copy()

        type_data["Planet Type"] = (
            type_data["pl_rade"].apply(get_planet_type)
        )

        type_counts = (
            type_data["Planet Type"]
            .value_counts()
            .reset_index()
        )

        type_counts.columns = [
            "Planet Type",
            "Count"
        ]

        fig = px.bar(
            type_counts,
            x="Planet Type",
            y="Count",
            title="Planets by Approximate Radius Category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# PAGE 8: DISCOVERY INSIGHTS
# =========================================================

elif selected_page == "Discovery Insights":

    st.title("🔎 Exoplanet Discovery Insights")

    st.write("""
    Discover patterns in exoplanet discoveries using
    interactive filters and visualizations.
    """)

    insights_df = df.copy()

    if "disc_year" in insights_df.columns:

        valid_years = insights_df["disc_year"].dropna()

        if not valid_years.empty:

            min_year = int(valid_years.min())
            max_year = int(valid_years.max())

            st.subheader("📅 Explore Discovery Period")

            year_range = st.slider(
                "Choose a discovery year range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                key="insights_year_range"
            )

            insights_df = insights_df[
                insights_df["disc_year"].between(
                    year_range[0],
                    year_range[1]
                )
            ]

    if "discoverymethod" in insights_df.columns:

        st.subheader("🔬 Filter by Discovery Method")

        available_methods = sorted(
            insights_df["discoverymethod"]
            .dropna()
            .astype(str)
            .unique()
        )

        chosen_methods = st.multiselect(
            "Select discovery methods",
            available_methods,
            key="insights_methods"
        )

        if chosen_methods:

            insights_df = insights_df[
                insights_df["discoverymethod"]
                .astype(str)
                .isin(chosen_methods)
            ]

    st.divider()

    st.subheader("📌 Filtered Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Planets Found",
        len(insights_df)
    )

    if "hostname" in insights_df.columns:

        col2.metric(
            "Host Stars",
            insights_df["hostname"].nunique()
        )

    else:
        col2.metric("Host Stars", "N/A")

    if "discoverymethod" in insights_df.columns:

        col3.metric(
            "Methods Used",
            insights_df["discoverymethod"].nunique()
        )

    else:
        col3.metric("Methods Used", "N/A")

    if "disc_year" in insights_df.columns:

        st.subheader("📈 Discovery Trend")

        yearly = (
            insights_df["disc_year"]
            .dropna()
            .value_counts()
            .sort_index()
            .reset_index()
        )

        yearly.columns = [
            "Year",
            "Number of Planets"
        ]

        if not yearly.empty:

            fig = px.line(
                yearly,
                x="Year",
                y="Number of Planets",
                markers=True,
                title="Discoveries During Selected Period"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    if "discoverymethod" in insights_df.columns:

        st.subheader("🛰️ Most Common Discovery Methods")

        method_counts = (
            insights_df["discoverymethod"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        method_counts.columns = [
            "Discovery Method",
            "Number of Planets"
        ]

        if not method_counts.empty:

            fig = px.bar(
                method_counts,
                x="Number of Planets",
                y="Discovery Method",
                orientation="h",
                title="Top Discovery Methods"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    if "pl_rade" in insights_df.columns:

        st.subheader("🌍 Planet Radius Insights")

        radius_data = insights_df.dropna(
            subset=["pl_rade"]
        ).copy()

        if not radius_data.empty:

            radius_data["Planet Type"] = (
                radius_data["pl_rade"].apply(get_planet_type)
            )

            type_counts = (
                radius_data["Planet Type"]
                .value_counts()
                .reset_index()
            )

            type_counts.columns = [
                "Planet Type",
                "Count"
            ]

            fig = px.pie(
                type_counts,
                names="Planet Type",
                values="Count",
                title="Approximate Planet Type Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    st.subheader("📥 Download Insights Data")

    create_download_button(
        insights_df,
        "exomatch_discovery_insights.csv",
        "Download Filtered Insights",
        "download_insights"
    )


# =========================================================
# PAGE 9: ABOUT EXOMATCH
# =========================================================

elif selected_page == "About ExoMatch":

    st.title("ℹ️ About ExoMatch")

    st.subheader(
        "AI-Powered Exoplanet Discovery and Recommendation System"
    )

    st.write("""
    ExoMatch is a web-based platform designed to help users
    explore confirmed exoplanets and understand their properties.

    It uses data from the NASA Exoplanet Archive and provides
    interactive tools for searching, filtering, comparing,
    and recommending similar planets.
    """)

    st.divider()

    st.subheader("🎯 Project Objectives")

    st.markdown("""
    - Explore confirmed exoplanet data.
    - Display important planetary characteristics.
    - Allow users to compare different planets.
    - Use machine learning to identify similar planets.
    - Visualize discovery patterns and statistics.
    """)

    st.divider()

    st.subheader("🧠 Machine Learning Approach")

    st.write("""
    ExoMatch uses K-Means clustering to group planets
    according to available numerical features.

    StandardScaler is used to scale the features, and
    Euclidean distance is used to identify planets
    with similar characteristics.
    """)

    st.divider()

    st.subheader("🛠️ Technologies Used")

    st.markdown("""
    - **Python:** Main programming language.
    - **Streamlit:** Web application interface.
    - **Pandas:** Dataset loading and manipulation.
    - **NumPy:** Numerical operations.
    - **Scikit-learn:** Machine learning and clustering.
    - **Plotly:** Interactive data visualizations.
    - **NASA Exoplanet Archive:** Exoplanet dataset source.
    """)

    st.divider()

    st.subheader("⚠️ Limitations")

    st.markdown("""
    - Some planetary measurements may be missing.
    - Planet classifications based on radius are approximate.
    - Similarity does not establish habitability or life.
    - Gallery images are illustrations, not actual photographs
      of individual exoplanets.
    """)

    st.divider()

    st.subheader("🌌 Data Source")

    st.markdown(
        "[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)"
    )

    st.caption(
        "ExoMatch is an educational project for exoplanet data exploration."
    )
