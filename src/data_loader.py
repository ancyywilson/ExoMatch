
import pandas as pd
import requests
from pathlib import Path

# NASA Exoplanet Archive API
NASA_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def download_exoplanet_data():

    query = """
    SELECT
        pl_name,
        hostname,
        discoverymethod,
        disc_year,
        pl_rade,
        pl_bmasse,
        pl_orbper,
        pl_eqt,
        st_teff
    FROM pscomppars
    """

    params = {
        "query": query,
        "format": "csv"
    }

    response = requests.get(NASA_URL, params=params, timeout=60)
    response.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))

    project_root = Path(__file__).resolve().parent.parent
    data_folder = project_root / "data"
    data_folder.mkdir(exist_ok=True)

    output_file = data_folder / "confirmed_planets.csv"
    df.to_csv(output_file, index=False)

    print("Exoplanet data downloaded successfully!")
    print("Total planets:", len(df))
    print("Saved to:", output_file)

if __name__ == "__main__":
    download_exoplanet_data()
