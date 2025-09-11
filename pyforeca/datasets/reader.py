"""Module for reading data."""

import os

import pandas as pd
import sklearn

_CWD = os.path.dirname(os.path.realpath(__file__))
_DATA_DIR = os.path.join(_CWD, "data")
_EU_STOCK_MARKETS_FNAME = "eu-stock-markets.csv"
_FIGSHARE_TEMPERATURE_URL = "https://figshare.com/ndownloader/files/4938964"


HEMISPHERE_LOOKUP = {
    "north": [
        "Kherson",
        "Kiev",
        "Lvov",
        "Marseille",
        "Odesa",
        "Paris",
        "Stockholm",
        "Tokyo",
        "Tottori",
        "Uppsala",
        "Warsaw",
        "Wroclaw",
    ],
    "south": [
        "Auckland",
        "Brasília",
        "Canoas",
        "Cape Town",
        "Hamilton",
        "Johannesburg",
    ],
}


def _fahrenheit_to_celsius(fahrenheit: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a temperature from Fahrenheit to Celsius.

    Args:
      fahrenheit: The temperature in Fahrenheit.

    Returns:
      The equivalent temperature in Celsius.
    """
    celsius = (fahrenheit - 32) * 5 / 9.0
    return celsius


def load_eu_stock_markets_data() -> pd.DataFrame:
    """Loads the EU stock market data. See also R EuStockMarkets."""
    df = pd.read_csv(os.path.join(_DATA_DIR, _EU_STOCK_MARKETS_FNAME))
    return df


def request_temperature_data() -> pd.DataFrame:
    """Loads temperature data from URL.

    See also
    https://figshare.com/ndownloader/files/4938964
    """
    df = pd.read_csv(_FIGSHARE_TEMPERATURE_URL)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + df["month"].astype(str), format="%Y%m"
    )
    return df


def process_temperature_data(
    df: pd.DataFrame,
    start_date: str = "1859-10-01",
    ffill: bool = True,
    to_celsius: bool = False,
    zero_mean_unit_variance: bool = False,
) -> pd.DataFrame:
    """Processes temperature data and turns into wide-format for ForeCA processing."""
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + df["month"].astype(str), format="%Y%m"
    )
    df = df.rename(
        columns={c: c.lower().replace(" ", "_").replace(".", "") for c in df.columns}
    )
    df = df.loc[df["country"].notna()]
    df = df.set_index(["country", "city", "date"], verify_integrity=True)
    df = df.sort_index()

    df_wide = pd.pivot_table(
        df, index="date", columns="city", values="averagetemperaturefahr"
    )
    full_months = pd.date_range(
        df.index.get_level_values("date").min(),
        df.index.get_level_values("date").max(),
        freq="MS",
    )
    df_wide = df_wide.reindex(full_months)
    df_wide = df_wide.loc[df_wide.index >= start_date]

    if ffill:
        df_wide = df_wide.ffill()

    if to_celsius:
        df_wide = _fahrenheit_to_celsius(df_wide)

    df_wide = df_wide[HEMISPHERE_LOOKUP["south"] + HEMISPHERE_LOOKUP["north"]]

    if zero_mean_unit_variance:
        mod_scaler = sklearn.preprocessing.StandardScaler()
        mod_scaler.fit(df_wide)
        df_wide = pd.DataFrame(
            mod_scaler.transform(df_wide), index=df_wide.index, columns=df_wide.columns
        )
    return df_wide


def split_by_hemisphere(df_wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Splits data by hemisphere."""
    df_split = {k: df_wide[v] for k, v in HEMISPHERE_LOOKUP.items()}
    return df_split
