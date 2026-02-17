"""Global city scoring pipeline using public datasets.

This module builds a worldwide city dataset from GeoNames and World Bank
indicators, enriches it with requested fields, scores places, and renders a
Folium map.
"""

from __future__ import annotations

import html
import io
import math
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests

try:
    import folium
    from branca.colormap import linear
    from branca.element import MacroElement, Template
    from folium import plugins
except ImportError as exc:
    raise SystemExit(
        "Folium dependencies are missing. Run: pip3 install -r requirements.txt"
    ) from exc

GEONAMES_CITIES_URL = "https://download.geonames.org/export/dump/cities500.zip"
GEONAMES_COUNTRY_INFO_URL = "https://download.geonames.org/export/dump/countryInfo.txt"
GEONAMES_ADMIN1_URL = "https://download.geonames.org/export/dump/admin1CodesASCII.txt"
WORLD_BANK_BASE = "https://api.worldbank.org/v2/country/all/indicator"

WORLD_BANK_INDICATORS = {
    "life_expectancy": "SP.DYN.LE00.IN",
    "migration_rate": "SM.POP.NETM",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "pollution_pm25": "EN.ATM.PM25.MC.M3",
    "homicide_rate": "VC.IHR.PSRC.P5",
    "internet_users": "IT.NET.USER.ZS",
    "urban_population_share": "SP.URB.TOTL.IN.ZS",
    "unemployment_rate_country": "SL.UEM.TOTL.ZS",
    "freedom_signal": "PV.EST",  # WGI political stability estimate (-2.5 to 2.5)
}

MAJOR_CITY_DATASET_URLS = {
    "geonames_cities500": GEONAMES_CITIES_URL,
    "geonames_country_info": GEONAMES_COUNTRY_INFO_URL,
    "geonames_admin1_codes": GEONAMES_ADMIN1_URL,
    "world_bank_life_expectancy": f"{WORLD_BANK_BASE}/{WORLD_BANK_INDICATORS['life_expectancy']}",
    "world_bank_migration": f"{WORLD_BANK_BASE}/{WORLD_BANK_INDICATORS['migration_rate']}",
    "world_bank_gdp_per_capita": f"{WORLD_BANK_BASE}/{WORLD_BANK_INDICATORS['gdp_per_capita']}",
    "world_bank_pollution_pm25": f"{WORLD_BANK_BASE}/{WORLD_BANK_INDICATORS['pollution_pm25']}",
    "world_bank_homicide_rate": f"{WORLD_BANK_BASE}/{WORLD_BANK_INDICATORS['homicide_rate']}",
}

CONTINENT_NAMES = {
    "AF": "Africa",
    "AN": "Antarctica",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "North America",
    "OC": "Oceania",
    "SA": "South America",
}

LANGUAGE_CODE_TO_NAME = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

# Approximate country hazard groupings for a broad global signal.
EARTHQUAKE_HIGH_ISO3 = {
    "AFG",
    "ARG",
    "CHL",
    "COL",
    "CRI",
    "ECU",
    "FJI",
    "GRC",
    "GTM",
    "IDN",
    "IRN",
    "ISL",
    "ITA",
    "JPN",
    "MEX",
    "MMR",
    "NPL",
    "NZL",
    "PAK",
    "PER",
    "PHL",
    "PNG",
    "SLV",
    "TUR",
    "TWN",
    "USA",
    "VUT",
}

EARTHQUAKE_MEDIUM_ISO3 = {
    "ALB",
    "ARM",
    "AUT",
    "BGR",
    "BIH",
    "CAN",
    "CHN",
    "DEU",
    "DZA",
    "ESP",
    "FRA",
    "GEO",
    "HUN",
    "IND",
    "IRQ",
    "KAZ",
    "KGZ",
    "MAR",
    "MNG",
    "MYS",
    "NOR",
    "POL",
    "PRT",
    "ROU",
    "RUS",
    "SRB",
    "SVN",
    "THA",
    "TJK",
    "UZB",
}

TORNADO_HIGH_ISO3 = {
    "ARG",
    "AUS",
    "BGD",
    "BRA",
    "CAN",
    "IND",
    "PAK",
    "USA",
}

TORNADO_MEDIUM_ISO3 = {
    "CHN",
    "DEU",
    "ESP",
    "FRA",
    "GBR",
    "ITA",
    "JPN",
    "MEX",
    "NLD",
    "NZL",
    "POL",
    "RUS",
    "TUR",
    "UKR",
}

LANDLOCKED_ISO3 = {
    "AFG",
    "AND",
    "ARM",
    "AUT",
    "AZE",
    "BDI",
    "BFA",
    "BTN",
    "BOL",
    "CAF",
    "CHE",
    "CZE",
    "ETH",
    "HUN",
    "KAZ",
    "KGZ",
    "LAO",
    "LIE",
    "LSO",
    "LUX",
    "MDA",
    "MKD",
    "MLI",
    "MNG",
    "MWI",
    "NER",
    "NPL",
    "PRY",
    "RWA",
    "SRB",
    "SSD",
    "SVK",
    "SWZ",
    "TCD",
    "TJK",
    "TKM",
    "UGA",
    "UZB",
    "VAT",
    "ZMB",
    "ZWE",
}

GLOBAL_DEFAULTS = {
    "government_form": "Varies by country (see official country profile)",
    "visa_info": "Visa requirements vary by your passport and destination policy.",
}

COUNTRY_INFO_COLUMNS = [
    "country_code",
    "iso3",
    "iso_numeric",
    "fips",
    "country",
    "capital",
    "area_sq_km",
    "country_population",
    "continent_code",
    "tld",
    "currency",
    "currency_name",
    "phone",
    "postal_code_format",
    "postal_code_regex",
    "languages_raw",
    "country_geonameid",
    "neighbors",
    "equivalent_fips",
]

CITY_COLUMNS = [
    "geonameid",
    "city",
    "city_ascii",
    "alternate_names",
    "lat",
    "lon",
    "feature_class",
    "feature_code",
    "country_code",
    "cc2",
    "state_code",
    "admin2_code",
    "admin3_code",
    "admin4_code",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modification_date",
]


def normalize_score(
    series: pd.Series,
    invert: bool = False,
    clip_quantiles: Tuple[float, float] = (0.02, 0.98),
) -> pd.Series:
    clipped = pd.to_numeric(series, errors="coerce").copy()
    clipped = clipped.fillna(clipped.median())

    low_q, high_q = clip_quantiles
    if 0 <= low_q < high_q <= 1:
        low_value = float(clipped.quantile(low_q))
        high_value = float(clipped.quantile(high_q))
        if not math.isclose(low_value, high_value):
            clipped = clipped.clip(lower=low_value, upper=high_value)

    minimum = float(clipped.min())
    maximum = float(clipped.max())
    if math.isclose(minimum, maximum):
        return pd.Series([100.0] * len(series), index=series.index)

    normalized = (clipped - minimum) / (maximum - minimum)
    if invert:
        normalized = 1 - normalized
    return normalized * 100


def _read_zip_member(url: str, member_name: str, timeout: int = 120) -> io.BytesIO:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        if member_name not in zf.namelist():
            raise ValueError(f"{member_name} not found in {url}")
        with zf.open(member_name) as handle:
            return io.BytesIO(handle.read())


def _clean_text_lines(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    return "\n".join(lines)


def fetch_geonames_cities() -> pd.DataFrame:
    buffer = _read_zip_member(GEONAMES_CITIES_URL, "cities500.txt")
    data = pd.read_csv(buffer, sep="\t", header=None, names=CITY_COLUMNS, low_memory=False)

    data = data[data["feature_class"] == "P"].copy()

    numeric_cols = ["lat", "lon", "population", "elevation", "dem"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["lat", "lon", "population"])
    data = data[data["population"] > 0].copy()

    data["city"] = data["city"].astype(str).str.strip()
    data["country_code"] = data["country_code"].astype(str).str.upper().str.strip()
    data["state_code"] = data["state_code"].fillna("").astype(str).str.strip()

    return data


def fetch_country_info() -> pd.DataFrame:
    response = requests.get(GEONAMES_COUNTRY_INFO_URL, timeout=60)
    response.raise_for_status()

    cleaned = _clean_text_lines(response.text)
    countries = pd.read_csv(
        io.StringIO(cleaned),
        sep="\t",
        header=None,
        names=COUNTRY_INFO_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )

    countries["country_code"] = countries["country_code"].str.upper().str.strip()
    countries["iso3"] = countries["iso3"].str.upper().str.strip()
    countries["continent_code"] = countries["continent_code"].str.upper().str.strip()
    countries["continent"] = countries["continent_code"].map(CONTINENT_NAMES).fillna("Unknown")
    countries["currency"] = countries["currency"].replace("", "Unknown")

    countries["languages_raw"] = countries["languages_raw"].fillna("")
    countries["country"] = countries["country"].replace("", "Unknown")

    return countries


def fetch_admin1_codes() -> pd.DataFrame:
    response = requests.get(GEONAMES_ADMIN1_URL, timeout=60)
    response.raise_for_status()

    data = pd.read_csv(
        io.StringIO(response.text),
        sep="\t",
        header=None,
        names=["admin1_key", "state", "state_ascii", "state_geonameid"],
        dtype=str,
        keep_default_na=False,
    )

    split = data["admin1_key"].str.split(".", n=1, expand=True)
    data["country_code"] = split[0].str.upper().str.strip()
    data["state_code"] = split[1].fillna("").str.strip()

    data["state"] = data["state"].replace("", "Unknown")

    return data[["country_code", "state_code", "state"]]


def _world_bank_indicator_latest(indicator: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    page = 1
    pages = 1

    while page <= pages:
        response = requests.get(
            f"{WORLD_BANK_BASE}/{indicator}",
            params={"format": "json", "per_page": 20000, "page": page},
            timeout=75,
        )
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, list) or len(payload) < 2:
            break

        meta = payload[0]
        pages = int(meta.get("pages", 1)) if isinstance(meta, dict) else 1
        chunk = payload[1] if isinstance(payload[1], list) else []
        rows.extend(chunk)
        page += 1

    latest_values: Dict[str, float] = {}
    latest_years: Dict[str, int] = {}

    for row in rows:
        iso3 = str(row.get("countryiso3code") or "").strip().upper()
        if not iso3 or iso3 in latest_values:
            continue

        value = row.get("value")
        if value is None:
            continue

        try:
            latest_values[iso3] = float(value)
        except (TypeError, ValueError):
            continue

        try:
            latest_years[iso3] = int(row.get("date"))
        except (TypeError, ValueError):
            pass

    result = pd.DataFrame(
        {
            "iso3": list(latest_values.keys()),
            indicator: [latest_values[k] for k in latest_values],
            f"{indicator}_year": [latest_years.get(k) for k in latest_values],
        }
    )

    return result


def fetch_world_bank_country_profile() -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    for output_name, indicator in WORLD_BANK_INDICATORS.items():
        df = _world_bank_indicator_latest(indicator)
        df = df.rename(
            columns={
                indicator: output_name,
                f"{indicator}_year": f"{output_name}_year",
            }
        )

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="iso3", how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame(columns=["iso3"] + list(WORLD_BANK_INDICATORS.keys()))

    # Scale WGI political stability signal to an intuitive 0-100 freedom-like score.
    merged["freedom_score"] = ((merged["freedom_signal"] + 2.5) / 5.0) * 100.0
    merged["freedom_score"] = merged["freedom_score"].clip(lower=0, upper=100)

    return merged


def _humanize_languages(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return "Unknown"

    seen = set()
    names: List[str] = []

    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        base = token.split("-")[0].lower()
        if base in seen:
            continue
        seen.add(base)
        names.append(LANGUAGE_CODE_TO_NAME.get(base, base.upper()))
        if len(names) == 3:
            break

    return ", ".join(names) if names else "Unknown"


def _state_risk_bucket(
    iso3: str,
    high_iso3: set[str],
    medium_iso3: set[str],
) -> str:
    if iso3 in high_iso3:
        return "High"
    if iso3 in medium_iso3:
        return "Medium"
    return "Low"


def _estimate_summer_high(lat: float, continent: str) -> float:
    base = 99.0 - abs(lat) * 0.45
    adjustments = {
        "Africa": 3.0,
        "Asia": 1.0,
        "Europe": -2.0,
        "North America": 0.0,
        "South America": 1.0,
        "Oceania": 0.0,
        "Antarctica": -20.0,
    }
    return float(max(40.0, min(115.0, base + adjustments.get(continent, 0.0))))


def _estimate_winter_low(lat: float, continent: str) -> float:
    base = 78.0 - abs(lat) * 1.55
    adjustments = {
        "Africa": 4.0,
        "Asia": -1.0,
        "Europe": -4.0,
        "North America": -2.0,
        "South America": 1.0,
        "Oceania": 1.0,
        "Antarctica": -35.0,
    }
    return float(max(-40.0, min(80.0, base + adjustments.get(continent, 0.0))))


def _safe_series_fill(series: pd.Series, fallback: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(float(numeric.median()))
    return numeric.fillna(float(fallback))


def add_requested_category_columns(data: pd.DataFrame) -> pd.DataFrame:
    enriched = data.copy()

    enriched["continent"] = enriched["continent"].fillna("Unknown")
    enriched["country"] = enriched["country"].fillna("Unknown")
    enriched["state"] = enriched["state"].fillna("Unknown")
    enriched["city"] = enriched["city"].fillna("Unknown")

    enriched["language_spoken"] = enriched["languages_raw"].map(_humanize_languages)

    enriched["summer_high"] = enriched.apply(
        lambda row: _estimate_summer_high(float(row["lat"]), str(row["continent"])),
        axis=1,
    )
    enriched["winter_low"] = enriched.apply(
        lambda row: _estimate_winter_low(float(row["lat"]), str(row["continent"])),
        axis=1,
    )

    enriched["earthquakes"] = enriched["iso3"].map(
        lambda code: _state_risk_bucket(
            str(code),
            high_iso3=EARTHQUAKE_HIGH_ISO3,
            medium_iso3=EARTHQUAKE_MEDIUM_ISO3,
        )
    )
    enriched["tornadoes"] = enriched["iso3"].map(
        lambda code: _state_risk_bucket(
            str(code),
            high_iso3=TORNADO_HIGH_ISO3,
            medium_iso3=TORNADO_MEDIUM_ISO3,
        )
    )

    enriched["elevation"] = _safe_series_fill(enriched["elevation"], fallback=0.0)
    enriched["dem"] = _safe_series_fill(enriched["dem"], fallback=0.0)

    enriched["beach"] = enriched.apply(
        lambda row: (
            "No"
            if str(row["iso3"]) in LANDLOCKED_ISO3
            else ("Yes" if float(row["elevation"]) <= 80 else "Possible")
        ),
        axis=1,
    )
    enriched["mountains"] = enriched.apply(
        lambda row: "Yes"
        if (float(row["elevation"]) >= 600 or float(row["dem"]) >= 900)
        else "No",
        axis=1,
    )

    enriched["gdp_per_capita"] = _safe_series_fill(enriched["gdp_per_capita"], fallback=15000.0)
    enriched["internet_users"] = _safe_series_fill(enriched["internet_users"], fallback=55.0)
    enriched["urban_population_share"] = _safe_series_fill(
        enriched["urban_population_share"],
        fallback=58.0,
    )
    enriched["life_expectancy"] = _safe_series_fill(enriched["life_expectancy"], fallback=72.0)
    enriched["migration_rate"] = _safe_series_fill(enriched["migration_rate"], fallback=0.0)
    enriched["pollution_pm25"] = _safe_series_fill(enriched["pollution_pm25"], fallback=25.0)
    enriched["homicide_rate"] = _safe_series_fill(enriched["homicide_rate"], fallback=5.0)
    enriched["unemployment_rate_country"] = _safe_series_fill(
        enriched["unemployment_rate_country"],
        fallback=7.0,
    )
    enriched["freedom_score"] = _safe_series_fill(enriched["freedom_score"], fallback=60.0)

    # Rent and living cost proxies anchored to country GDP-per-capita and city size.
    city_scale = normalize_score(enriched["population"].map(lambda value: math.log1p(float(value)))) / 100.0
    monthly_income_proxy = enriched["gdp_per_capita"] / 12.0
    enriched["avg_rent_2_bed"] = (monthly_income_proxy * (0.28 + 0.24 * city_scale)).clip(200, 12000)
    enriched["monthly_living_cost"] = (enriched["avg_rent_2_bed"] / 0.35).clip(400, 30000)

    enriched["government_form"] = GLOBAL_DEFAULTS["government_form"]
    enriched["language"] = enriched["language_spoken"]
    enriched["currency"] = enriched["currency"].replace("", "Unknown")

    # Transparent proxies for requested LGBT-related fields.
    enriched["lgbt_equality_index"] = (
        0.60 * enriched["freedom_score"]
        + 0.25 * enriched["internet_users"]
        + 0.15 * enriched["urban_population_share"]
    ).clip(0, 100)
    enriched["lgbt_legal_index"] = (
        0.70 * enriched["freedom_score"] + 0.30 * enriched["urban_population_share"]
    ).clip(0, 100)
    enriched["lgbt_public_opinion_index"] = (
        0.55 * enriched["freedom_score"] + 0.45 * enriched["internet_users"]
    ).clip(0, 100)

    enriched["visa_info"] = GLOBAL_DEFAULTS["visa_info"]

    enriched["cost_of_living"] = normalize_score(enriched["monthly_living_cost"])
    crime_raw = 0.70 * enriched["homicide_rate"] + 0.30 * enriched["unemployment_rate_country"]
    enriched["crime"] = normalize_score(crime_raw)
    enriched["pollution_level"] = normalize_score(enriched["pollution_pm25"])
    enriched["quality_of_life"] = (
        0.45 * normalize_score(enriched["life_expectancy"])
        + 0.35 * normalize_score(enriched["gdp_per_capita"])
        + 0.20 * (100 - enriched["pollution_level"])
    )

    # Requested title-style columns.
    enriched["Continent"] = enriched["continent"]
    enriched["Country"] = enriched["country"]
    enriched["City"] = enriched["city"]
    enriched["Population"] = enriched["population"].round(0)
    enriched["Language Spoken"] = enriched["language_spoken"]
    enriched["Summer High"] = enriched["summer_high"].round(1)
    enriched["Winter Low"] = enriched["winter_low"].round(1)
    enriched["Earthquakes"] = enriched["earthquakes"]
    enriched["Tornadoes"] = enriched["tornadoes"]
    enriched["Beach"] = enriched["beach"]
    enriched["Mountians"] = enriched["mountains"]
    enriched["Mountains"] = enriched["mountains"]
    enriched["Visa Info"] = enriched["visa_info"]
    enriched["Avg Rent (2 Bed)"] = enriched["avg_rent_2_bed"].round(0)
    enriched["Cost of living"] = enriched["cost_of_living"]
    enriched["Quality of life"] = enriched["quality_of_life"]
    enriched["Crime"] = enriched["crime"]
    enriched["Pollution Level"] = enriched["pollution_level"]

    return enriched


def build_major_cities_dataset(
    min_population: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    cities = fetch_geonames_cities()
    cities = cities[cities["population"] >= float(min_population)].copy()

    admin1 = fetch_admin1_codes()
    countries = fetch_country_info()
    wb = fetch_world_bank_country_profile()

    cities = cities.merge(admin1, on=["country_code", "state_code"], how="left")
    cities = cities.merge(countries, on="country_code", how="left")
    cities = cities.merge(wb, on="iso3", how="left")

    cities["state"] = cities["state"].fillna("")
    cities["state"] = cities.apply(
        lambda row: str(row["state"]).strip()
        if str(row["state"]).strip()
        else (str(row["state_code"]).strip() if str(row["state_code"]).strip() else "Unknown"),
        axis=1,
    )

    cities = add_requested_category_columns(cities)

    metadata: Dict[str, object] = {
        "scope": "global_places",
        "dataset": "GeoNames cities500 (cities with population >= 500 where available)",
        "min_population": int(min_population),
        "cities_count": int(len(cities)),
        "countries_count": int(cities["country_code"].nunique()),
        "continents_count": int(cities["continent"].nunique()),
    }

    if "life_expectancy_year" in cities.columns:
        year = pd.to_numeric(cities["life_expectancy_year"], errors="coerce")
        if year.notna().any():
            metadata["life_expectancy_year"] = int(year.dropna().mode().iloc[0])

    if "migration_rate_year" in cities.columns:
        year = pd.to_numeric(cities["migration_rate_year"], errors="coerce")
        if year.notna().any():
            metadata["migration_year"] = int(year.dropna().mode().iloc[0])

    return cities, metadata


def score_major_cities(data: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    scored = data.copy()

    scored["cost_of_living"] = normalize_score(scored["monthly_living_cost"])
    scored["crime"] = normalize_score(scored["crime"])
    scored["pollution_level"] = normalize_score(scored["pollution_level"])

    scored["affordability_score"] = normalize_score(scored["monthly_living_cost"], invert=True)
    scored["safety_score"] = 100 - scored["crime"]

    transit_city = normalize_score(scored["population"].map(lambda value: math.log1p(float(value))))
    transit_country = normalize_score(scored["urban_population_share"])
    scored["transit_score"] = 0.65 * transit_city + 0.35 * transit_country

    amenities_connectivity = normalize_score(scored["internet_users"])
    amenities_health = normalize_score(scored["life_expectancy"])
    amenities_air = 100 - scored["pollution_level"]
    scored["amenities_score"] = (
        0.40 * amenities_connectivity + 0.35 * amenities_health + 0.25 * amenities_air
    )

    scored["overall_score"] = (
        weights["affordability"] * scored["affordability_score"]
        + weights["safety"] * scored["safety_score"]
        + weights["transit"] * scored["transit_score"]
        + weights["amenities"] * scored["amenities_score"]
    )

    scored["quality_of_life"] = (
        0.55 * scored["overall_score"]
        + 0.25 * normalize_score(scored["life_expectancy"])
        + 0.20 * (100 - scored["pollution_level"])
    )

    scored["Cost of living"] = scored["cost_of_living"]
    scored["Quality of life"] = scored["quality_of_life"]
    scored["Crime"] = scored["crime"]
    scored["Pollution Level"] = scored["pollution_level"]

    scored = scored.sort_values("overall_score", ascending=False).reset_index(drop=True)
    scored["rank"] = scored.index + 1

    rounded_cols = [
        "overall_score",
        "affordability_score",
        "safety_score",
        "transit_score",
        "amenities_score",
        "quality_of_life",
        "cost_of_living",
        "crime",
        "pollution_level",
        "summer_high",
        "winter_low",
        "avg_rent_2_bed",
        "monthly_living_cost",
        "life_expectancy",
        "migration_rate",
        "freedom_score",
        "lgbt_equality_index",
        "lgbt_legal_index",
        "lgbt_public_opinion_index",
    ]
    for col in rounded_cols:
        if col in scored.columns:
            scored[col] = pd.to_numeric(scored[col], errors="coerce").round(4)

    return scored


def _city_popup(row: pd.Series) -> str:
    return f"""
    <div style="min-width:300px;font-family:Arial,sans-serif;">
      <h4 style="margin:0 0 8px 0;">{html.escape(str(row['city']))}, {html.escape(str(row['state']))}</h4>
      <div style="font-size:13px;line-height:1.35;">
        <b>Country:</b> {html.escape(str(row['country']))}<br>
        <b>Continent:</b> {html.escape(str(row['continent']))}<br>
        <b>Rank:</b> #{int(row['rank'])}<br>
        <b>Overall Score:</b> {row['overall_score']:.1f}/100<br><hr style="margin:8px 0;">
        <b>Population:</b> {int(row['population']):,}<br>
        <b>Language:</b> {html.escape(str(row['language_spoken']))}<br>
        <b>Currency:</b> {html.escape(str(row['currency']))}<br>
        <b>Avg Rent (2 Bed):</b> ${row['avg_rent_2_bed']:,.0f}<br>
        <b>Cost of Living:</b> {row['cost_of_living']:.1f}<br>
        <b>Quality of Life:</b> {row['quality_of_life']:.1f}<br>
        <b>Crime:</b> {row['crime']:.1f}<br>
        <b>LGBT Equality:</b> {row['lgbt_equality_index']:.1f}<br>
        <b>LGBT Legal:</b> {row['lgbt_legal_index']:.1f}<br>
        <b>LGBT Public Opinion:</b> {row['lgbt_public_opinion_index']:.1f}<br>
        <b>Life Expectancy:</b> {row['life_expectancy']:.1f}<br>
        <b>Migration Rate:</b> {row['migration_rate']:.1f}<br>
        <b>Pollution:</b> {row['pollution_level']:.1f}<br>
        <b>Summer High:</b> {row['summer_high']:.1f}°F<br>
        <b>Winter Low:</b> {row['winter_low']:.1f}°F<br>
        <b>Earthquakes:</b> {html.escape(str(row['earthquakes']))}<br>
        <b>Tornadoes:</b> {html.escape(str(row['tornadoes']))}<br>
        <b>Beach:</b> {html.escape(str(row['beach']))}<br>
        <b>Mountains:</b> {html.escape(str(row['mountains']))}
      </div>
    </div>
    """


def _estimated_city_radius_m(population: float) -> float:
    # Global proxy: estimate city footprint area from population and convert to radius.
    pop = max(float(population), 1.0)
    area_sq_km = max(15.0, min(2500.0, pop / 3500.0))
    return math.sqrt(area_sq_km / math.pi) * 1000.0


def _add_summary_panel(
    map_object: folium.Map,
    weights: Dict[str, float],
    metadata: Dict[str, object],
) -> None:
    def _fmt_int(value: object) -> str:
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return "N/A"

    min_population_label = _fmt_int(metadata.get("min_population"))
    rows_label = _fmt_int(metadata.get("cities_count"))
    countries_label = _fmt_int(metadata.get("countries_count"))

    weight_items = "".join(
        (
            f"<li><span>{key.title()}</span>"
            f"<strong>{value * 100:.0f}%</strong></li>"
        )
        for key, value in weights.items()
    )

    template = Template(
        f"""
        {{% macro html(this, kwargs) %}}
        <style>
          #scout-card {{
            position: fixed;
            bottom: 18px;
            left: 18px;
            z-index: 9999;
            width: 300px;
            background: rgba(255, 255, 255, 0.96);
            border-radius: 10px;
            border: 1px solid #d6dde8;
            box-shadow: 0 8px 20px rgba(10, 25, 47, 0.15);
            padding: 12px;
            font-family: Arial, sans-serif;
          }}
          #scout-card h3 {{
            margin: 0 0 6px 0;
            font-size: 16px;
            color: #0f172a;
          }}
          #scout-card p {{
            margin: 0 0 6px 0;
            font-size: 12px;
            color: #334155;
          }}
          #scout-card .section-title {{
            margin: 10px 0 5px 0;
            font-size: 12px;
            font-weight: 700;
            color: #1f2937;
            text-transform: uppercase;
            letter-spacing: 0.03em;
          }}
          #scout-card ul {{
            list-style: none;
            margin: 0;
            padding: 0;
          }}
          #scout-card li {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            padding: 3px 0;
            color: #1f2937;
          }}
          #scout-card strong {{
            color: #0f766e;
          }}
        </style>
        <div id="scout-card">
          <h3>Neighborhood Scout</h3>
          <p>Global City / State / Country Mode</p>
          <p>Min Population >= {min_population_label} | Cities: {rows_label} | Countries: {countries_label}</p>
          <div class="section-title">Weighting</div>
          <ul>{weight_items}</ul>
        </div>
        {{% endmacro %}}
        """
    )

    macro = MacroElement()
    macro._template = template
    map_object.get_root().add_child(macro)


def build_major_cities_map(
    scored: pd.DataFrame,
    weights: Dict[str, float],
    metadata: Dict[str, object],
    top_n: int,
    zoom: int,
) -> folium.Map:
    _ = top_n
    center = [20.0, 0.0]
    scout_map = folium.Map(
        location=center,
        zoom_start=zoom,
        control_scale=True,
        tiles="cartodbpositron",
    )

    plugins.Fullscreen(
        position="topright",
        title="Full screen",
        title_cancel="Exit full screen",
        force_separate_button=True,
    ).add_to(scout_map)
    plugins.MiniMap(toggle_display=True, position="bottomright").add_to(scout_map)

    score_min = float(scored["overall_score"].min())
    score_max = float(scored["overall_score"].max())
    if math.isclose(score_min, score_max):
        score_max = score_min + 1.0

    colormap = linear.RdYlGn_09.scale(score_min, score_max)
    colormap.caption = "Global Place Fit Score"

    heat_layer = folium.FeatureGroup(name="Score Heatmap", show=False)

    city_layer = folium.FeatureGroup(name="City Limit Outlines", show=True)
    for _, row in scored.iterrows():
        score = float(row["overall_score"])
        color = colormap(score)
        location = [float(row["lat"]), float(row["lon"])]
        popup = folium.Popup(_city_popup(row), max_width=370)

        folium.Circle(
            location=location,
            radius=_estimated_city_radius_m(float(row["population"])),
            color=color,
            weight=2,
            opacity=0.95,
            fill=True,
            fill_color=color,
            fill_opacity=0.08,
            popup=popup,
            tooltip=(
                f"{row['city']}, {row['country']} | "
                f"#{int(row['rank'])} | {score:.1f}"
            ),
        ).add_to(city_layer)
    city_layer.add_to(scout_map)

    if len(scored) <= 80_000:
        heat_data = [
            [float(row["lat"]), float(row["lon"]), float(row["overall_score"]) / 100]
            for _, row in scored.iterrows()
        ]
        plugins.HeatMap(heat_data, radius=20, blur=18, min_opacity=0.22).add_to(heat_layer)
        heat_layer.add_to(scout_map)

    colormap.add_to(scout_map)
    folium.LayerControl(collapsed=False).add_to(scout_map)

    _add_summary_panel(
        map_object=scout_map,
        weights=weights,
        metadata=metadata,
    )

    return scout_map
