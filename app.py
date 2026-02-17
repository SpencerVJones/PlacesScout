from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import streamlit.runtime as st_runtime

from major_cities import (
    MAJOR_CITY_DATASET_URLS,
    build_major_cities_dataset,
    build_major_cities_map,
    score_major_cities,
)
from scoring_config import DEFAULT_WEIGHTS
from world_cache_schema import validate_world_cache

DEFAULT_CACHE_FILE = Path("data/world_cities_cache.csv")
DEFAULT_OUTPUT_FILE = Path("output/world_cities_map.html")
DEFAULT_ZOOM = 2
MAP_POINT_LIMIT = 12_000
MAP_POINT_TOP_BIAS = 20

SORT_OPTIONS = {
    "Best overall fit": ("overall_score", False),
    "Most affordable": ("cost_of_living", True),
    "Lowest crime": ("crime", True),
    "Best LGBT equality": ("lgbt_equality_index", False),
    "Best quality of life": ("quality_of_life", False),
    "Highest population": ("population", False),
}

DEFAULT_FILTER_STATE = {
    "search_query": "",
    "sort_label": "Best overall fit",
    "shortlist_rows": 100,
    "max_cost_of_living": 85,
    "max_crime": 65,
    "min_quality_of_life": 45,
    "min_lgbt_equality": 55,
    "require_beach": False,
    "require_mountains": False,
}

DEFAULT_WEIGHT_SLIDER_STATE = {
    "weight_affordability": int(DEFAULT_WEIGHTS["affordability"] * 100),
    "weight_safety": int(DEFAULT_WEIGHTS["safety"] * 100),
    "weight_transit": int(DEFAULT_WEIGHTS["transit"] * 100),
    "weight_amenities": int(DEFAULT_WEIGHTS["amenities"] * 100),
}

DEFAULT_SIDEBAR_STATE = {
    "min_population": 50_000,
    "map_point_limit": MAP_POINT_LIMIT,
    "map_top_bias": MAP_POINT_TOP_BIAS,
}


def _initialize_ui_state() -> None:
    defaults = {
        **DEFAULT_FILTER_STATE,
        **DEFAULT_WEIGHT_SLIDER_STATE,
        **DEFAULT_SIDEBAR_STATE,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_filter_controls() -> None:
    for key, value in DEFAULT_FILTER_STATE.items():
        st.session_state[key] = value


def _reset_weight_controls() -> None:
    for key, value in DEFAULT_WEIGHT_SLIDER_STATE.items():
        st.session_state[key] = value


def _baseline_signature(
    min_population: int,
    weights: Dict[str, float],
) -> Tuple[int, Tuple[Tuple[str, float], ...]]:
    rounded = tuple(sorted((key, round(value, 6)) for key, value in weights.items()))
    return int(min_population), rounded


def _format_location_label(city: object, state: object, country: object) -> str:
    parts = [str(city).strip()]
    state_text = str(state).strip()
    if state_text and state_text.lower() != "unknown":
        parts.append(state_text)
    country_text = str(country).strip()
    if country_text:
        parts.append(country_text)
    return ", ".join(parts)


def _streamlit_runtime_exists() -> bool:
    try:
        return bool(st_runtime.exists())
    except Exception:
        return False


def _cache_data_passthrough(*_args, **_kwargs):
    def decorator(func):
        return func

    return decorator


def _safe_cache_data(*args, **kwargs):
    if _streamlit_runtime_exists():
        return st.cache_data(*args, **kwargs)
    return _cache_data_passthrough(*args, **kwargs)


@_safe_cache_data(ttl=86400, show_spinner=False)
def fetch_live_data_world(min_population: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    return build_major_cities_dataset(min_population=min_population)


@_safe_cache_data(show_spinner=False)
def load_cached_world(path_str: str, modified_time: float) -> pd.DataFrame:
    _ = modified_time
    return pd.read_csv(Path(path_str), low_memory=False)


def normalize_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("At least one weight must be greater than zero.")
    return {key: value / total for key, value in raw_weights.items()}


def get_scored_data_world(
    min_population: int,
    cache_file: Path,
    weights: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, object], str]:
    metadata: Dict[str, object] = {
        "scope": "global_places",
        "dataset": "GeoNames cities500",
        "min_population": min_population,
    }

    try:
        live_data, metadata = fetch_live_data_world(min_population=min_population)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        live_data.to_csv(cache_file, index=False)
        scored = score_major_cities(live_data, weights=weights)
        metadata["cities_count"] = int(len(scored))
        metadata["countries_count"] = int(scored["country_code"].nunique())
        return scored, metadata, "live"
    except (requests.RequestException, ValueError) as exc:
        if cache_file.exists():
            modified = cache_file.stat().st_mtime
            cached_data = load_cached_world(str(cache_file), modified)
            validate_world_cache(cached_data)
            scored = score_major_cities(cached_data, weights=weights)
            metadata["cities_count"] = int(len(scored))
            metadata["countries_count"] = int(scored["country_code"].nunique())
            return scored, metadata, f"cache_fallback ({exc})"
        raise ValueError(f"Live fetch failed and no cache was found: {exc}") from exc


def _results_table_columns() -> List[str]:
    return [
        "rank",
        "city",
        "state",
        "country",
        "overall_score",
        "quality_of_life",
        "cost_of_living",
        "crime",
        "lgbt_equality_index",
        "lgbt_legal_index",
        "lgbt_public_opinion_index",
        "avg_rent_2_bed",
        "population",
        "beach",
        "mountains",
    ]


def _apply_moving_filters(
    scored: pd.DataFrame,
    search_query: str,
    max_cost_of_living: float,
    max_crime: float,
    min_quality_of_life: float,
    min_lgbt_equality: float,
    require_beach: bool,
    require_mountains: bool,
) -> pd.DataFrame:
    filtered = scored.copy()

    filtered = filtered[filtered["cost_of_living"] <= max_cost_of_living]
    filtered = filtered[filtered["crime"] <= max_crime]
    filtered = filtered[filtered["quality_of_life"] >= min_quality_of_life]
    filtered = filtered[filtered["lgbt_equality_index"] >= min_lgbt_equality]

    if require_beach:
        filtered = filtered[filtered["beach"].isin(["Yes", "Possible"])]
    if require_mountains:
        filtered = filtered[filtered["mountains"] == "Yes"]

    query = search_query.strip()
    if query:
        query_mask = (
            filtered["city"].str.contains(query, case=False, na=False)
            | filtered["state"].str.contains(query, case=False, na=False)
            | filtered["country"].str.contains(query, case=False, na=False)
        )
        filtered = filtered[query_mask]

    return filtered


def _sort_results(filtered: pd.DataFrame, sort_label: str) -> pd.DataFrame:
    col, ascending = SORT_OPTIONS.get(sort_label, SORT_OPTIONS["Best overall fit"])
    sorted_data = filtered.sort_values(col, ascending=ascending).reset_index(drop=True)
    sorted_data["rank"] = sorted_data.index + 1
    return sorted_data


def _limit_map_rows(
    filtered: pd.DataFrame,
    max_map_points: int,
    top_bias: int,
) -> Tuple[pd.DataFrame, str]:
    if len(filtered) <= max_map_points:
        return filtered.copy(), ""

    dedupe_key = "geonameid" if "geonameid" in filtered.columns else "__dedupe_key__"
    work = filtered.copy()
    if dedupe_key == "__dedupe_key__":
        work[dedupe_key] = (
            work["city"].astype(str)
            + "|"
            + work["state"].astype(str)
            + "|"
            + work["country"].astype(str)
        )

    keep_top = min(top_bias * 10, max(300, int(max_map_points * 0.25)))
    top_slice = work.head(keep_top)

    remaining_budget = max_map_points - len(top_slice)
    if remaining_budget > 0:
        pop_slice = work.iloc[keep_top:].nlargest(remaining_budget, "population")
        limited = (
            pd.concat([top_slice, pop_slice], ignore_index=True)
            .drop_duplicates(subset=[dedupe_key], keep="first")
            .sort_values("overall_score", ascending=False)
            .reset_index(drop=True)
        )
    else:
        limited = top_slice.copy().reset_index(drop=True)

    if "__dedupe_key__" in limited.columns:
        limited = limited.drop(columns=["__dedupe_key__"])

    note = (
        f"Showing {len(limited):,} map outlines (from {len(filtered):,} matches) "
        f"for performance."
    )
    return limited, note


def _profile_rows(row: pd.Series) -> List[Tuple[str, str]]:
    return [
        (
            "Location",
            _format_location_label(
                city=row["city"],
                state=row["state"],
                country=row["country"],
            ),
        ),
        ("Overall Score", f"{float(row['overall_score']):.1f}/100"),
        ("Quality of Life", f"{float(row['quality_of_life']):.1f}"),
        ("Cost of Living", f"{float(row['cost_of_living']):.1f}"),
        ("Crime", f"{float(row['crime']):.1f}"),
        ("LGBT Equality", f"{float(row['lgbt_equality_index']):.1f}"),
        ("LGBT Legal", f"{float(row['lgbt_legal_index']):.1f}"),
        ("LGBT Public Opinion", f"{float(row['lgbt_public_opinion_index']):.1f}"),
        ("Avg Rent (2 Bed)", f"${float(row['avg_rent_2_bed']):,.0f}"),
        ("Population", f"{int(row['population']):,}"),
        ("Beach", str(row["beach"])),
        ("Mountains", str(row["mountains"])),
    ]


def app() -> None:
    st.set_page_config(
        page_title="Neighborhood Scout",
        page_icon=":world_map:",
        layout="wide",
    )

    st.title("Neighborhood Scout")
    st.caption(
        "Find cities that match your moving priorities: budget, safety, lifestyle, and LGBT friendliness."
    )
    _initialize_ui_state()

    map_container = st.container()
    filters_container = st.container()
    results_container = st.container()

    with st.sidebar:
        st.header("Moving Baseline")

        min_population = st.slider(
            "Minimum Population",
            min_value=0,
            max_value=2_000_000,
            step=5_000,
            key="min_population",
            help="Start with larger cities for more stable comparisons.",
        )

        st.subheader("Category Weights")
        affordability = st.slider(
            "Affordability",
            min_value=0,
            max_value=100,
            key="weight_affordability",
        )
        safety = st.slider(
            "Safety",
            min_value=0,
            max_value=100,
            key="weight_safety",
        )
        transit = st.slider(
            "Transit",
            min_value=0,
            max_value=100,
            key="weight_transit",
        )
        amenities = st.slider(
            "Amenities",
            min_value=0,
            max_value=100,
            key="weight_amenities",
        )

        raw_weights = {
            "affordability": float(affordability),
            "safety": float(safety),
            "transit": float(transit),
            "amenities": float(amenities),
        }

        try:
            weights = normalize_weights(raw_weights)
        except ValueError as exc:
            st.error(f"Invalid weight setup: {exc}")
            st.stop()

        st.caption(
            "Effective weighting: "
            f"A {weights['affordability']:.0%} | "
            f"S {weights['safety']:.0%} | "
            f"T {weights['transit']:.0%} | "
            f"M {weights['amenities']:.0%}"
        )
        st.button("Reset Weights", on_click=_reset_weight_controls)

        st.subheader("Map Performance")
        map_point_limit = st.slider(
            "Max map outlines",
            min_value=1_000,
            max_value=20_000,
            step=1_000,
            key="map_point_limit",
            help="Lower this if map rendering feels slow.",
        )
        map_top_bias = st.slider(
            "Keep more top-ranked cities",
            min_value=5,
            max_value=50,
            step=5,
            key="map_top_bias",
            help="Higher values preserve more top results before population balancing.",
        )

        left, right = st.columns(2)
        refresh_clicked = left.button("Refresh Live Data")
        run_clicked = right.button("Regenerate Scores", type="primary")

    with filters_container:
        title_col, reset_col = st.columns([4, 1])
        with title_col:
            st.subheader("Search & Moving Filters")
        with reset_col:
            st.button("Reset Filters", on_click=_reset_filter_controls)

        search_col, sort_col, short_col = st.columns([2, 1, 1])
        with search_col:
            search_query = st.text_input(
                "Search city, state, or country",
                placeholder="e.g., Lisbon, Texas, Japan",
                key="search_query",
            )
        with sort_col:
            sort_label = st.selectbox(
                "Sort Results",
                options=list(SORT_OPTIONS.keys()),
                key="sort_label",
            )
        with short_col:
            shortlist_rows = st.slider(
                "Shortlist Rows",
                min_value=25,
                max_value=300,
                step=25,
                key="shortlist_rows",
            )

        filter_left, filter_right = st.columns(2)
        with filter_left:
            max_cost_of_living = st.slider(
                "Max Cost of Living",
                min_value=0,
                max_value=100,
                key="max_cost_of_living",
                help=(
                    "Global normalized scale. Higher-income countries like the U.S. "
                    "often sit above 75."
                ),
            )
            max_crime = st.slider(
                "Max Crime",
                min_value=0,
                max_value=100,
                key="max_crime",
            )
            require_beach = st.checkbox("Prefer beach access", key="require_beach")
        with filter_right:
            min_quality_of_life = st.slider(
                "Min Quality of Life",
                min_value=0,
                max_value=100,
                key="min_quality_of_life",
            )
            min_lgbt_equality = st.slider(
                "Min LGBT Equality",
                min_value=0,
                max_value=100,
                key="min_lgbt_equality",
            )
            require_mountains = st.checkbox("Prefer mountains", key="require_mountains")

    cache_file = DEFAULT_CACHE_FILE
    output_file = DEFAULT_OUTPUT_FILE

    if refresh_clicked:
        fetch_live_data_world.clear()
        st.session_state.pop("scored_data", None)
        st.session_state.pop("scored_metadata", None)
        st.session_state.pop("scored_source", None)
        st.session_state.pop("baseline_signature", None)
        st.session_state.pop("last_loaded_at", None)

    signature = _baseline_signature(min_population=min_population, weights=weights)
    has_cached_scores = isinstance(st.session_state.get("scored_data"), pd.DataFrame)
    needs_rescore = (
        refresh_clicked
        or run_clicked
        or not has_cached_scores
        or st.session_state.get("baseline_signature") != signature
    )

    if needs_rescore:
        with st.spinner("Fetching and scoring city data..."):
            try:
                scored, metadata, source = get_scored_data_world(
                    min_population=min_population,
                    cache_file=cache_file,
                    weights=weights,
                )
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            scored = scored.sort_values("overall_score", ascending=False).reset_index(drop=True)
            scored["rank"] = scored.index + 1
            st.session_state["scored_data"] = scored
            st.session_state["scored_metadata"] = metadata
            st.session_state["scored_source"] = source
            st.session_state["baseline_signature"] = signature
            st.session_state["last_loaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        scored = st.session_state["scored_data"]
        metadata = st.session_state.get("scored_metadata", {})
        source = str(st.session_state.get("scored_source", "session_cache"))

    filtered = _apply_moving_filters(
        scored=scored,
        search_query=search_query,
        max_cost_of_living=float(max_cost_of_living),
        max_crime=float(max_crime),
        min_quality_of_life=float(min_quality_of_life),
        min_lgbt_equality=float(min_lgbt_equality),
        require_beach=require_beach,
        require_mountains=require_mountains,
    )
    if filtered.empty:
        st.warning("No cities match your current filters. Reset or loosen one or two constraints.")
        st.button("Reset Filters to Defaults", on_click=_reset_filter_controls)
        st.stop()

    filtered = _sort_results(filtered, sort_label=sort_label)

    map_scored, map_note = _limit_map_rows(
        filtered=filtered,
        max_map_points=int(map_point_limit),
        top_bias=int(map_top_bias),
    )

    map_metadata = dict(metadata)
    map_metadata["cities_count"] = int(len(filtered))
    map_metadata["countries_count"] = int(filtered["country_code"].nunique())
    with st.spinner("Rendering map..."):
        scout_map = build_major_cities_map(
            scored=map_scored,
            weights=weights,
            metadata=map_metadata,
            top_n=0,
            zoom=DEFAULT_ZOOM,
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    scout_map.save(str(output_file))

    with map_container:
        st.subheader("Main Map")
        load_time = st.session_state.get("last_loaded_at")
        status_bits = [f"Data source: `{source}`"]
        if load_time:
            status_bits.append(f"Scored at: `{load_time}`")
        status_bits.append(f"Saved map: `{output_file}`")
        st.caption(" | ".join(status_bits))
        if str(source).startswith("cache_fallback"):
            st.info("Live fetch failed, so the app used your local cached dataset.")

        map_html = scout_map.get_root().render()
        components.html(map_html, height=760, scrolling=True)
        if map_note:
            st.info(map_note)

    median_rent = float(filtered["avg_rent_2_bed"].median())
    median_lgbt = float(filtered["lgbt_equality_index"].median())
    median_quality = float(filtered["quality_of_life"].median())

    with results_container:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Matched Cities", f"{len(filtered):,}")
        col2.metric("Countries", f"{filtered['country_code'].nunique():,}")
        col3.metric("Median 2-Bed Rent", f"${median_rent:,.0f}")
        col4.metric("Median QoL", f"{median_quality:.1f}")
        col5.metric("Median LGBT Score", f"{median_lgbt:.1f}")

        tab_shortlist, tab_profile = st.tabs(["Shortlist", "City Profile"])

    with tab_shortlist:
        st.subheader("Shortlist")
        st.dataframe(
            filtered.loc[:, _results_table_columns()].head(shortlist_rows),
            use_container_width=True,
            hide_index=True,
        )

        download_data = filtered.drop(columns=["geometry"], errors="ignore")
        st.download_button(
            label="Download Filtered Shortlist CSV",
            data=download_data.to_csv(index=False).encode("utf-8"),
            file_name="moving_shortlist.csv",
            mime="text/csv",
        )

    with tab_profile:
        picker_data = filtered.head(2000).copy()
        picker_data["city_label"] = picker_data.apply(
            lambda row: _format_location_label(
                city=row["city"],
                state=row["state"],
                country=row["country"],
            ),
            axis=1,
        )
        selected_label = st.selectbox(
            "Find a city profile",
            options=picker_data["city_label"].tolist(),
        )
        selected_row = picker_data[picker_data["city_label"] == selected_label].iloc[0]

        left_col, right_col = st.columns(2)
        profile_rows = _profile_rows(selected_row)
        midpoint = len(profile_rows) // 2

        with left_col:
            for label, value in profile_rows[:midpoint]:
                st.markdown(f"**{label}:** {value}")
        with right_col:
            for label, value in profile_rows[midpoint:]:
                st.markdown(f"**{label}:** {value}")

        st.caption(
            f"Quality of life median in your current result set: {median_quality:.1f}. "
            "Use the filters above the shortlist to tighten or broaden matches."
        )

    with st.expander("Data Sources"):
        for name, url in MAJOR_CITY_DATASET_URLS.items():
            st.markdown(f"- `{name}`: [{url}]({url})")


if __name__ == "__main__":
    if not _streamlit_runtime_exists():
        raise SystemExit("Run this UI with: python3 -m streamlit run app.py")
    app()
