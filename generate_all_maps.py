"""Generate the global Neighborhood Scout map in one run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

from major_cities import (
    build_major_cities_dataset,
    build_major_cities_map,
    score_major_cities,
)
from scoring_config import parse_weights
from world_cache_schema import validate_world_cache

DEFAULT_WORLD_CACHE = Path("data/world_cities_cache.csv")
MIN_ZOOM = 1
MAX_ZOOM = 12


@dataclass(frozen=True)
class WorldBuildResult:
    map_obj: object
    scored: pd.DataFrame
    metadata: Dict[str, object]
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the global Neighborhood Scout map in a single run."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where output HTML files will be written.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="world_cities_map.html",
        help="Output HTML filename (inside output-dir).",
    )
    parser.add_argument(
        "--min-population",
        type=int,
        default=50_000,
        help="Minimum population threshold for global city mode.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Top markers for map metadata summary.",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=2,
        help="Initial zoom for world map.",
    )
    parser.add_argument(
        "--weights",
        default="",
        help=(
            "Comma-separated weights, e.g. "
            "'affordability=0.3,safety=0.35,transit=0.2,amenities=0.15'."
        ),
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=DEFAULT_WORLD_CACHE,
        help="Global scored cache CSV path.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.min_population < 0:
        raise SystemExit("--min-population must be zero or greater.")
    if args.zoom < MIN_ZOOM or args.zoom > MAX_ZOOM:
        raise SystemExit(f"--zoom must be between {MIN_ZOOM} and {MAX_ZOOM}.")
    if args.top_n < 0:
        raise SystemExit("--top-n must be zero or greater.")


def _load_world_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Global cache not found: {path}")
    data = pd.read_csv(path, low_memory=False)
    validate_world_cache(data)
    return data


def _fallback_metadata(scored: pd.DataFrame, min_population: int) -> Dict[str, object]:
    return {
        "scope": "global_places",
        "dataset": "GeoNames cities500",
        "min_population": min_population,
        "cities_count": len(scored),
        "countries_count": int(scored["country_code"].nunique()),
    }


def build_world(
    weights: Dict[str, float],
    min_population: int,
    top_n: int,
    zoom: int,
    cache_file: Path,
) -> WorldBuildResult:
    try:
        raw, metadata = build_major_cities_dataset(min_population=min_population)
        raw.to_csv(cache_file, index=False)
        scored = score_major_cities(raw, weights=weights)
        source = "live"
    except (requests.RequestException, ValueError) as exc:
        if not cache_file.exists():
            raise RuntimeError(
                f"Global live fetch failed and no cache found: {exc}"
            ) from exc
        raw = _load_world_cache(cache_file)
        scored = score_major_cities(raw, weights=weights)
        metadata = _fallback_metadata(scored, min_population=min_population)
        source = f"cache_fallback ({exc})"

    map_obj = build_major_cities_map(
        scored=scored,
        weights=weights,
        metadata=metadata,
        top_n=top_n,
        zoom=zoom,
    )
    return WorldBuildResult(
        map_obj=map_obj,
        scored=scored,
        metadata=metadata,
        source=source,
    )


def _print_summary(result: WorldBuildResult, output_path: Path) -> None:
    print("Generated map:")
    print(f"- Global cities: {output_path.resolve()} | source={result.source}")
    print("\nCounts:")
    print(f"- Cities scored: {len(result.scored)}")
    print(f"- Countries represented: {result.scored['country_code'].nunique()}")
    print("\nMetadata:")
    print(f"- Global: {result.metadata}")
    print(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")


def main() -> int:
    args = parse_args()
    _validate_args(args)

    try:
        weights = parse_weights(args.weights)
    except ValueError as exc:
        raise SystemExit(f"Invalid weights: {exc}") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_file.parent.mkdir(parents=True, exist_ok=True)

    result = build_world(
        weights=weights,
        min_population=args.min_population,
        top_n=args.top_n,
        zoom=args.zoom,
        cache_file=args.cache_file,
    )

    output_path = args.output_dir / args.output
    result.map_obj.save(str(output_path))
    _print_summary(result, output_path=output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
