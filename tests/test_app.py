import unittest

import pandas as pd

from app import _apply_moving_filters, normalize_weights


class NormalizeWeightsTests(unittest.TestCase):
    def test_normalize_weights_sums_to_one(self) -> None:
        weights = normalize_weights(
            {
                "affordability": 30.0,
                "safety": 35.0,
                "transit": 20.0,
                "amenities": 15.0,
            }
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertAlmostEqual(weights["safety"], 0.35, places=6)

    def test_normalize_weights_rejects_zero_total(self) -> None:
        with self.assertRaises(ValueError):
            normalize_weights(
                {
                    "affordability": 0.0,
                    "safety": 0.0,
                    "transit": 0.0,
                    "amenities": 0.0,
                }
            )


class FilterBehaviorTests(unittest.TestCase):
    def test_beach_and_mountain_filters(self) -> None:
        scored = pd.DataFrame(
            [
                {
                    "city": "Beach City",
                    "state": "CA",
                    "country": "United States",
                    "cost_of_living": 70.0,
                    "crime": 40.0,
                    "quality_of_life": 60.0,
                    "lgbt_equality_index": 70.0,
                    "beach": "Yes",
                    "mountains": "No",
                },
                {
                    "city": "Mountain City",
                    "state": "CO",
                    "country": "United States",
                    "cost_of_living": 72.0,
                    "crime": 35.0,
                    "quality_of_life": 65.0,
                    "lgbt_equality_index": 68.0,
                    "beach": "No",
                    "mountains": "Yes",
                },
                {
                    "city": "Both City",
                    "state": "WA",
                    "country": "United States",
                    "cost_of_living": 74.0,
                    "crime": 30.0,
                    "quality_of_life": 70.0,
                    "lgbt_equality_index": 75.0,
                    "beach": "Possible",
                    "mountains": "Yes",
                },
            ]
        )

        filtered = _apply_moving_filters(
            scored=scored,
            search_query="",
            max_cost_of_living=90.0,
            max_crime=65.0,
            min_quality_of_life=45.0,
            min_lgbt_equality=55.0,
            require_beach=True,
            require_mountains=True,
        )

        self.assertEqual(filtered["city"].tolist(), ["Both City"])


if __name__ == "__main__":
    unittest.main()
