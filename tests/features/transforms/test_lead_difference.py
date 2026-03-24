import unittest

import numpy as np

from alpha_research_framework.features.transforms.lead_difference import (
    LeadDifference,
)


class TestLeadDifference(unittest.TestCase):

    def test_compute(self) -> None:
        """Verify `arr` is modified to be the lead difference of itself."""

        SIZE = 1000
        rng = np.random.default_rng(0)

        for lead in [1, 10, 100]:

            arr = rng.uniform(size=SIZE)
            nan_mask = rng.uniform(size=arr.shape) < 0.1
            arr[nan_mask] = np.nan

            expected = np.concatenate(
                [
                    arr[lead:] - arr[:-lead],
                    np.full(lead, np.nan),
                ]
            )
            LeadDifference.compute(arr, lead=lead)
            
            np.testing.assert_array_equal(arr, expected)
        

if __name__ == "__main__":
    unittest.main()
