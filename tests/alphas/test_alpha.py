import unittest
from itertools import combinations

import alpha_research_framework.alphas.factors as factors
import alpha_research_framework.features as features
from alpha_research_framework import Window
from alpha_research_framework.alphas import Alpha
from tests.utils import RegistryIsolatedTestCase


class TestAlphaCategory(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `CATEGORY` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoCategory(Alpha):
                ID = "no_category"

        with self.assertRaises(TypeError):
            class IncompatibleCategory(Alpha):
                ID = "incompatible_tag"
                CATEGORY = 1


class TestAlphaSignal(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_class_var_assertion(self) -> None:
        """Verify definition and type of `SIGNAL` asserted when subclassing."""

        with self.assertRaises(AttributeError):
            class NoSignal(Alpha):
                ID = "no_signal"
                CATEGORY = "testing_signal"

        with self.assertRaises(TypeError):
            class IncompatibleSignal(Alpha):
                ID = "incompatible_signal"
                CATEGORY = "testing_signal"
                SIGNAL = features.Feature


class TestAlphaHorizons(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `HORIZONS` asserted when subclassing.
        """

        class Dummy(factors.Factor):
            REQUIRED_FEATURES = set()

        with self.assertRaises(AttributeError):
            class NoHorizons(Alpha):
                ID = "no_horizons"
                CATEGORY = "testing_horizons"
                SIGNAL = Dummy

        with self.assertRaises(TypeError):
            class IncompatibleHorizonsContainer(Alpha):
                ID = "incompatible_horizons_container"
                CATEGORY = "testing_horizons"
                SIGNAL = Dummy
                HORIZONS = list()

        with self.assertRaises(TypeError):
            class IncompatibleHorizonsElement(Alpha):
                ID = "incompatible_horizons_element"
                CATEGORY = "testing_horizons"
                SIGNAL = Dummy
                HORIZONS = {1}

    def test_required_features_forward_returns(self) -> None:
        """
        Verify `ForwardReturns` dependency created for all windows in
        `HORIZONS`.
        """

        class Dummy(factors.Factor):
            REQUIRED_FEATURES = set()

        class WithHorizons(Alpha):
            ID = "with_horizons"
            CATEGORY = "testing_horizons"
            SIGNAL = Dummy
            HORIZONS = set(Window)

        self.assertEqual(
            len(WithHorizons.REQUIRED_FEATURES),
            len(Dummy.REQUIRED_FEATURES) + len(Window)
        )


class TestAlphaWindowsToForwardReturns(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_windows_to_forward_returns(self) -> None:
        """
        Verify `ForwardReturns` features are returned per group of `Window`s in
        the same order as passed.
        """

        N = len(Window)

        windows_fut_ret_pairs: list[tuple[Window, type[features.Feature]]] = [
            (Window.DAY,        features.ForwardReturns1d),
            (Window.WEEK,       features.ForwardReturns5d),
            (Window.MONTH,      features.ForwardReturns20d),
            (Window.QUARTER,    features.ForwardReturns63d),
            (Window.HALF_YEAR,  features.ForwardReturns126d),
            (Window.YEAR,       features.ForwardReturns252d),
        ]
        for n in range(1, N + 1):
            for pairs in combinations(windows_fut_ret_pairs, n):
                windows = tuple(pair[0] for pair in pairs)
                forward_returns = tuple(pair[1] for pair in pairs)
                self.assertTupleEqual(
                    Alpha._windows_to_forward_returns(*windows),
                    forward_returns,
                )


if __name__ == "__main__":
    unittest.main()
