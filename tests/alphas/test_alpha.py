import unittest
from itertools import combinations
from typing import override

import alpha_research_framework.features as features
import alpha_research_framework.market_data as md
from alpha_research_framework import Window
from alpha_research_framework.alphas import Alpha
from alpha_research_framework.features.future_returns import FutureReturns
from alpha_research_framework.universe import CrossSection
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


class TestAlphaDependencies(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `DEPENDENCIES` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoDependencies(Alpha):
                ID = "no_dependencies"
                CATEGORY = "testing_dependencies"

        with self.assertRaises(TypeError):
            class IncompatibleDependenciesContainer(Alpha):
                ID = "incompatible_dependencies_container"
                CATEGORY = "test_dependencies"
                DEPENDENCIES = list()

        with self.assertRaises(TypeError):
            class IncompatibleDependenciesElement(Alpha):
                ID = "incompatible_dependencies_element"
                CATEGORY = "test_dependencies"
                DEPENDENCIES = {1}

    def test_future_returns(self) -> None:
        """
        Verify `FutureReturns` dependency created for all windows in `HORIZONS`.
        """

        class WithHorizons(Alpha):
            ID = "with_horizons"
            CATEGORY = "testing_dependencies"
            DEPENDENCIES = set()
            HORIZONS = set(Window)

        self.assertEqual(len(WithHorizons.DEPENDENCIES), len(Window))

    def test_wrap_compute(self) -> None:
        """
        Verify `compute` is wrapped wtih custom error reporting for insufficient
        `cls.DEPENDENCIES or `cross-section` argument.
        """

        class TemporaryFeatureRoot(
            features.Feature,
            registry_root=True,
            abstract=True
        ):
            pass

        class NotDependedOn(TemporaryFeatureRoot):
            ID = "not_depended_on"
            TAG = features.Feature.Tag.PREDICTOR
            DEPENDENCIES = set()

        class DependedOn(TemporaryFeatureRoot):
            ID = "depended_on"
            TAG = features.Feature.Tag.PREDICTOR
            DEPENDENCIES = set()

        class HasCompute(Alpha, abstract=True):
            DEPENDENCIES = {DependedOn}
            @classmethod
            @override
            def compute(cls, x: CrossSection) -> md.Array:
                # attempt to use non-dependency
                x[NotDependedOn.ID]
                # attempt to use the dependency
                x[DependedOn.ID]

        HasCompute._wrap_compute()

        with self.assertRaises(features.DependencyError):
            HasCompute.compute({DependedOn.ID: None})

        with self.assertRaises(features.DependencyError):
            HasCompute.compute({NotDependedOn.ID: None})

        HasCompute.compute({NotDependedOn.ID: None, DependedOn.ID: None})


class TestAlphaHorizons(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_class_var_assertion(self) -> None:
        """
        Verify definition and type of `HORIZONS` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoHorizons(Alpha):
                ID = "no_horizons"
                CATEGORY = "testing_horizons"
                DEPENDENCIES = set()

        with self.assertRaises(TypeError):
            class IncompatibleHorizonsContainer(Alpha):
                ID = "incompatible_horizons_container"
                CATEGORY = "testing_horizons"
                DEPENDENCIES = set()
                HORIZONS = list()

        with self.assertRaises(TypeError):
            class IncompatibleHorizonsElement(Alpha):
                ID = "incompatible_horizons_element"
                CATEGORY = "testing_horizons"
                DEPENDENCIES = set()
                HORIZONS = {1}


class TestAlphaWindowsToFeatures(RegistryIsolatedTestCase):
    REGISTRY_OWNER = Alpha

    def test_windows_to_returns(self) -> None:
        """
        Verify correct `Returns` or `FutureReturns` features are returned per
        group of `Window`s in the same order as passed.
        """

        N = len(Window)

        windows_returns_pairs: list[tuple[Window, type[features.Returns]]] = [
            (Window.DAY, features.DailyReturns),
            (Window.WEEK, features.WeeklyReturns),
            (Window.MONTH, features.MonthlyReturns),
            (Window.QUARTER, features.QuarterlyReturns),
            (Window.HALF_YEAR, features.HalfYearlyReturns),
            (Window.YEAR, features.YearlyReturns),
        ]
        for n in range(1, N + 1):
            for pairs in combinations(windows_returns_pairs, n):
                windows = tuple(pair[0] for pair in pairs)
                returns = tuple(pair[1] for pair in pairs)
                self.assertTupleEqual(
                    Alpha._windows_to_returns(*windows),
                    returns,
                )

        windows_fut_ret_pairs: list[tuple[Window, type[FutureReturns]]] = [
            (Window.DAY, features.DailyFutureReturns),
            (Window.WEEK, features.WeeklyFutureReturns),
            (Window.MONTH, features.MonthlyFutureReturns),
            (Window.QUARTER, features.QuarterlyFutureReturns),
            (Window.HALF_YEAR, features.HalfYearlyFutureReturns),
            (Window.YEAR, features.YearlyFutureReturns),
        ]
        for n in range(1, N + 1):
            for pairs in combinations(windows_fut_ret_pairs, n):
                windows = tuple(pair[0] for pair in pairs)
                future_returns = tuple(pair[1] for pair in pairs)
                self.assertTupleEqual(
                    Alpha._windows_to_returns(*windows, future=True),
                    future_returns,
                )
        
    def test_windows_to_volatilities(self) -> None:
        """
        Verify correct `Volatilities` features are returned per group of
        `Window`s in the correct order as passed.
        """

        N = len(Window)

        windows_vol_pairs: list[tuple[Window, type[features.Volatility]]] = [
            (Window.DAY, features.DailyVolatility),
            (Window.WEEK, features.WeeklyVolatility),
            (Window.MONTH, features.MonthlyVolatility),
            (Window.QUARTER, features.QuarterlyVolatility),
            (Window.HALF_YEAR, features.HalfYearlyVolatility),
            (Window.YEAR, features.YearlyVolatility),
        ]
        for n in range(1, N + 1):
            for pairs in combinations(windows_vol_pairs, n):
                windows = tuple(pair[0] for pair in pairs)
                volatilities = tuple(pair[1] for pair in pairs)
                self.assertTupleEqual(
                    Alpha._windows_to_volatilities(*windows),
                    volatilities,
                )


if __name__ == "__main__":
    unittest.main()
