import unittest
from itertools import combinations
from typing import override

import alpha_research_framework.market_data as md
from alpha_research_framework import Window
from alpha_research_framework.alphas import Alpha
from alpha_research_framework.features import (
    DailyFutureReturns,
    DailyReturns,
    DependencyError,
    Feature,
    HalfYearlyFutureReturns,
    HalfYearlyReturns,
    MonthlyFutureReturns,
    MonthlyReturns,
    QuarterlyFutureReturns,
    QuarterlyReturns,
    Returns,
    WeeklyFutureReturns,
    WeeklyReturns,
    YearlyFutureReturns,
    YearlyReturns,
)
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

    def test_wrap_compute(self) -> None:
        """
        Verify `compute` is wrapped wtih custom error reporting for insufficient
        `cls.DEPENDENCIES or `cross-section` argument.
        """

        class TemporaryFeatureRoot(Feature, registry_root=True, abstract=True):
            pass

        class NotDependedOn(TemporaryFeatureRoot):
            ID = "not_depended_on"
            TAG = Feature.Tag.PREDICTOR
            DEPENDENCIES = set()

        class DependedOn(TemporaryFeatureRoot):
            ID = "depended_on"
            TAG = Feature.Tag.PREDICTOR
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

        with self.assertRaises(DependencyError):
            HasCompute.compute({DependedOn.ID: None})

        with self.assertRaises(DependencyError):
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

    def test_windows_to_returns(self) -> None:
        """
        Verify correct `Returns` or `FutureReturns` is returned per group of
        `Window`s in the correct order.
        """

        N = len(Window)

        windows_ret_pairs: list[tuple[Window, type[Returns]]] = [
            (Window.DAY, DailyReturns),
            (Window.WEEK, WeeklyReturns),
            (Window.MONTH, MonthlyReturns),
            (Window.QUARTER, QuarterlyReturns),
            (Window.HALF_YEAR, HalfYearlyReturns),
            (Window.YEAR, YearlyReturns),
        ]
        for n in range(1, N + 1):
            for pairs in combinations(windows_ret_pairs, n):
                windows = tuple(pair[0] for pair in pairs)
                returns = tuple(pair[1] for pair in pairs)
                self.assertTupleEqual(
                    Alpha._windows_to_returns(*windows),
                    returns,
                )

        windows_fut_ret_pairs: list[tuple[Window, type[FutureReturns]]] = [
            (Window.DAY, DailyFutureReturns),
            (Window.WEEK, WeeklyFutureReturns),
            (Window.MONTH, MonthlyFutureReturns),
            (Window.QUARTER, QuarterlyFutureReturns),
            (Window.HALF_YEAR, HalfYearlyFutureReturns),
            (Window.YEAR, YearlyFutureReturns),
        ]
        for n in range(1, N + 1):
            for pairs in combinations(windows_fut_ret_pairs, n):
                windows = tuple(pair[0] for pair in pairs)
                future_returns = tuple(pair[1] for pair in pairs)
                self.assertTupleEqual(
                    Alpha._windows_to_returns(*windows, future=True),
                    future_returns,
                )


if __name__ == "__main__":
    unittest.main()
