import unittest

import alpha_research_framework.observables as observables
from alpha_research_framework.features.feature import Feature
from alpha_research_framework.features.primitive_feature import PrimitiveFeature


class TestPrimitiveFeatureObservable(unittest.TestCase):
     
    def test_class_var_assertion(self) -> None:
        """
        Verify definiition and type of `OBSERVABLE` asserted when subclassing.
        """

        with self.assertRaises(AttributeError):
            class NoObservable(PrimitiveFeature):
                TAG = Feature.Tag.PREDICTOR
        
        with self.assertRaises(TypeError):
            class IncompatibleObservable(PrimitiveFeature):
                TAG = Feature.Tag.PREDICTOR
                OBSERVABLE = str


class TestPrimitiveFeatureCompute(unittest.TestCase):

    def test_absent_observable(self) -> None:
        """
        Verify a `ValueError` is thrown only when the required observable is
        absent from the market data.
        """

        class DummyObservable(observables.Observable):
            NAME = "dummy"

        class DummyPrimitive(PrimitiveFeature):
            TAG = Feature.Tag.PREDICTOR
            OBSERVABLE = DummyObservable

        DummyPrimitive.compute({DummyObservable: []}, {}, [])
        with self.assertRaises(ValueError):
            DummyPrimitive.compute({}, {}, [])


if __name__ == "__main__":
    unittest.main()
