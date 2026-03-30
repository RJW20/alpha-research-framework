import unittest

from tests.utils import set_e2e_data_dir


def main():
    """Run all tests of the form test_*.py inside tests/ and its subfolders."""

    set_e2e_data_dir()

    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir='tests',
        pattern='test*.py',
        top_level_dir='.',
    )
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    main()
