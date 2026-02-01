import unittest


def main():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='tests', pattern='*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    main()
