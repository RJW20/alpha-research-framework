# Testing

## Purpose

The test suite is designed to provide confidence at several levels rather than relying solely on end-to-end research examples.

The framework contains a mixture of:

- unit tests for individual classes and validation rules
- structural tests for the expression and dependency graphs
- numerical tests for transformational and statistical operations
- integration-style tests for downloading and universe construction
- end-to-end tests that exercise the complete research path

The suite is intentionally varied because much of the framework's complexity is architectural rather than algorithmic: metaclasses, class registration, dynamically generated types, dependency deduction and disk-backed arrays all need different forms of validation.

## Running the tests

The project exposes a Poetry script:

```bash
poetry run tests
```

The script runs unittest discovery under `tests/`.

End-to-end tests can use a separately prepared download directory. Pass it with:

```bash
poetry run tests --e2e-data-dir /path/to/data
```

When no end-to-end data directory is supplied, tests that require it are skipped rather than manufacturing external market data.

## Test layers

### 1. Class-variable and inheritance validation

The framework relies heavily on class-level declarations. Concrete classes are expected to declare values such as `ID`,
`CATEGORY`, `SIGNAL`, `HORIZONS`, `SERIES`, `TAG` and `MEASURES` with the correct types.

These tests verify behaviour such as:

- required class variables are defined, correctly typed and have acceptable value at class definition time
- registry roots are isolated correctly
- duplicate IDs cannot be registered under the same root

This is important because the framework uses Python's type system and class creation as part of its user-facing API.
Errors should therefore appear when a researcher defines an invalid concept, rather than much later during evaluation.

---

### 2. Operator and expression construction

The framework allows `Series` and `Signal` objects to be built purely through arithmetic expressions and also for new
`Series` to be created by transforms. It uses metaclasses and functions that work with types to dynamically generate
subclasses.

The purpose of these tests is to verify that these generated subclasses:

- preserve/continue the dependency structure 
- produce the expected runtime values when executing

This protects the framework's central declarative expression model.

---

### 3. Dependency deduction

One of the more important structural parts of the framework is the automatic discovery of the data required by an 
`Alpha`.

The tests for this construct artificial expression graphs and verify that the framework finds the correct roots. For
example, a simple graph such as:

```text
Observable A
     |
     v
  Series B
     |
     v
  Series C
```

must be able to be resolved to market-data root `A` when only supplying `C`. More complicated merge and diamond graphs
are also tested to ensure that shared dependencies are not missed or duplicated.

This is a key test area because a dependency-deduction error can cause an otherwise valid `Alpha` to fail much later with missing data.

---

### 4. Transformations and statistics

The framework contains a handful or purely numerical functions providing both temporal numerical transforms and
statistical operations.

The testing of these layers are done independently of the higher-level systems that use them and cover:

- numerical correctness
- edge cases
- NaN values
- speed-up over standard NumPy implementations

The validity of these functions is of utmost importance for having confidence in any successful alphas identified by the
framework.

---

### 5. Download integration tests

One of the three main execution stages of the framework is the retrieval of daily stock data via yfinance.

The current tests exercise real yfinance retrieval for a small set of known tickers with different behaviour throughout
a chosen window (full trading history, IPO'd, delisted, fictional) and verify:

- requested date ranges
- generated log
- generated metadata
- adjusted-close values at known endpoints

These tests verify the correctness of the `download` entry point rather than guaranteeing anything about the cleanliness
of the data it retrieves.

---

### 6. Universe construction

The second of the three main execution stages of the framework is the transformation from downloaded ticker data into a
read-only cross-sectional `Universe` instance.

The tests cover both the ingestion of the data and the behaviour of the `Universe` container, including:

- sector/industry validation and resulting ticker selection
- existence and calendar alignment of per ticker data
- unified market data creation
- mask creation according to input thresholds
- indexing for a cross-section (with `TAG` selection)

This ensures generated research environments reflect the intended conditions set out by the caller.

---

### 7. End-to-end research tests

These tests exercise the complete pipeline using a prepared data directory (which may simply come from the `download` entry point).

The tests build custom informationless alphas:

- `RandomNoise` - generates a random signal per stock drawn from the standard normal distribution
- `Constant` - assigns a value of 1 to each stock

and checks that the average information coefficient is approximately zero.

This is particularly valuable because it is a negative-control experiment: the framework should not manufacture a
persistent predictive relationship from an informationless signal.

---

## Test abstractions

### Registry isolation

Because concrete framework classes register themselves globally under their registry roots, tests that define temporary
subclasses use `RegistryIsolatedTestCase`.

It snapshots the relevant registry before each test and restores it afterwards. This prevents dynamically defined test
classes from leaking into subsequent tests.

---

### Synthetic arrays

Numerical unit tests use deterministic/random helper arrays rather than requiring market data. This keeps the numerical
layer fast and makes failures reproducible.

---

### Synthetic download directories

`tests/utils/create_download_dir.py` can construct a minimal download-directory structure containing metadata and
deterministic per-ticker data. This allows `Universe` and `Series` building tests to exercise the expected on-disk
contract without depending on a live data provider.

---