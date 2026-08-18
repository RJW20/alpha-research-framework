# Alpha Research Framework

A Python framework for researching cross-sectional equity alphas.

This framework is not designed for live trading or high-frequency execution. Its purpose is to provide a robust
environment for researching and validating cross-sectional equity signals before they are considered for production
deployment.

## Motivation

The ultimate goal of the framework is to make familiar research ideas first-class objects. It exposes stateless
operators that can be composed algebraically, allowing researchers to construct new signals using the same notation they
would use to describe them mathematically. For example, expressions like:

```python
(arf.signals.Returns252d - arf.signals.Returns20d) / arf.signals.Volatility252d
```

are immediately interpretable from a research standpoint, whilst the framework itself can treat the result as another operator, automatically generating its computation logic and managing the caching of intermediate results when executing.

## Getting Started

### Requirements

The project currently targets **Python 3.13+**.

Runtime dependencies include NumPy, pandas, Numba, PyArrow, pandas-market-calendars and yfinance.

### Installation

Installing directly with `pip`:

```bash
pip install git+https://github.com/RJW20/alpha-research-framework.git@v0.1.0
```

Using with a `pyproject.toml`:

```toml
dependencies = [
    ...
    "alpha-research-framework @ git+https://github.com/RJW20/alpha-research-framework.git@v0.1.0",
    ...
]
```

## API

### Download market data

#### Entry point

```python
def download(
    dest: Path,
    tickers: Iterable[str],
    start_date: str,
    years: int,
) -> None:
```

#### Parameters

- `dest` : `Path` - Directory to write all downloaded/created files to (created or overwritten).
- `tickers` : `Iterable[str]` - Listing of all tickers to download data for.
- `start_date` : `str` (yyyy-mm-dd) - First date to retrieve data for.
- `years` : `int` - Number of years to download daily data over.

#### Effect

Creates a self-contained research data directory containing per-ticker Parquet files, static metadata and a download
log.

#### Example

```python
from pathlib import Path

import alpha_research_framework as arf

arf.download(
    Path("data"),
    ["AAPL", "MSFT", "NVDA"],
    "2015-01-01",
    10,
)
```

#### For larger downloads (recommended)

A script is provided as part of the package:

```bash
download [-d DESTINATION] [-t TICKERS] [-s START_DATE] [-y YEARS]
```

with options:

- `-d`, `--destination` - Directory to write all downloaded/created files to (created or overwritten).
- `-t`, `--tickers` - json containing listing of all tickers to download data for.
- `-s`, `--start-date` - First date to retrieve data for (yyyy-mm-dd).
- `-y`, `--years` - Number of years to download daily data over.

A json containing all S&P 500 tickers is available at `tickers/sp500.json` for ease of getting started.

The data directory obtained via either method is then suitable for `build_universe_for`.

---

### Build a universe

#### Entry point

```python
def build_universe_for(
    alphas_: Sequence[str | type[alphas.Alpha]],
    *,
    src: Path,
    path: Path,
    sector: Sector | None = None,
    industry: Industry | None = None,
    liquidity_threshold: float = 5e8,
    mcap_threshold: float = 5e6,
    lookback: Window = Window.MONTH,
) -> Universe:
```

#### Parameters

- `alphas_` : `Sequence[str | type[Alpha]]` - List of alphas wanting to be evaluated, given by either `ID` or type name.
Available options include:
    - `"reversal_1d"` or `Reversal1d`
    - `"close_z_score_12m"` or `CloseZScore12m`
    - `"volatility_20d"` or `Volatility20d`

    For a full list of available alphas along with information on creating custom alphas see [Alphas](#alphas).
- `src` : `Path` - Directory containing equity metadata and per-ticker market data, ideally created by `download`.
- `path` : `Path` - Directory to store memory-mapped arrays of market series (wiped if already exists).
- `sector` : `Sector`, optional - [Sector](#sectors-and-industries) to exclusively include tickers from.
- `industry` : `Industry`, optional - [Industry](#sectors-and-industries) to exclusively include tickers from (must
belong to `sector`, cannot be specified if `sector` isn't).
- `liquidity_threshold`: `float` - Required average liquidity (over `lookback`) for a stock to be considered
in-universe on a given date.
- `mcap_threshold` : `float` - Required average market cap (over `lookback`) for a stock to be considered in-universe
on a given date.
- `lookback` : `Window`, default `Window.MONTH` - Size of lookback [window](#window) used in calculation of in-universe mask.

#### Effect

Returns a universe instance containing all market series required to compute the listed alphas for stocks meeting the
specified sector and industry filtering and liquidity and market cap thresholds.

#### Example

```python
import alpha_research_framework as arf

alphas = [
    "reversal_1d",
    "close_z_score_12m",
    "volatility_20d",
]

universe = arf.build_universe_for(
    alphas,
    src=Path("data"),
    path=Path("universe"),
)
```

---

### Evaluate alphas

#### Entry point

```python
def evaluate(
    universe: Universe,
    alphas_: Sequence[str | type[alphas.Alpha]],
    metrics_: Sequence[str | type[metrics.Metric]] = ["information_coefficient"]
) -> dict[str | type[alphas.Alpha], pd.DataFrame]:
```

#### Parameters

- `universe` : `Universe` - Cross-sectional universe with pre-applied (optional) sector/industry, liquidity and
market-cap filtering.
- `alphas_` : `Sequence[str | type[Alpha]]` - List of alphas to evaluate, given by either `ID` or type name. Available
options include:
    - `"reversal_1d"` or `Reversal1d`
    - `"close_z_score_12m"` or `CloseZScore12m`
    - `"volatility_20d"` or `Volatility20d`

    For a full list of available alphas along with information on creating custom alphas see [Alphas](#alphas).
- `metrics_` : `Sequence[str | type[Metric]]`, default `["information_coefficient"]` - List of metrics to compute per
alpha at every `t`, given by either `ID` or type name. Available options are:
    - `"information_coefficient"` or `InformationCoefficient`
    - `"quantile_portfolio"` or `QuantilePorfolio`

#### Effect

Evaluates the cross-sectional predictive power of the given alphas. Returns a dictionary mapping each request alpha to a
DataFrame indexed by date, with columns per (horizon, metric) pair, containing a measure of the correlation between each
alpha's signal at time $t$ and the realised forward returns from $t$ to $t + horizon$.

#### Example

```python
result = arf.evaluate(universe, alphas)

for alpha in alphas:
    print(alpha, '\n', result[alpha].mean())
```

## Alphas

An `Alpha` represents a research hypothesis embodied by a signal. In this framework, the `Alpha` class pairs a `Signal` with
the metadata needed to make it a named research object.

### Built-in alphas

| Name | ID | Category | Signal | Horizons |
|------|----|----------|--------|----------|
| Reversal1d | "reversal_1d" | "short_term_reversal" | - returns_1d | 1d, 5d |
| Reversal5d | "reversal_5d" | "short_term_reversal" | - returns_5d | 1d, 5d |
| OvernightReversal | "overnight_reversal" | "short_term_reversal" | - (log_open - log_close_lag_1d) | 1d, 5d |
| IntradayReversal | "intraday_reversal" | "short_term_reversal" | - (log_close - log_open) | 1d, 5d |
| CloseLocation | "close_location" | "short_term_reversal" | - (2 * close - high - low) / (high - low) | 1d, 5d |
| Momentum20d | "momentum_20d" | "momentum" | returns_20d | 1d, 5d, 20d |
| Momentum12m | "momentum_12m" | "momentum" | returns_252d | 20d, 63d, 126d, 252d |
| Momentum12m1m | "momentum_12m_1m" | "momentum" | returns_252d - returns20d | 20d, 63d, 126d, 252d |
| RiskAdjustedMomentum12m1m | "risk_adjusted_momentum_12m_1m" | "momentum" | (returns_252d - returns_20d) / volatility_252d | 20d, 63d, 126d, 252d |
| RollingAverageDeviation20d | "rolling_average_deviation_20d" | "mean_reversion" | - (close - rolling_avg_close_20d) / rolling_avg_close_20d | 1d, 5d, 20d |
| RollingAverageDeviation12m | "rolling_average_deviation_12m" | "mean_reversion" | - (close - rolling_avg_close_252d) / rolling_avg_close_252d | 20d, 63d, 126d, 252d |
| RollingAverageTrendDeviation5d20d | "rolling_average_trend_deviation_5d_20d" | "mean_reversion" | - (rolling_avg_close_5d - rolling_avg_close_20d) / rolling_avg_close_20d | 1d, 5d, 20d |
| RollingAverageTrendDeviation1m12m | "rolling_average_trend_deviation_1m_12m" | "mean_reversion" | - (rolling_avg_close_20d - rolling_avg_close_252d) / rolling_avg_close_252d | 20d, 63d, 126d, 252d |
| CloseZScore20d | "close_z_score_20d" | "mean_reversion" | - (close - rolling_avg_close_20d) / rolling_std_close_20d | 1d, 5d, 20d |
| CloseZScore12m | "close_z_score_12m" | "mean_reversion" | - (close - rolling_avg_close_252d) / rolling_std_close_252d | 20d, 63d, 126d, 252d |
| VolumeSpike20d | "volume_spike_20d" | "volume" | volume / rolling_avg_volume_20d | 1d, 5d, 20d |
| VolumeSpike6m | "volume_spike_6m" | "volume" | volume / rolling_avg_volume_63d | 1d, 5d, 20d |
| VolumeSpike12m | "volume_spike_12m" | "volume" | volume / rolling_avg_volume_252d | 1d, 5d, 20d |
| VolumeMomentum20d | "volume_momentum_20d | "volume" | log_volume - log_volume_lag_20d | 1d, 5d, 20d |
| VolumeMomentum12m | "volume_momentum_12m | "volume" | log_volume - log_volume_lag_252d | 20d, 63d, 126d, 252d |
| Volatility20d | "volatility_20d" | "volatility" | - volatility_20d | 1d, 5d, 20d |
| Volatility12m | "volatility_12m" | "volatility" | - volatility_252d | 20d, 63d, 126d, 252d |
| RelativeVolatility1m12m | "relative_volatility_1m_12m" | "volatility" | volatility_20d / volatility_252d | 20d, 63d, 126d, 252d |

### Designing new alphas

A concrete `Alpha` must define:
- `ID` : `str` - Unique identifier.
- `CATEGORY` : `str` - Logical grouping label.
- `SIGNAL` : `type[signals.Signal]` - Cross-sectional signal; can be any built-in or custom signal, or combinations of
them via $+, -, \times, \div$. For a full list of available signals along with information on creating custom signals
see [Signals](#signals).
- `HORIZONS` : `set[Window]` - Prediction horizons to evaluate the alpha against.

As an example:

```python
import alpha_research_framework as arf

class DollarVolume(arf.alphas.Alpha):
    ID = "dollar_volume"
    CATEGORY = "liquidity"
    SIGNAL = arf.signals.Close * arf.signals.Volume
    HORIZONS = {arf.Window.DAY, arf.Window.WEEK}
```

## Signals

A `Signal` represents a one-dimensional cross-section: one scalar per in-universe ticker at a single date. They are
deliberately only explicitly defined for simple forwarding of `Series` objects to increase clarity in the `Alpha` layer.

### Built-in signals

| Observables | Log Observables | Lagged Log Observables | Returns | Rolling Averages | Rolling Standard Deviations |
|-------------|-----------------|------------------------|---------|------------------|-----------------------------|
| Open | LogOpen | LogCloseLag1d | Returns1d | RollingAvgClose5d | RollingStdClose5d |
| High | LogHigh | LogCloseLag5d | Returns5d | RollingAvgClose20d | RollingStdClose20d |
| Low | LogLow | LogCloseLag20d | Returns20d | RollingAvgClose63d | RollingStdClose63d |
| Close | LogClose | LogCloseLag63d | Returns63d | RollingAvgClose126d | RollingStdClose126d |
| Volume | LogVolume | LogCloseLag126d | Returns126d | RollingAvgClose252d | RollingStdClose252d|
| | | LogCloseLag252d | Returns252d | RollingAvgVolume5d | Volatility5d |
| | | LogVolumeLag1d | | RollingAvgVolume20d | Volatility20d |
| | | LogVolumeLag5d | | RollingAvgVolume63d | Volatility63d |
| | | LogVolumeLag20d | | RollingAvgVolume126d | Volatility126d |
| | | LogVolumeLag63d | | RollingAvgVolume252d | Volatility252d |
| | | LogVolumeLag126d | | | |
| | | LogVolumeLag252d | | | |

### Designing new signals

The definition of a new `Signal` should purely be for forwarding a new `Series`. Whilst more complicated ones may be
constructed (via combination through $+, -, \times, \div$ or purely negation) it is a design choice to delay doing so
until defining an `Alpha.SIGNAL`. One caveat to this would be for giving a name to a combination that is being used
frequently (note this does not improve computational efficiency due to caching within signal computation).

To forward a new `Series`, a new `Signal` need only define the `SERIES` attribute of a `SeriesSignal` subclass. For example:

```python
import alpha_research_framework as arf

class NewSignal(arf.signals.SeriesSignal):
    SERIES = NewSeries
```

## Series

A `Series` represents a quantity over the complete time × ticker market-data domain. This layer is where all temporal
logic belongs; if an alpha needs to know something about the history of a market series, that must first be expressed in
a `Series` and propagated through a `SeriesSignal` so that it may be used when defining an `Alpha.SIGNAL`.

### Built-in series

| Observables | Log Observables | Lagged Log Observables | Returns | Rolling Averages | Rolling Standard Deviations |
|-------------|-----------------|------------------------|---------|------------------|-----------------------------|
| Open | LogOpen | LogCloseLag1d | Returns1d | RollingAvgClose5d | RollingStdClose5d |
| High | LogHigh | LogCloseLag5d | Returns5d | RollingAvgClose20d | RollingStdClose20d |
| Low | LogLow | LogCloseLag20d | Returns20d | RollingAvgClose63d | RollingStdClose63d |
| Close | LogClose | LogCloseLag63d | Returns63d | RollingAvgClose126d | RollingStdClose126d |
| Volume | LogVolume | LogCloseLag126d | Returns126d | RollingAvgClose252d | RollingStdClose252d|
| | | LogCloseLag252d | Returns252d | RollingAvgVolume5d | Volatility5d |
| | | LogVolumeLag1d | | RollingAvgVolume20d | Volatility20d |
| | | LogVolumeLag5d | | RollingAvgVolume63d | Volatility63d |
| | | LogVolumeLag20d | | RollingAvgVolume126d | Volatility126d |
| | | LogVolumeLag63d | | RollingAvgVolume252d | Volatility252d |
| | | LogVolumeLag126d | | | |
| | | LogVolumeLag252d | | | |

### Designing new series

New `Series` may be created by either combined existing `Series` via $+, -, \times, \div$:

```python
import alpha_research_framework as arf

DollarVolume = arf.series.Close * arf.series.Volume
```

or [transforming](#built-in-transforms) existing `Series`:

```python
RollingAvgDollarVolume20d = arf.series.transform(
    DollarVolume,
    func=arf.series.transforms.rolling_avg,
    lookback=arf.Window.MONTH.value,
)
```

Arithmetic combination should be delayed to the alpha layer (i.e. forward both `Series` through `SeriesSignal` objects)
unless a temporal transform of the combination is required.

### Built-in transforms

| Transform | kwargs |
|-----------|--------|
| log       |        |
| shift_forward | period: int |
| shift_back | period: int |
| rolling_avg | lookback: int |
| rolling_std | lookback: int |

### Designing a new transform

A new transform:
- must have signature:

```python
new_transform(arr: npt.NDArray[np.floating], **kwargs) -> None
```

where `kwargs` are forwarded when passed to `arf.series.transform`.
- should only act over axis 0 (the time-axis)
- must modify `arr` purely in place
- should avoid materialising `arr` or any transient array of similar size

The source code for both `rolling_avg` and `rolling_std` give clear examples of this.

## Metrics

### Built-in metrics

| Name | ID| Measure |
| -----|---|---------|
| InformationCoefficient | "information_coefficient" | Spearman's rank correlation coefficient between signal and forward returns |
| QuantilePortfolio | "quantile_portfolio" | Average forward returns per decile in signal |

## Window

An `IntEnum` defining the number of trading days within common time-periods:

| Time-period | # trading-days |
|-------------|----------------|
| DAY | 1 |
| WEEK | 5 |
| MONTH | 20 |
| QUARTER | 63 |
| HALF_YEAR | 126 |
| YEAR | 252 |

## Sectors and Industries

Financial market sectors and industries as categorised by yfinance:

| Sector | Industry |
|--------|----------|
|  basic materials  |  agricultural inputs<br>aluminum<br>building materials<br>chemicals<br>coking coal<br>copper<br>gold<br>lumber & wood production<br>other industrial metals & mining<br>other precious metals & mining<br>paper & paper products<br>silver<br>specialty chemicals<br>steel  |
|  communication services  |  advertising agencies<br>broadcasting<br>electronic gaming & multimedia<br>entertainment<br>internet content & information<br>publishing<br>telecom services  |
|  consumer cyclical  |  apparel manufacturing<br>apparel retail<br>auto & truck dealerships<br>auto manufacturers<br>auto parts<br>department stores<br>footwear & accessories<br>furnishings, fixtures & appliances<br>gambling<br>home improvement retail<br>internet retail<br>leisure<br>lodging<br>luxury goods<br>packaging & containers<br>personal services<br>recreational vehicles<br>residential construction<br>resorts & casinos<br>restaurants<br>specialty retail<br>textile manufacturing<br>travel services  |
|  consumer defensive  |  beverages - brewers<br>beverages - non-alcoholic<br>beverages - wineries & distilleries<br>confectioners<br>discount stores<br>education & training services<br>farm products<br>food distribution<br>grocery stores<br>household & personal products<br>packaged foods<br>tobacco  |
|  energy  |  oil & gas drilling<br>oil & gas e&p<br>oil & gas equipment & services<br>oil & gas integrated<br>oil & gas midstream<br>oil & gas refining & marketing<br>thermal coal<br>uranium  |
|  financial services  |  asset management<br>banks - diversified<br>banks - regional<br>capital markets<br>credit services<br>financial conglomerates<br>financial data & stock exchanges<br>insurance - diversified<br>insurance - life<br>insurance - property & casualty<br>insurance - reinsurance<br>insurance - specialty<br>insurance brokers<br>mortgage finance<br>shell companies  |
|  healthcare  |  biotechnology<br>diagnostics & research<br>drug manufacturers - general<br>drug manufacturers - specialty & generic<br>health information services<br>healthcare plans<br>medical care facilities<br>medical devices<br>medical distribution<br>medical instruments & supplies<br>pharmaceutical retailers  |
|  industrials  |  aerospace & defense<br>airlines<br>airports & air services<br>building products & equipment<br>business equipment & supplies<br>conglomerates<br>consulting services<br>electrical equipment & parts<br>engineering & construction<br>farm & heavy construction machinery<br>industrial distribution<br>infrastructure operations<br>integrated freight & logistics<br>marine shipping<br>metal fabrication<br>pollution & treatment controls<br>railroads<br>rental & leasing services<br>security & protection services<br>specialty business services<br>specialty industrial machinery<br>staffing & employment services<br>tools & accessories<br>trucking<br>waste management  |
|  real estate  |  real estate - development<br>real estate - diversified<br>real estate services<br>reit - diversified<br>reit - healthcare facilities<br>reit - hotel & motel<br>reit - industrial<br>reit - mortgage<br>reit - office<br>reit - residential<br>reit - retail<br>reit - specialty  |
|  technology  |  communication equipment<br>computer hardware<br>consumer electronics<br>electronic components<br>electronics & computer distribution<br>information technology services<br>scientific & technical instruments<br>semiconductor equipment & materials<br>semiconductors<br>software - application<br>software - infrastructure<br>solar  |
|  utilities  |  utilities - diversified<br>utilities - independent power producers<br>utilities - regulated electric<br>utilities - regulated gas<br>utilities - regulated water<br>utilities - renewable  |

## Limitations

This project is intended to make research mechanics easy; it should not currently be interpreted as a production-quality
empirical research stack. In particular:

- **Survivorship bias**: The current downloader builds a static ticker universe from the tickers supplied to it and stores
current/static ticker metadata. It does not provide survivorship-bias-safe historical universe membership or delisting
history.

- **Corporate action**: The downloader uses adjusted close to derive adjusted OHL values but volume is not adjusted in
the same way. This means its consistent throughout dividends but not stock splits.

## Future Improvements

- **Cross-sectional transformations**:  Signals currently provide arithmetic composition but do not yet provide
first-class operators for common cross-sectional transformations such as rank, z-score, winsorisation or neutralisation.

- **Alpha combination**: There is currently no higher-level framework abstraction for combining or optimising multiple
alphas into a composite signal.