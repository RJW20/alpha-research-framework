# ruff: noqa: E501
import alpha_research_framework.signals as signals
from alpha_research_framework.window import Window

from .alpha import Alpha

# -------------------------------------------- Notes -----------------------------------------------
# Contained here is a full definition of all built-in Alphas that this library
# provides.
# For information on creating custom Alphas see the "Alphas" section of the
# documentation.
# -------------------------------------------- Notes -----------------------------------------------

# --------------------------------------------------------------------------------------------------
# Short Term Reversal
# Hypothesis: large short-term moves are often an overreaction
# --------------------------------------------------------------------------------------------------

class Reversal1d(Alpha):
    """
    - Signal: `-returns_1d`
    - Horizons: `1d`, `5d`
    """
    ID = "reversal_1d"
    CATEGORY = "short_term_reversal"
    SIGNAL = -signals.Returns1d
    HORIZONS = {Window.DAY, Window.WEEK}

class Reversal5d(Alpha):
    """
    - Signal: `-returns_5d`
    - Horizons: `1d`, `5d`
    """
    ID = "reversal_5d"
    CATEGORY = "short_term_reversal"
    SIGNAL = -signals.Returns5d
    HORIZONS = {Window.DAY, Window.WEEK}

class OvernightReversal(Alpha):
    """
    - Signal: `- (log_open - log_close_lag_1d)`
    - Horizons: `1d`, `5d`
    """
    ID = "overnight_reversal"
    CATEGORY = "short_term_reversal"
    SIGNAL = - (signals.LogOpen - signals.LogCloseLag1d)
    HORIZONS = {Window.DAY, Window.WEEK}

class IntradayReversal(Alpha):
    """
    - Signal: `- (log_close - log_open)`
    - Horizons: `1d`, `5d`
    """
    ID = "intraday_reversal"
    CATEGORY = "short_term_reversal"
    SIGNAL = - (signals.LogClose - signals.LogOpen)
    HORIZONS = {Window.DAY, Window.WEEK}

class CloseLocation(Alpha):
    """
    - Signal: - (2 * close - high - low) / (high - low)
    - Horizons: `1d`, `5d`
    """
    ID = "close_location"
    CATEGORY = "short_term_reversal"
    SIGNAL = - (signals.Close + signals.Close - signals.High - signals.Low) / (signals.High - signals.Low)
    HORIZONS = {Window.DAY, Window.WEEK}
    
# --------------------------------------------------------------------------------------------------
# Momentum
# Hypothesis: price trends persist
# --------------------------------------------------------------------------------------------------

class Momentum20d(Alpha):
    """
    - Signal: `returns_20d`
    - Horizons: `1d`, `5d`, `20d`
    """
    ID = "momentum_20d"
    CATEGORY = "momentum"
    SIGNAL = signals.Returns20d
    HORIZONS = {Window.DAY, Window.WEEK, Window.MONTH}

class Momentum12m(Alpha):
    """
    - Signal: `returns_252d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "momentum_12m"
    CATEGORY = "momentum"
    SIGNAL = signals.Returns252d
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

class Momentum12m1m(Alpha):
    """
    - Signal: `returns_252d - returns_20d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "momentum_12m_1m"
    CATEGORY = "momentum"
    SIGNAL = signals.Returns252d - signals.Returns20d
    SKIP = Window.MONTH
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

class RiskAdjustedMomentum12m1m(Alpha):
    """
    - Signal: `(returns_252d - returns_20d) / volatility_252d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "risk_adjusted_momentum_12m_1m"
    CATEGORY = "momentum"
    SIGNAL = (signals.Returns252d - signals.Returns20d) / signals.Volatility252d
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

# --------------------------------------------------------------------------------------------------
# Mean Reversion
# Hypothesis: price returns to average
# --------------------------------------------------------------------------------------------------

class RollingAverageDeviation20d(Alpha):
    """
    - Signal: `- (close - rolling_avg_close_20d) / rolling_avg_close_20d`
    - Horizons: `1d`, `5d`, `20d`
    """
    ID = "rolling_average_deviation_20d"
    CATEGORY = "mean_reversion"
    SIGNAL = - (signals.Close - signals.RollingAvgClose20d) / signals.RollingAvgClose20d
    HORIZONS = {Window.DAY, Window.WEEK, Window.MONTH}

class RollingAverageDeviation12m(Alpha):
    """
    - Signal: `- (close - rolling_avg_close_252d) / rolling_avg_close_252d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "rolling_average_deviation_12m"
    CATEGORY = "mean_reversion"
    SIGNAL = - (signals.Close - signals.RollingAvgClose252d) / signals.RollingAvgClose252d
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

class RollingAverageTrendDeviation5d20d(Alpha):
    """
    - Signal: `- (rolling_avg_close_5d - rolling_avg_close_20d) / rolling_avg_close_20d`
    - Horizons: `1d`, `5d`, `20d`
    """
    ID = "rolling_average_trend_deviation_5d_20d"
    CATEGORY = "momentum"
    SIGNAL = - (signals.RollingAvgClose5d - signals.RollingAvgClose20d) / signals.RollingAvgClose20d
    HORIZONS = {Window.DAY, Window.WEEK, Window.MONTH}

class RollingAverageTrendDeviation1m12m(Alpha):
    """
    - Signal: `- (rolling_avg_close_20d - rolling_avg_close_252d) / rolling_avg_close_252d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "rolling_average_trend_deviation_1m_12m"
    CATEGORY = "momentum"
    SIGNAL = - (signals.RollingAvgClose20d - signals.RollingAvgClose252d) / signals.RollingAvgClose252d
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

class CloseZScore20d(Alpha):
    """
    - Signal: `- (close - rolling_avg_close_20d) / rolling_std_close_20d`
    - Horizons: `1d`, `5d`, `20d`
    """
    ID = "close_z_score_20d"
    CATEGORY = "mean_reversion"
    SIGNAL = - (signals.Close - signals.RollingAvgClose20d) / signals.RollingStdClose20d
    HORIZONS = {Window.DAY, Window.WEEK, Window.MONTH}

class CloseZScore12m(Alpha):
    """
    - Signal: `- (close - rolling_avg_close_252d) / rolling_std_close_252d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "close_z_score_12m"
    CATEGORY = "mean_reversion"
    SIGNAL = - (signals.Close - signals.RollingAvgClose252d) / signals.RollingStdClose252d
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

# --------------------------------------------------------------------------------------------------
# Volume
# Hypothesis: volume spike before price spike
# --------------------------------------------------------------------------------------------------

class VolumeSpike20d(Alpha):
    """
    - Signal: `volume / rolling_avg_volume_20d`
    - Horizons: `1d`, `5d`, `20d`
    """
    ID = "volume_spike_20d"
    CATEGORY = "volume"
    SIGNAL = signals.Volume / signals.RollingAvgVolume20d
    HORIZONS = {Window.DAY, Window.WEEK, Window.MONTH}

class VolumeSpike6m(Alpha):
    """
    - Signal: `volume / rolling_avg_volume_63d`
    - Horizons: `1d`, `5d`, `20d`
    """
    ID = "volume_spike_6m"
    CATEGORY = "volume"
    SIGNAL = signals.Volume / signals.RollingAvgVolume63d
    HORIZONS = {Window.DAY, Window.WEEK, Window.MONTH}

class VolumeSpike12m(Alpha):
    """
    - Signal: `volume / rolling_avg_volume_252d`
    - Horizons: `1d`, `5d`, `20d`
    """
    ID = "volume_spike_12m"
    CATEGORY = "volume"
    SIGNAL = signals.Volume / signals.RollingAvgVolume252d
    HORIZONS = {Window.DAY, Window.WEEK, Window.MONTH}

class VolumeMomentum20d(Alpha):
    """
    - Signal: `log_volume - log_volume_lag_20d`
    - Horizons: `1d`, `5d`, `20d`
    """
    ID = "volume_momentum_20d"
    CATEGORY = "volume"
    SIGNAL = signals.LogVolume - signals.LogVolumeLag20d
    HORIZONS = {Window.DAY, Window.WEEK, Window.MONTH}

class VolumeMomentum12m(Alpha):
    """
    - Signal: `log_volume - log_volume_lag_252d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "volume_momentum_12m"
    CATEGORY = "volume"
    SIGNAL = signals.LogVolume - signals.LogVolumeLag252d
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

# --------------------------------------------------------------------------------------------------
# Volatility
# Hypothesis: market overprices volatility
# --------------------------------------------------------------------------------------------------

class Volatility20d(Alpha):
    """
    - Signal: `-volatility_20d`
    - Horizons: `1d`, `5d`, `20d`
    """
    ID = "volatility_20d"
    CATEGORY = "volatility"
    SIGNAL = -signals.Volatility20d
    HORIZONS = {Window.DAY, Window.WEEK, Window.MONTH}

class Volatility12m(Alpha):
    """
    - Signal: `-volatility_252d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "volatility_12m"
    CATEGORY = "volatility"
    SIGNAL = -signals.Volatility252d
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}

class RelativeVolatility1m12m(Alpha):
    """
    - Signal: `volatility_20d / volatility_252d`
    - Horizons: `20d`, `63d`, `126d`, `252d`
    """
    ID = "relative_volatility_1m_12m"
    CATEGORY = "volatility"
    SIGNAL = signals.Volatility20d / signals.RollingAvgClose252d
    HORIZONS = {Window.MONTH, Window.QUARTER, Window.HALF_YEAR, Window.YEAR}
