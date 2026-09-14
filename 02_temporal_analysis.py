"""
02_temporal_analysis.py

Temporal analysis of ERA5 city-level temperature metrics.

INPUT
-----
city_year_metrics_1950_2025.csv

OUTPUT
------
CSV tables + PNG figures only.

No parquet / pyarrow / fastparquet required.

Analyses
--------
1. City-specific temporal trends
   - Sen's slope
   - Hamed-Rao modified Mann-Kendall
   - fallback Kendall tau when Hamed-Rao is unstable
   - Benjamini-Hochberg FDR correction

2. Early vs recent period comparison
   - nominal early period  : 1951-1980
   - nominal recent period : 1996-2025
   - automatically uses only years actually present

3. National city-distribution temporal evolution
   - yearly median across cities
   - P25-P75 range

4. Figures
   - major temperature trends
   - city warming-rate distributions
   - cold-extreme trends
   - hot-extreme trends
   - cold vs hot change relationship
   - percentage of cities with significant trends
   - early vs recent changes

IMPORTANT
---------
"National" curves here are MEDIANS ACROSS CITIES.

They are NOT area-weighted national temperature means.
"""

from __future__ import annotations

import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

try:
    import pymannkendall as mk
    HAS_PYMK = True
except ImportError:
    HAS_PYMK = False


# =============================================================================
# SETTINGS
# =============================================================================

START_YEAR = 1950
END_YEAR = 2025

EARLY_START = 1951
EARLY_END = 1980

RECENT_START = 1996
RECENT_END = 2025

ALPHA = 0.05

# Minimum number of valid annual observations required for trend analysis
MIN_YEARS = 30


DEFAULT_INPUT = pathlib.Path(
    r"C:\Users\joyal\OneDrive\Desktop\oscar-team"
    r"\mask_era5_china"
    r"\results_city_year_metrics_1950_2025"
    r"\city_year_metrics_1950_2025.csv"
)

DEFAULT_OUTPUT = pathlib.Path(
    r"C:\Users\joyal\OneDrive\Desktop\oscar-team"
    r"\mask_era5_china"
    r"\results_temporal_analysis_1950_2025"
)


# =============================================================================
# METRICS
# =============================================================================

METRICS = {

    # -------------------------------------------------------------------------
    # Temperature
    # -------------------------------------------------------------------------

    "annual_tmean": {
        "label": "Annual mean temperature",
        "unit": "°C",
        "trend_unit": "°C/decade",
        "family": "temperature",
    },

    "annual_tmin": {
        "label": "Annual mean Tmin",
        "unit": "°C",
        "trend_unit": "°C/decade",
        "family": "temperature",
    },

    "annual_tmax": {
        "label": "Annual mean Tmax",
        "unit": "°C",
        "trend_unit": "°C/decade",
        "family": "temperature",
    },

    "djf_tmin": {
        "label": "Winter (DJF) mean Tmin",
        "unit": "°C",
        "trend_unit": "°C/decade",
        "family": "temperature",
    },

    "jja_tmax": {
        "label": "Summer (JJA) mean Tmax",
        "unit": "°C",
        "trend_unit": "°C/decade",
        "family": "temperature",
    },

    "txx": {
        "label": "Annual maximum Tmax (TXx)",
        "unit": "°C",
        "trend_unit": "°C/decade",
        "family": "temperature",
    },

    "tnn": {
        "label": "Annual minimum Tmin (TNn)",
        "unit": "°C",
        "trend_unit": "°C/decade",
        "family": "temperature",
    },

    # -------------------------------------------------------------------------
    # Cold extremes
    # -------------------------------------------------------------------------

    "fd0_pct": {
        "label": "Days with Tmin < 0°C",
        "unit": "%",
        "trend_unit": "percentage points/decade",
        "family": "cold",
    },

    "fd5_pct": {
        "label": "Days with Tmin < -5°C",
        "unit": "%",
        "trend_unit": "percentage points/decade",
        "family": "cold",
    },

    "fd10_pct": {
        "label": "Days with Tmin < -10°C",
        "unit": "%",
        "trend_unit": "percentage points/decade",
        "family": "cold",
    },

    # -------------------------------------------------------------------------
    # Absolute hot extremes
    # -------------------------------------------------------------------------

    "hd35_pct": {
        "label": "Days with Tmax > 35°C",
        "unit": "%",
        "trend_unit": "percentage points/decade",
        "family": "hot",
    },

    "hd40_pct": {
        "label": "Days with Tmax > 40°C",
        "unit": "%",
        "trend_unit": "percentage points/decade",
        "family": "hot",
    },

    # -------------------------------------------------------------------------
    # Relative hot extremes
    # -------------------------------------------------------------------------

    "tx95p_pct": {
        "label": "Days with Tmax > city P95",
        "unit": "%",
        "trend_unit": "percentage points/decade",
        "family": "hot",
    },

    "tx99p_pct": {
        "label": "Days with Tmax > city P99",
        "unit": "%",
        "trend_unit": "percentage points/decade",
        "family": "hot",
    },

    # -------------------------------------------------------------------------
    # Relative cold extremes
    # -------------------------------------------------------------------------

    "tn05p_pct": {
        "label": "Days with Tmin < city P05",
        "unit": "%",
        "trend_unit": "percentage points/decade",
        "family": "cold",
    },

    "tn01p_pct": {
        "label": "Days with Tmin < city P01",
        "unit": "%",
        "trend_unit": "percentage points/decade",
        "family": "cold",
    },
}


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data(path: pathlib.Path) -> pd.DataFrame:

    print("\n" + "=" * 80)
    print("LOADING CITY-YEAR METRICS")
    print("=" * 80)

    if not path.exists():

        raise FileNotFoundError(
            f"Input does not exist:\n{path}"
        )

    if path.suffix.lower() != ".csv":

        raise ValueError(
            "This revised script expects a CSV input."
        )

    df = pd.read_csv(path)

    required = {
        "city_id",
        "city_name",
        "year",
    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["year"]
    ).copy()

    df["year"] = df["year"].astype(int)

    df = df.loc[
        df["year"].between(
            START_YEAR,
            END_YEAR,
        )
    ].copy()

    if df.empty:

        raise ValueError(
            "No rows remain after year filtering."
        )

    print(
        f"Rows   : {len(df):,}"
    )

    print(
        f"Cities : {df['city_id'].nunique():,}"
    )

    print(
        f"Years  : "
        f"{df['year'].min()}-"
        f"{df['year'].max()}"
    )

    available_metrics = [
        metric
        for metric in METRICS
        if metric in df.columns
    ]

    print(
        f"Available analysis metrics: "
        f"{len(available_metrics)}"
    )

    return df


# =============================================================================
# BENJAMINI-HOCHBERG FDR
# =============================================================================

def benjamini_hochberg(pvalues):

    p = np.asarray(
        pvalues,
        dtype=float,
    )

    result = np.full(
        len(p),
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(p)

    pv = p[valid]

    if len(pv) == 0:
        return result

    order = np.argsort(pv)

    ranked = pv[order]

    n = len(ranked)

    adjusted_sorted = (
        ranked
        * n
        /
        np.arange(
            1,
            n + 1,
        )
    )

    adjusted_sorted = (
        np.minimum.accumulate(
            adjusted_sorted[::-1]
        )[::-1]
    )

    adjusted_sorted = np.clip(
        adjusted_sorted,
        0,
        1,
    )

    adjusted_original_order = (
        np.empty_like(
            adjusted_sorted
        )
    )

    adjusted_original_order[
        order
    ] = adjusted_sorted

    result[
        valid
    ] = adjusted_original_order

    return result


# =============================================================================
# MANN-KENDALL
# =============================================================================

def fallback_kendall(values):

    y = np.asarray(
        values,
        dtype=float,
    )

    x = np.arange(
        len(y),
        dtype=float,
    )

    tau, p = stats.kendalltau(
        x,
        y,
        nan_policy="omit",
    )

    if not np.isfinite(tau):
        tau = 0.0

    if not np.isfinite(p):
        p = 1.0

    if p < ALPHA and tau > 0:
        trend = "increasing"

    elif p < ALPHA and tau < 0:
        trend = "decreasing"

    else:
        trend = "no trend"

    return {
        "mk_method":
            "standard Kendall tau fallback",

        "mk_tau":
            float(tau),

        "mk_p":
            float(p),

        "mk_z":
            np.nan,

        "mk_trend":
            trend,
    }


def mann_kendall_test(values):

    y = np.asarray(
        values,
        dtype=float,
    )

    y = y[
        np.isfinite(y)
    ]

    if len(y) < 3:

        return {
            "mk_method":
                "insufficient data",

            "mk_tau":
                np.nan,

            "mk_p":
                np.nan,

            "mk_z":
                np.nan,

            "mk_trend":
                "unknown",
        }

    unique_values = np.unique(
        y
    )

    # -------------------------------------------------------------------------
    # Fully constant series
    # -------------------------------------------------------------------------

    if len(unique_values) == 1:

        return {
            "mk_method":
                "constant series",

            "mk_tau":
                0.0,

            "mk_p":
                1.0,

            "mk_z":
                0.0,

            "mk_trend":
                "no trend",
        }

    # -------------------------------------------------------------------------
    # Very highly tied series
    #
    # Example:
    # HD40 = 0 for almost every year.
    #
    # Hamed-Rao variance adjustment can become numerically unstable.
    # -------------------------------------------------------------------------

    dominant_fraction = (
        pd.Series(y)
        .value_counts(
            normalize=True
        )
        .iloc[0]
    )

    if dominant_fraction >= 0.90:

        return fallback_kendall(
            y
        )

    # -------------------------------------------------------------------------
    # Try Hamed-Rao modified MK
    # -------------------------------------------------------------------------

    if HAS_PYMK:

        try:

            with warnings.catch_warnings():

                # Convert RuntimeWarning into exception
                # so invalid sqrt does not flood the terminal.
                warnings.simplefilter(
                    "error",
                    RuntimeWarning,
                )

                result = (
                    mk.hamed_rao_modification_test(
                        y
                    )
                )

            tau = float(
                result.Tau
            )

            p = float(
                result.p
            )

            z = float(
                result.z
            )

            # Reject numerically invalid results
            if (
                not np.isfinite(tau)
                or
                not np.isfinite(p)
                or
                not np.isfinite(z)
            ):

                raise ValueError(
                    "Non-finite Hamed-Rao output"
                )

            return {
                "mk_method":
                    "Hamed-Rao modified MK",

                "mk_tau":
                    tau,

                "mk_p":
                    p,

                "mk_z":
                    z,

                "mk_trend":
                    str(result.trend),
            }

        except Exception:

            # Silent fallback
            return fallback_kendall(
                y
            )

    return fallback_kendall(
        y
    )


# =============================================================================
# SEN'S SLOPE
# =============================================================================

def sen_slope(
    years,
    values,
):

    x = np.asarray(
        years,
        dtype=float,
    )

    y = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(y) < 3:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    if len(np.unique(y)) == 1:

        return (
            0.0,
            0.0,
            0.0,
        )

    try:

        result = stats.theilslopes(
            y,
            x,
            alpha=0.95,
        )

        return (
            float(result.slope),
            float(result.low_slope),
            float(result.high_slope),
        )

    except Exception:

        return (
            np.nan,
            np.nan,
            np.nan,
        )


# =============================================================================
# LINEAR TREND
# =============================================================================

def linear_trend(
    years,
    values,
):

    x = np.asarray(
        years,
        dtype=float,
    )

    y = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(y) < 3:

        return {
            "slope":
                np.nan,

            "p":
                np.nan,

            "r":
                np.nan,
        }

    if len(np.unique(y)) == 1:

        return {
            "slope":
                0.0,

            "p":
                1.0,

            "r":
                0.0,
        }

    try:

        result = stats.linregress(
            x,
            y,
        )

        return {
            "slope":
                float(result.slope),

            "p":
                float(result.pvalue),

            "r":
                float(result.rvalue),
        }

    except Exception:

        return {
            "slope":
                np.nan,

            "p":
                np.nan,

            "r":
                np.nan,
        }


# =============================================================================
# ONE CITY × ONE METRIC
# =============================================================================

def calculate_one_trend(
    group: pd.DataFrame,
    metric: str,
):

    subset = (
        group[
            [
                "year",
                metric,
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .sort_values(
            "year"
        )
    )

    # DJF 1950 should not be used when Dec 1949 is unavailable.
    if metric.startswith(
        "djf_"
    ):

        subset = subset.loc[
            subset["year"] >= 1951
        ]

    if len(subset) < MIN_YEARS:

        return None

    years = subset[
        "year"
    ].to_numpy(
        dtype=float
    )

    values = subset[
        metric
    ].to_numpy(
        dtype=float
    )

    sen, sen_low, sen_high = (
        sen_slope(
            years,
            values,
        )
    )

    mk_result = (
        mann_kendall_test(
            values
        )
    )

    linear = (
        linear_trend(
            years,
            values,
        )
    )

    return {

        "n_years":
            int(len(values)),

        "first_year":
            int(
                years.min()
            ),

        "last_year":
            int(
                years.max()
            ),

        "mean":
            float(
                np.mean(values)
            ),

        "median":
            float(
                np.median(values)
            ),

        # Sen slope
        "sen_slope_per_year":
            sen,

        "sen_slope_per_decade":
            sen * 10,

        "sen_ci_low_per_decade":
            sen_low * 10,

        "sen_ci_high_per_decade":
            sen_high * 10,

        # MK
        **mk_result,

        # OLS
        "linear_slope_per_year":
            linear["slope"],

        "linear_slope_per_decade":
            (
                linear["slope"]
                * 10
            ),

        "linear_p":
            linear["p"],

        "linear_r":
            linear["r"],
    }


# =============================================================================
# ALL CITY TRENDS
# =============================================================================

def calculate_city_trends(
    df: pd.DataFrame,
):

    print("\n" + "=" * 80)
    print("CALCULATING CITY-SPECIFIC TEMPORAL TRENDS")
    print("=" * 80)

    available_metrics = [
        metric
        for metric in METRICS
        if metric in df.columns
    ]

    print(
        f"Metrics analysed: "
        f"{len(available_metrics)}"
    )

    rows = []

    grouped = list(
        df.groupby(
            "city_id",
            sort=False,
        )
    )

    n_cities = len(
        grouped
    )

    for number, (
        city_id,
        group,
    ) in enumerate(
        grouped,
        start=1,
    ):

        city_name = str(
            group[
                "city_name"
            ].iloc[0]
        )

        for metric in available_metrics:

            result = (
                calculate_one_trend(
                    group,
                    metric,
                )
            )

            if result is None:
                continue

            rows.append(
                {
                    "city_id":
                        city_id,

                    "city_name":
                        city_name,

                    "metric":
                        metric,

                    "metric_label":
                        METRICS[
                            metric
                        ]["label"],

                    "family":
                        METRICS[
                            metric
                        ]["family"],

                    "trend_unit":
                        METRICS[
                            metric
                        ]["trend_unit"],

                    **result,
                }
            )

        if (
            number % 100 == 0
            or
            number == n_cities
        ):

            print(
                f"Cities processed: "
                f"{number}/{n_cities}"
            )

    trends = pd.DataFrame(
        rows
    )

    if trends.empty:

        raise ValueError(
            "No valid temporal trends were calculated."
        )

    # -------------------------------------------------------------------------
    # FDR correction separately by metric
    # -------------------------------------------------------------------------

    trends[
        "mk_p_fdr"
    ] = np.nan

    for metric in trends[
        "metric"
    ].unique():

        mask = (
            trends[
                "metric"
            ]
            == metric
        )

        trends.loc[
            mask,
            "mk_p_fdr",
        ] = benjamini_hochberg(
            trends.loc[
                mask,
                "mk_p",
            ].to_numpy()
        )

    trends[
        "significant_fdr"
    ] = (
        trends[
            "mk_p_fdr"
        ]
        < ALPHA
    )

    trends[
        "trend_class"
    ] = "not significant"

    trends.loc[
        (
            trends[
                "significant_fdr"
            ]
            &
            (
                trends[
                    "sen_slope_per_decade"
                ]
                > 0
            )
        ),
        "trend_class",
    ] = "significant increase"

    trends.loc[
        (
            trends[
                "significant_fdr"
            ]
            &
            (
                trends[
                    "sen_slope_per_decade"
                ]
                < 0
            )
        ),
        "trend_class",
    ] = "significant decrease"

    return trends


# =============================================================================
# EARLY VS RECENT PERIOD
# =============================================================================

def period_comparison(
    df: pd.DataFrame,
):

    print("\n" + "=" * 80)
    print("EARLY VS RECENT PERIOD COMPARISON")
    print("=" * 80)

    actual_min = int(
        df["year"].min()
    )

    actual_max = int(
        df["year"].max()
    )

    early_actual_start = max(
        EARLY_START,
        actual_min,
    )

    early_actual_end = min(
        EARLY_END,
        actual_max,
    )

    recent_actual_start = max(
        RECENT_START,
        actual_min,
    )

    recent_actual_end = min(
        RECENT_END,
        actual_max,
    )

    print(
        f"Requested early period : "
        f"{EARLY_START}-{EARLY_END}"
    )

    print(
        f"Actual early period    : "
        f"{early_actual_start}-{early_actual_end}"
    )

    print(
        f"Requested recent period: "
        f"{RECENT_START}-{RECENT_END}"
    )

    print(
        f"Actual recent period   : "
        f"{recent_actual_start}-{recent_actual_end}"
    )

    available_metrics = [
        metric
        for metric in METRICS
        if metric in df.columns
    ]

    rows = []

    for city_id, group in df.groupby(
        "city_id"
    ):

        city_name = str(
            group[
                "city_name"
            ].iloc[0]
        )

        for metric in available_metrics:

            early = (
                group.loc[
                    group[
                        "year"
                    ].between(
                        early_actual_start,
                        early_actual_end,
                    ),
                    metric,
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )

            recent = (
                group.loc[
                    group[
                        "year"
                    ].between(
                        recent_actual_start,
                        recent_actual_end,
                    ),
                    metric,
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )

            if (
                len(early) < 10
                or
                len(recent) < 10
            ):

                continue

            early_mean = float(
                np.mean(
                    early
                )
            )

            recent_mean = float(
                np.mean(
                    recent
                )
            )

            early_median = float(
                np.median(
                    early
                )
            )

            recent_median = float(
                np.median(
                    recent
                )
            )

            try:

                test = stats.mannwhitneyu(
                    recent,
                    early,
                    alternative="two-sided",
                )

                p = float(
                    test.pvalue
                )

            except Exception:

                p = np.nan

            rows.append(
                {
                    "city_id":
                        city_id,

                    "city_name":
                        city_name,

                    "metric":
                        metric,

                    "metric_label":
                        METRICS[
                            metric
                        ]["label"],

                    "early_start":
                        early_actual_start,

                    "early_end":
                        early_actual_end,

                    "recent_start":
                        recent_actual_start,

                    "recent_end":
                        recent_actual_end,

                    "early_n":
                        len(early),

                    "recent_n":
                        len(recent),

                    "early_mean":
                        early_mean,

                    "recent_mean":
                        recent_mean,

                    "mean_change":
                        (
                            recent_mean
                            -
                            early_mean
                        ),

                    "early_median":
                        early_median,

                    "recent_median":
                        recent_median,

                    "median_change":
                        (
                            recent_median
                            -
                            early_median
                        ),

                    "mannwhitney_p":
                        p,
                }
            )

    result = pd.DataFrame(
        rows
    )

    result[
        "mannwhitney_p_fdr"
    ] = np.nan

    for metric in result[
        "metric"
    ].unique():

        mask = (
            result[
                "metric"
            ]
            == metric
        )

        result.loc[
            mask,
            "mannwhitney_p_fdr",
        ] = benjamini_hochberg(
            result.loc[
                mask,
                "mannwhitney_p",
            ].to_numpy()
        )

    return result


# =============================================================================
# YEARLY CITY SUMMARY
# =============================================================================

def yearly_city_summary(
    df: pd.DataFrame,
):

    rows = []

    available_metrics = [
        metric
        for metric in METRICS
        if metric in df.columns
    ]

    for year, group in df.groupby(
        "year"
    ):

        for metric in available_metrics:

            x = (
                group[
                    metric
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
            )

            if len(x) == 0:
                continue

            rows.append(
                {
                    "year":
                        int(year),

                    "metric":
                        metric,

                    "metric_label":
                        METRICS[
                            metric
                        ]["label"],

                    "n_cities":
                        int(len(x)),

                    "median":
                        float(
                            x.median()
                        ),

                    "p25":
                        float(
                            x.quantile(
                                0.25
                            )
                        ),

                    "p75":
                        float(
                            x.quantile(
                                0.75
                            )
                        ),

                    "mean":
                        float(
                            x.mean()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# TREND OF YEARLY CITY MEDIAN
# =============================================================================

def national_median_trends(
    summary: pd.DataFrame,
):

    rows = []

    for metric, group in summary.groupby(
        "metric"
    ):

        group = group.sort_values(
            "year"
        )

        if metric.startswith(
            "djf_"
        ):

            group = group.loc[
                group[
                    "year"
                ]
                >= 1951
            ]

        years = group[
            "year"
        ].to_numpy(
            dtype=float
        )

        values = group[
            "median"
        ].to_numpy(
            dtype=float
        )

        if len(values) < MIN_YEARS:
            continue

        sen, low, high = (
            sen_slope(
                years,
                values,
            )
        )

        mk_result = (
            mann_kendall_test(
                values
            )
        )

        rows.append(
            {
                "metric":
                    metric,

                "metric_label":
                    METRICS[
                        metric
                    ]["label"],

                "n_years":
                    len(values),

                "first_year":
                    int(
                        years.min()
                    ),

                "last_year":
                    int(
                        years.max()
                    ),

                "sen_slope_per_decade":
                    sen * 10,

                "sen_ci_low_per_decade":
                    low * 10,

                "sen_ci_high_per_decade":
                    high * 10,

                **mk_result,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# BUILD WIDE CITY TABLE
# =============================================================================

def build_wide_city_trend_table(
    trends: pd.DataFrame,
):

    metadata = (
        trends[
            [
                "city_id",
                "city_name",
            ]
        ]
        .drop_duplicates()
        .set_index(
            "city_id"
        )
    )

    slope = trends.pivot(
        index="city_id",
        columns="metric",
        values="sen_slope_per_decade",
    )

    slope.columns = [
        f"{x}_slope_decade"
        for x in slope.columns
    ]

    p_fdr = trends.pivot(
        index="city_id",
        columns="metric",
        values="mk_p_fdr",
    )

    p_fdr.columns = [
        f"{x}_p_fdr"
        for x in p_fdr.columns
    ]

    significant = trends.pivot(
        index="city_id",
        columns="metric",
        values="significant_fdr",
    )

    significant.columns = [
        f"{x}_significant"
        for x in significant.columns
    ]

    trend_class = trends.pivot(
        index="city_id",
        columns="metric",
        values="trend_class",
    )

    trend_class.columns = [
        f"{x}_trend_class"
        for x in trend_class.columns
    ]

    result = (
        metadata
        .join(
            slope
        )
        .join(
            p_fdr
        )
        .join(
            significant
        )
        .join(
            trend_class
        )
        .reset_index()
    )

    return result


# =============================================================================
# FIGURE 1
# =============================================================================

def plot_major_temperature_trends(
    summary,
    national_trends,
    output_dir,
):

    metrics = [
        "annual_tmean",
        "djf_tmin",
        "jja_tmax",
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(
            11,
            10,
        ),
        sharex=True,
    )

    for ax, metric in zip(
        axes,
        metrics,
    ):

        data = (
            summary.loc[
                summary[
                    "metric"
                ]
                == metric
            ]
            .sort_values(
                "year"
            )
            .copy()
        )

        if data.empty:
            continue

        ax.plot(
            data["year"],
            data["median"],
            linewidth=1.7,
        )

        ax.fill_between(
            data["year"],
            data["p25"],
            data["p75"],
            alpha=0.18,
        )

        trend_row = (
            national_trends.loc[
                national_trends[
                    "metric"
                ]
                == metric
            ]
        )

        if not trend_row.empty:

            slope_decade = float(
                trend_row[
                    "sen_slope_per_decade"
                ].iloc[0]
            )

            slope_year = (
                slope_decade
                / 10
            )

            x = data[
                "year"
            ].to_numpy(
                dtype=float
            )

            y = data[
                "median"
            ].to_numpy(
                dtype=float
            )

            x0 = np.median(
                x
            )

            y0 = np.median(
                y
            )

            fitted = (
                y0
                +
                slope_year
                *
                (
                    x
                    -
                    x0
                )
            )

            ax.plot(
                x,
                fitted,
                linestyle="--",
                linewidth=1.4,
            )

            p = float(
                trend_row[
                    "mk_p"
                ].iloc[0]
            )

            ax.text(
                0.02,
                0.93,
                (
                    f"Sen slope = "
                    f"{slope_decade:+.3f} "
                    f"°C/decade\n"
                    f"MK p = {p:.3g}"
                ),
                transform=ax.transAxes,
                va="top",
                fontsize=9,
            )

        ax.set_ylabel(
            "Temperature (°C)"
        )

        ax.set_title(
            METRICS[
                metric
            ]["label"],
            loc="left",
        )

        ax.grid(
            alpha=0.2
        )

    axes[-1].set_xlabel(
        "Year"
    )

    fig.suptitle(
        "Temporal evolution of city temperature conditions",
        fontsize=14,
    )

    fig.text(
        0.5,
        0.01,
        (
            "Solid line = median across cities; "
            "shading = P25–P75; "
            "dashed line = Sen trend"
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=[
            0,
            0.03,
            1,
            0.97,
        ]
    )

    fig.savefig(
        output_dir
        /
        "figure_01_major_temperature_trends.png",
        dpi=300,
        bbox_inches="tight",
    )


# =============================================================================
# FIGURE 2
# =============================================================================

def plot_temperature_slope_distributions(
    trends,
    output_dir,
):

    metrics = [
        "annual_tmean",
        "djf_tmin",
        "jja_tmax",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            14,
            4.5,
        ),
    )

    for ax, metric in zip(
        axes,
        metrics,
    ):

        values = (
            trends.loc[
                trends[
                    "metric"
                ]
                == metric,
                "sen_slope_per_decade",
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if len(values) == 0:
            continue

        ax.hist(
            values,
            bins=30,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.axvline(
            values.median(),
            linewidth=1.4,
        )

        ax.text(
            0.03,
            0.95,
            (
                f"Median = "
                f"{values.median():+.3f}"
            ),
            transform=ax.transAxes,
            va="top",
        )

        ax.set_title(
            METRICS[
                metric
            ]["label"],
            loc="left",
            fontsize=10,
        )

        ax.set_xlabel(
            "Sen slope (°C/decade)"
        )

        ax.set_ylabel(
            "Number of cities"
        )

        ax.grid(
            axis="y",
            alpha=0.15,
        )

    fig.suptitle(
        "Distribution of city-level warming trends",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    fig.savefig(
        output_dir
        /
        "figure_02_city_temperature_slope_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )


# =============================================================================
# FIGURE 3
# =============================================================================

def plot_cold_trend_distributions(
    trends,
    output_dir,
):

    metrics = [
        "fd0_pct",
        "fd5_pct",
        "fd10_pct",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            14,
            4.5,
        ),
    )

    for ax, metric in zip(
        axes,
        metrics,
    ):

        values = (
            trends.loc[
                trends[
                    "metric"
                ]
                == metric,
                "sen_slope_per_decade",
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if len(values) == 0:
            continue

        ax.hist(
            values,
            bins=30,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.axvline(
            values.median(),
            linewidth=1.4,
        )

        ax.text(
            0.03,
            0.95,
            (
                f"Median = "
                f"{values.median():+.3f}"
            ),
            transform=ax.transAxes,
            va="top",
        )

        ax.set_title(
            METRICS[
                metric
            ]["label"],
            loc="left",
            fontsize=10,
        )

        ax.set_xlabel(
            "Percentage points/decade"
        )

        ax.set_ylabel(
            "Number of cities"
        )

        ax.grid(
            axis="y",
            alpha=0.15,
        )

    fig.suptitle(
        "City-level trends in cold-day frequency",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    fig.savefig(
        output_dir
        /
        "figure_03_cold_extreme_trend_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )


# =============================================================================
# FIGURE 4
# =============================================================================

def plot_hot_trend_distributions(
    trends,
    output_dir,
):

    metrics = [
        "hd35_pct",
        "tx95p_pct",
        "tx99p_pct",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            14,
            4.5,
        ),
    )

    for ax, metric in zip(
        axes,
        metrics,
    ):

        values = (
            trends.loc[
                trends[
                    "metric"
                ]
                == metric,
                "sen_slope_per_decade",
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if len(values) == 0:
            continue

        ax.hist(
            values,
            bins=30,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.axvline(
            values.median(),
            linewidth=1.4,
        )

        ax.text(
            0.03,
            0.95,
            (
                f"Median = "
                f"{values.median():+.3f}"
            ),
            transform=ax.transAxes,
            va="top",
        )

        ax.set_title(
            METRICS[
                metric
            ]["label"],
            loc="left",
            fontsize=10,
        )

        ax.set_xlabel(
            "Percentage points/decade"
        )

        ax.set_ylabel(
            "Number of cities"
        )

        ax.grid(
            axis="y",
            alpha=0.15,
        )

    fig.suptitle(
        "City-level trends in hot-day frequency",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    fig.savefig(
        output_dir
        /
        "figure_04_hot_extreme_trend_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )


# =============================================================================
# FIGURE 5
# =============================================================================

def plot_cold_hot_relationship(
    trends,
    output_dir,
):

    x = (
        trends.loc[
            trends[
                "metric"
            ]
            == "fd5_pct",
            [
                "city_id",
                "sen_slope_per_decade",
            ],
        ]
        .rename(
            columns={
                "sen_slope_per_decade":
                    "cold_slope"
            }
        )
    )

    y = (
        trends.loc[
            trends[
                "metric"
            ]
            == "tx99p_pct",
            [
                "city_id",
                "sen_slope_per_decade",
            ],
        ]
        .rename(
            columns={
                "sen_slope_per_decade":
                    "hot_slope"
            }
        )
    )

    merged = (
        x.merge(
            y,
            on="city_id",
            how="inner",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    fig, ax = plt.subplots(
        figsize=(
            7.5,
            6,
        )
    )

    ax.scatter(
        merged[
            "cold_slope"
        ],
        merged[
            "hot_slope"
        ],
        s=22,
        alpha=0.5,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    if len(merged) >= 3:

        rho, p = stats.spearmanr(
            merged[
                "cold_slope"
            ],
            merged[
                "hot_slope"
            ],
        )

        ax.text(
            0.03,
            0.97,
            (
                f"Spearman ρ = "
                f"{rho:.3f}\n"
                f"p = {p:.3g}"
            ),
            transform=ax.transAxes,
            va="top",
        )

    ax.set_xlabel(
        "Tmin < -5°C trend\n"
        "(percentage points/decade)"
    )

    ax.set_ylabel(
        "Tmax > city P99 trend\n"
        "(percentage points/decade)"
    )

    ax.set_title(
        (
            "Relationship between declining cold "
            "and increasing extreme heat"
        ),
        loc="left",
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "figure_05_cold_vs_hot_city_trends.png",
        dpi=300,
        bbox_inches="tight",
    )


# =============================================================================
# FIGURE 6
# =============================================================================

def plot_significant_trend_share(
    trends,
    output_dir,
):

    metrics = [
        "annual_tmean",
        "djf_tmin",
        "jja_tmax",
        "fd5_pct",
        "hd35_pct",
        "tx95p_pct",
        "tx99p_pct",
    ]

    rows = []

    for metric in metrics:

        sub = trends.loc[
            trends[
                "metric"
            ]
            == metric
        ]

        if sub.empty:
            continue

        n = len(sub)

        rows.append(
            {
                "label":
                    METRICS[
                        metric
                    ]["label"],

                "increase":
                    (
                        (
                            sub[
                                "trend_class"
                            ]
                            ==
                            "significant increase"
                        ).sum()
                        /
                        n
                        * 100
                    ),

                "decrease":
                    (
                        (
                            sub[
                                "trend_class"
                            ]
                            ==
                            "significant decrease"
                        ).sum()
                        /
                        n
                        * 100
                    ),

                "not_significant":
                    (
                        (
                            sub[
                                "trend_class"
                            ]
                            ==
                            "not significant"
                        ).sum()
                        /
                        n
                        * 100
                    ),
            }
        )

    plot_df = pd.DataFrame(
        rows
    )

    y = np.arange(
        len(plot_df)
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    left = np.zeros(
        len(plot_df)
    )

    for col, label in [

        (
            "increase",
            "Significant increase",
        ),

        (
            "decrease",
            "Significant decrease",
        ),

        (
            "not_significant",
            "Not significant",
        ),

    ]:

        ax.barh(
            y,
            plot_df[
                col
            ],
            left=left,
            label=label,
        )

        left += plot_df[
            col
        ].to_numpy()

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        plot_df[
            "label"
        ]
    )

    ax.set_xlim(
        0,
        100,
    )

    ax.set_xlabel(
        "Percentage of cities (%)"
    )

    ax.set_title(
        (
            "Share of cities with significant "
            "temporal trends"
        ),
        loc="left",
    )

    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            1.01,
        ),
    )

    ax.grid(
        axis="x",
        alpha=0.15,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "figure_06_significant_city_trend_share.png",
        dpi=300,
        bbox_inches="tight",
    )


# =============================================================================
# FIGURE 7
# =============================================================================

def plot_period_changes(
    period_results,
    output_dir,
):

    metrics = [
        "annual_tmean",
        "djf_tmin",
        "jja_tmax",
        "fd5_pct",
        "tx99p_pct",
    ]

    data = []
    labels = []

    for metric in metrics:

        values = (
            period_results.loc[
                period_results[
                    "metric"
                ]
                == metric,
                "mean_change",
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if len(values) == 0:
            continue

        data.append(
            values.to_numpy()
        )

        labels.append(
            METRICS[
                metric
            ]["label"]
        )

    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    ax.boxplot(
        data,
        tick_labels=labels,
        vert=False,
        showfliers=False,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Recent-period mean minus early-period mean"
    )

    ax.set_title(
        (
            "City-level change between "
            "early and recent periods"
        ),
        loc="left",
    )

    ax.grid(
        axis="x",
        alpha=0.15,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "figure_07_early_recent_city_changes.png",
        dpi=300,
        bbox_inches="tight",
    )


# =============================================================================
# KEY RESULTS
# =============================================================================

def print_key_results(
    trends,
    national_trends,
):

    print("\n" + "=" * 80)
    print("KEY TEMPORAL RESULTS")
    print("=" * 80)

    major_metrics = [
        "annual_tmean",
        "djf_tmin",
        "jja_tmax",
        "fd5_pct",
        "tx99p_pct",
    ]

    for metric in major_metrics:

        city = trends.loc[
            trends[
                "metric"
            ]
            == metric
        ]

        national = (
            national_trends.loc[
                national_trends[
                    "metric"
                ]
                == metric
            ]
        )

        if city.empty:
            continue

        print(
            "\n"
            +
            METRICS[
                metric
            ]["label"]
        )

        print(
            "-" * 60
        )

        print(
            "Median city Sen slope: "
            f"{city['sen_slope_per_decade'].median():+.4f} "
            f"{METRICS[metric]['trend_unit']}"
        )

        increasing = (
            (
                city[
                    "trend_class"
                ]
                ==
                "significant increase"
            ).mean()
            * 100
        )

        decreasing = (
            (
                city[
                    "trend_class"
                ]
                ==
                "significant decrease"
            ).mean()
            * 100
        )

        print(
            f"Significant increase: "
            f"{increasing:.1f}% of cities"
        )

        print(
            f"Significant decrease: "
            f"{decreasing:.1f}% of cities"
        )

        if not national.empty:

            row = national.iloc[
                0
            ]

            print(
                "Trend of yearly city median: "
                f"{row['sen_slope_per_decade']:+.4f} "
                f"{METRICS[metric]['trend_unit']}"
            )

            print(
                f"MK method: "
                f"{row['mk_method']}"
            )

            print(
                f"MK p: "
                f"{row['mk_p']:.4g}"
            )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help=(
            "Save figures without opening them."
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():

        raise FileNotFoundError(
            f"Input does not exist:\n"
            f"{args.input}"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS OF CITY TEMPERATURE")
    print("=" * 80)

    if HAS_PYMK:

        print(
            "Mann-Kendall method: "
            "Hamed-Rao modified MK "
            "with automatic Kendall fallback."
        )

    else:

        print(
            "pymannkendall unavailable: "
            "using standard Kendall tau."
        )

    # =========================================================================
    # LOAD
    # =========================================================================

    df = load_data(
        args.input
    )

    # =========================================================================
    # CITY TRENDS
    # =========================================================================

    trends = (
        calculate_city_trends(
            df
        )
    )

    trend_long_path = (
        args.output
        /
        "city_temporal_trends_long.csv"
    )

    trends.to_csv(
        trend_long_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved:\n{trend_long_path}"
    )

    # =========================================================================
    # WIDE TABLE
    # =========================================================================

    wide = (
        build_wide_city_trend_table(
            trends
        )
    )

    wide_path = (
        args.output
        /
        "city_temporal_trends_wide.csv"
    )

    wide.to_csv(
        wide_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved:\n{wide_path}"
    )

    # =========================================================================
    # PERIOD COMPARISON
    # =========================================================================

    periods = (
        period_comparison(
            df
        )
    )

    periods_path = (
        args.output
        /
        "city_period_comparison.csv"
    )

    periods.to_csv(
        periods_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved:\n{periods_path}"
    )

    # =========================================================================
    # YEARLY CITY DISTRIBUTION
    # =========================================================================

    summary = (
        yearly_city_summary(
            df
        )
    )

    summary_path = (
        args.output
        /
        "yearly_city_distribution_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # NATIONAL CITY MEDIAN TREND
    # =========================================================================

    national_trends = (
        national_median_trends(
            summary
        )
    )

    national_path = (
        args.output
        /
        "national_city_median_temporal_trends.csv"
    )

    national_trends.to_csv(
        national_path,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # FIGURES
    # =========================================================================

    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    plot_major_temperature_trends(
        summary,
        national_trends,
        args.output,
    )

    print(
        "Saved figure 1."
    )

    plot_temperature_slope_distributions(
        trends,
        args.output,
    )

    print(
        "Saved figure 2."
    )

    plot_cold_trend_distributions(
        trends,
        args.output,
    )

    print(
        "Saved figure 3."
    )

    plot_hot_trend_distributions(
        trends,
        args.output,
    )

    print(
        "Saved figure 4."
    )

    plot_cold_hot_relationship(
        trends,
        args.output,
    )

    print(
        "Saved figure 5."
    )

    plot_significant_trend_share(
        trends,
        args.output,
    )

    print(
        "Saved figure 6."
    )

    plot_period_changes(
        periods,
        args.output,
    )

    print(
        "Saved figure 7."
    )

    # =========================================================================
    # RESULTS
    # =========================================================================

    print_key_results(
        trends,
        national_trends,
    )

    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        "\nMain output for future spatial analysis:"
    )

    print(
        wide_path
    )

    print(
        "\nNo Parquet files were created."
    )

    # =========================================================================
    # DISPLAY
    # =========================================================================

    if not args.no_show:

        print(
            "\nFigures will now be displayed."
        )

        print(
            "Close the figure windows to finish."
        )

        plt.show()


if __name__ == "__main__":
    main()