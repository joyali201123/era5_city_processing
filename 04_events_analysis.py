"""
04_extreme_heat_analysis.py

Extreme heat / return-period analysis for Chinese urban centres.

INPUTS
------
1. city_year_metrics_1950_2025.csv
   From 01_build_city_year_metrics.py

   Required:
       city_id
       city_name
       year
       txx

   TXx = annual maximum of daily city-area-weighted Tmax.

2. city_spatial_analysis_results.csv
   From 03_spatial_analysis.py

   Used only for:
       longitude
       latitude

MAIN QUESTIONS
--------------
1. What is the historical city-specific 100-year extreme heat level?

2. How does the recent 100-year return level differ?

3. How often would the historical 100-year event occur
   under the recent climate?

4. Where has the historical 100-year threshold already
   been exceeded?

5. When did each city first exceed it?

6. Are historically rare heat extremes increasingly
   occurring simultaneously across many cities?

METHOD
------
Block-maxima GEV:

    annual TXx -> GEV

Historical reference:
    1953-1989

Recent climate:
    1990-2025

For each city:

    z100_hist
        historical 100-year return level

    z100_recent
        recent 100-year return level

    recent_probability_of_hist_z100
        probability that recent climate exceeds
        historical z100

    effective_return_period_recent
        1 / recent exceedance probability

    probability_amplification
        recent probability / 0.01

Important:
A 100-year return level means 1% annual exceedance probability
under a stationary fitted distribution. It does NOT mean that the
event must happen exactly once every 100 years.

The 100-year level is an extrapolation because the observational
record is shorter than 100 years.

NO NEW PACKAGES REQUIRED
------------------------
numpy
pandas
scipy
matplotlib

OUTPUT
------
CSV + PNG only.
"""

from __future__ import annotations

import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import genextreme


# =============================================================================
# SETTINGS
# =============================================================================

ROOT = pathlib.Path(
    __file__
).resolve().parent


# -----------------------------------------------------------------------------
# Input from 01
# -----------------------------------------------------------------------------

DEFAULT_CITY_YEAR = (
    ROOT
    / "results_city_year_metrics_1950_2025"
    / "city_year_metrics_1950_2025.csv"
)


# -----------------------------------------------------------------------------
# Input from 03
# -----------------------------------------------------------------------------

DEFAULT_SPATIAL = (
    ROOT
    / "results_spatial_analysis_1950_2025"
    / "city_spatial_analysis_results.csv"
)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT = (
    ROOT
    / "results_extreme_heat_analysis_1950_2025"
)


# =============================================================================
# EXTREME-VALUE PERIODS
# =============================================================================

HIST_START = 1953
HIST_END = 1989

RECENT_START = 1990
RECENT_END = 2025


# =============================================================================
# RETURN PERIODS
# =============================================================================

RETURN_PERIODS = [
    10,
    20,
    50,
    100,
]


# Minimum annual maxima needed to fit a period-specific GEV
MIN_YEARS_GEV = 25


# Cap plotted return period so a few enormous values
# do not destroy figure readability.
RP_PLOT_CAP = 200


# =============================================================================
# BOOTSTRAP
# =============================================================================

# Default 0 = fast exploratory analysis.
#
# For publication-level uncertainty, run for example:
#
# python 04_extreme_heat_analysis.py --bootstrap 200
#
DEFAULT_BOOTSTRAP = 0

RANDOM_SEED = 42


# =============================================================================
# HELPERS
# =============================================================================

def save_csv(
    df: pd.DataFrame,
    path: pathlib.Path,
):

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved: {path}"
    )


def clean_numeric(
    series,
):

    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )


# =============================================================================
# LOAD CITY-YEAR DATA
# =============================================================================

def load_city_year(
    path: pathlib.Path,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LOADING CITY-YEAR EXTREME TEMPERATURE DATA"
    )

    print(
        "=" * 80
    )

    if not path.exists():

        raise FileNotFoundError(
            f"City-year file does not exist:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "city_id",
        "city_name",
        "year",
        "txx",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    df[
        "city_id"
    ] = clean_numeric(
        df[
            "city_id"
        ]
    )

    df[
        "year"
    ] = clean_numeric(
        df[
            "year"
        ]
    )

    df[
        "txx"
    ] = clean_numeric(
        df[
            "txx"
        ]
    )

    df = df.dropna(
        subset=[
            "city_id",
            "year",
            "txx",
        ]
    ).copy()

    df[
        "city_id"
    ] = (
        df[
            "city_id"
        ]
        .astype(
            int
        )
    )

    df[
        "year"
    ] = (
        df[
            "year"
        ]
        .astype(
            int
        )
    )

    df = df.sort_values(
        [
            "city_id",
            "year",
        ]
    )

    print(
        f"Rows   : {len(df):,}"
    )

    print(
        f"Cities : "
        f"{df['city_id'].nunique():,}"
    )

    print(
        f"Years  : "
        f"{df['year'].min()}-"
        f"{df['year'].max()}"
    )

    print(
        f"\nHistorical GEV: "
        f"{HIST_START}-{HIST_END}"
    )

    print(
        f"Recent GEV    : "
        f"{RECENT_START}-{RECENT_END}"
    )

    return df


# =============================================================================
# LOAD SPATIAL COORDINATES
# =============================================================================

def load_spatial(
    path: pathlib.Path,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LOADING CITY SPATIAL INFORMATION"
    )

    print(
        "=" * 80
    )

    if not path.exists():

        print(
            "WARNING: spatial analysis file "
            "was not found."
        )

        print(
            "Extreme-value calculations will run, "
            "but spatial maps cannot be produced."
        )

        return None

    spatial = pd.read_csv(
        path
    )

    required = {
        "city_id",
        "longitude",
        "latitude",
    }

    missing = (
        required
        - set(
            spatial.columns
        )
    )

    if missing:

        print(
            f"WARNING: spatial file missing "
            f"{sorted(missing)}"
        )

        return None

    spatial[
        "city_id"
    ] = clean_numeric(
        spatial[
            "city_id"
        ]
    )

    spatial[
        "longitude"
    ] = clean_numeric(
        spatial[
            "longitude"
        ]
    )

    spatial[
        "latitude"
    ] = clean_numeric(
        spatial[
            "latitude"
        ]
    )

    spatial = (
        spatial[
            [
                "city_id",
                "longitude",
                "latitude",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=[
                "city_id"
            ]
        )
    )

    spatial[
        "city_id"
    ] = (
        spatial[
            "city_id"
        ]
        .astype(
            int
        )
    )

    print(
        f"Cities with coordinates: "
        f"{len(spatial):,}"
    )

    return spatial


# =============================================================================
# GEV FIT
# =============================================================================

def fit_gev(
    values,
):

    """
    Fit scipy.stats.genextreme.

    SciPy parameterisation:
        c, loc, scale

    Returns None if fit is invalid.
    """

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if len(
        x
    ) < MIN_YEARS_GEV:

        return None

    if np.std(
        x
    ) <= 1e-8:

        return None

    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            shape, loc, scale = (
                genextreme.fit(
                    x
                )
            )

        if (
            not np.isfinite(
                shape
            )
            or
            not np.isfinite(
                loc
            )
            or
            not np.isfinite(
                scale
            )
            or
            scale <= 0
        ):

            return None

        return {
            "shape":
                float(
                    shape
                ),

            "loc":
                float(
                    loc
                ),

            "scale":
                float(
                    scale
                ),

            "n":
                int(
                    len(
                        x
                    )
                ),
        }

    except Exception:

        return None


# =============================================================================
# RETURN LEVEL
# =============================================================================

def return_level(
    fit,
    return_period,
):

    if fit is None:
        return np.nan

    probability = (
        1
        -
        1
        /
        return_period
    )

    try:

        value = genextreme.ppf(
            probability,
            fit[
                "shape"
            ],
            loc=fit[
                "loc"
            ],
            scale=fit[
                "scale"
            ],
        )

        if not np.isfinite(
            value
        ):

            return np.nan

        return float(
            value
        )

    except Exception:

        return np.nan


# =============================================================================
# EXCEEDANCE PROBABILITY
# =============================================================================

def exceedance_probability(
    threshold,
    fit,
):

    if (
        fit is None
        or
        not np.isfinite(
            threshold
        )
    ):

        return np.nan

    try:

        p = genextreme.sf(
            threshold,
            fit[
                "shape"
            ],
            loc=fit[
                "loc"
            ],
            scale=fit[
                "scale"
            ],
        )

        if not np.isfinite(
            p
        ):

            return np.nan

        return float(
            np.clip(
                p,
                0,
                1,
            )
        )

    except Exception:

        return np.nan


# =============================================================================
# EFFECTIVE RETURN PERIOD
# =============================================================================

def effective_return_period(
    probability,
):

    if (
        not np.isfinite(
            probability
        )
        or
        probability <= 0
    ):

        return np.inf

    return (
        1
        /
        probability
    )


# =============================================================================
# BOOTSTRAP RETURN LEVEL
# =============================================================================

def bootstrap_return_level(
    values,
    return_period=100,
    n_bootstrap=0,
    seed=42,
):

    if n_bootstrap <= 0:

        return (
            np.nan,
            np.nan,
        )

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if len(
        x
    ) < MIN_YEARS_GEV:

        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    estimates = []

    for _ in range(
        n_bootstrap
    ):

        sample = rng.choice(
            x,
            size=len(
                x
            ),
            replace=True,
        )

        fit = fit_gev(
            sample
        )

        if fit is None:
            continue

        rl = return_level(
            fit,
            return_period,
        )

        if np.isfinite(
            rl
        ):

            estimates.append(
                rl
            )

    if len(
        estimates
    ) < max(
        20,
        0.25
        *
        n_bootstrap,
    ):

        return (
            np.nan,
            np.nan,
        )

    return (
        float(
            np.percentile(
                estimates,
                2.5,
            )
        ),
        float(
            np.percentile(
                estimates,
                97.5,
            )
        ),
    )


# =============================================================================
# CITY-SPECIFIC EXTREME VALUE ANALYSIS
# =============================================================================

def analyse_city_extremes(
    df,
    n_bootstrap=0,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FITTING CITY-SPECIFIC GEV MODELS"
    )

    print(
        "=" * 80
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
            ].iloc[
                0
            ]
        )

        hist = (
            group.loc[
                group[
                    "year"
                ].between(
                    HIST_START,
                    HIST_END,
                ),
                "txx",
            ]
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
                    RECENT_START,
                    RECENT_END,
                ),
                "txx",
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        hist_fit = fit_gev(
            hist
        )

        recent_fit = fit_gev(
            recent
        )

        if (
            hist_fit is None
            or
            recent_fit is None
        ):

            continue

        row = {
            "city_id":
                city_id,

            "city_name":
                city_name,

            "hist_start":
                HIST_START,

            "hist_end":
                HIST_END,

            "recent_start":
                RECENT_START,

            "recent_end":
                RECENT_END,

            "n_hist":
                hist_fit[
                    "n"
                ],

            "n_recent":
                recent_fit[
                    "n"
                ],

            # Historical GEV
            "hist_shape":
                hist_fit[
                    "shape"
                ],

            "hist_loc":
                hist_fit[
                    "loc"
                ],

            "hist_scale":
                hist_fit[
                    "scale"
                ],

            # Recent GEV
            "recent_shape":
                recent_fit[
                    "shape"
                ],

            "recent_loc":
                recent_fit[
                    "loc"
                ],

            "recent_scale":
                recent_fit[
                    "scale"
                ],
        }

        # ---------------------------------------------------------------------
        # Return levels
        # ---------------------------------------------------------------------

        for rp in RETURN_PERIODS:

            row[
                f"hist_rl{rp}"
            ] = return_level(
                hist_fit,
                rp,
            )

            row[
                f"recent_rl{rp}"
            ] = return_level(
                recent_fit,
                rp,
            )

            row[
                f"rl{rp}_change"
            ] = (
                row[
                    f"recent_rl{rp}"
                ]
                -
                row[
                    f"hist_rl{rp}"
                ]
            )

        # ---------------------------------------------------------------------
        # Historical 100-year threshold under recent climate
        # ---------------------------------------------------------------------

        z100 = row[
            "hist_rl100"
        ]

        recent_probability = (
            exceedance_probability(
                z100,
                recent_fit,
            )
        )

        row[
            "recent_probability_hist_rl100"
        ] = recent_probability

        row[
            "recent_effective_return_period"
        ] = effective_return_period(
            recent_probability
        )

        # Historical probability = 0.01 by definition
        row[
            "probability_amplification"
        ] = (
            recent_probability
            /
            0.01
            if np.isfinite(
                recent_probability
            )
            else np.nan
        )

        # ---------------------------------------------------------------------
        # Optional historical z100 uncertainty
        # ---------------------------------------------------------------------

        (
            ci_low,
            ci_high,
        ) = bootstrap_return_level(
            hist,
            return_period=100,
            n_bootstrap=n_bootstrap,
            seed=(
                RANDOM_SEED
                +
                number
            ),
        )

        row[
            "hist_rl100_ci_low"
        ] = ci_low

        row[
            "hist_rl100_ci_high"
        ] = ci_high

        rows.append(
            row
        )

        if (
            number % 100 == 0
            or
            number
            == n_cities
        ):

            print(
                f"Cities processed: "
                f"{number}/{n_cities}"
            )

    result = pd.DataFrame(
        rows
    )

    print(
        f"\nSuccessful historical + recent "
        f"GEV fits: {len(result):,}"
    )

    return result


# =============================================================================
# IDENTIFY ACTUAL EXCEEDANCE EVENTS
# =============================================================================

def identify_exceedances(
    city_year,
    city_results,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "IDENTIFYING HISTORICAL 100-YEAR EXCEEDANCE EVENTS"
    )

    print(
        "=" * 80
    )

    thresholds = (
        city_results[
            [
                "city_id",
                "hist_rl100",
            ]
        ]
        .dropna()
    )

    events = city_year.merge(
        thresholds,
        on="city_id",
        how="inner",
    )

    events[
        "exceeds_hist_rl100"
    ] = (
        events[
            "txx"
        ]
        >
        events[
            "hist_rl100"
        ]
    )

    events[
        "excess_degC"
    ] = (
        events[
            "txx"
        ]
        -
        events[
            "hist_rl100"
        ]
    )

    # -------------------------------------------------------------------------
    # Per-city observed exceedance statistics
    # -------------------------------------------------------------------------

    city_event_rows = []

    for city_id, group in events.groupby(
        "city_id"
    ):

        exceed = group.loc[
            group[
                "exceeds_hist_rl100"
            ]
        ]

        hist_exceed = exceed.loc[
            exceed[
                "year"
            ].between(
                HIST_START,
                HIST_END,
            )
        ]

        recent_exceed = exceed.loc[
            exceed[
                "year"
            ].between(
                RECENT_START,
                RECENT_END,
            )
        ]

        city_event_rows.append(
            {
                "city_id":
                    city_id,

                "observed_exceedance_count_all":
                    len(
                        exceed
                    ),

                "observed_exceedance_count_hist":
                    len(
                        hist_exceed
                    ),

                "observed_exceedance_count_recent":
                    len(
                        recent_exceed
                    ),

                "first_exceedance_year":
                    (
                        int(
                            exceed[
                                "year"
                            ].min()
                        )
                        if len(
                            exceed
                        )
                        > 0
                        else np.nan
                    ),

                "last_exceedance_year":
                    (
                        int(
                            exceed[
                                "year"
                            ].max()
                        )
                        if len(
                            exceed
                        )
                        > 0
                        else np.nan
                    ),

                "maximum_observed_excess_degC":
                    (
                        float(
                            exceed[
                                "excess_degC"
                            ].max()
                        )
                        if len(
                            exceed
                        )
                        > 0
                        else np.nan
                    ),
            }
        )

    city_events = pd.DataFrame(
        city_event_rows
    )

    city_results = (
        city_results.merge(
            city_events,
            on="city_id",
            how="left",
        )
    )

    return (
        events,
        city_results,
    )


# =============================================================================
# SYNCHRONOUS MULTI-CITY EXTREMES
# =============================================================================

def synchronous_extremes(
    events,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CALCULATING MULTI-CITY SYNCHRONOUS EXTREMES"
    )

    print(
        "=" * 80
    )

    rows = []

    for year, group in events.groupby(
        "year"
    ):

        total = group[
            "city_id"
        ].nunique()

        affected = (
            group.loc[
                group[
                    "exceeds_hist_rl100"
                ],
                "city_id",
            ]
            .nunique()
        )

        rows.append(
            {
                "year":
                    int(
                        year
                    ),

                "n_cities_available":
                    int(
                        total
                    ),

                "n_cities_exceeding":
                    int(
                        affected
                    ),

                "pct_cities_exceeding":
                    (
                        affected
                        /
                        total
                        *
                        100
                        if total
                        > 0
                        else np.nan
                    ),
            }
        )

    result = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "year"
        )
    )

    top = (
        result
        .sort_values(
            "pct_cities_exceeding",
            ascending=False,
        )
        .head(
            10
        )
    )

    print(
        "\nTop synchronous extreme years:"
    )

    print(
        top[
            [
                "year",
                "n_cities_exceeding",
                "pct_cities_exceeding",
            ]
        ].to_string(
            index=False
        )
    )

    return result


# =============================================================================
# MERGE COORDINATES
# =============================================================================

def attach_coordinates(
    city_results,
    spatial,
):

    if spatial is None:
        return city_results

    return city_results.merge(
        spatial,
        on="city_id",
        how="left",
    )


# =============================================================================
# MAP HELPER
# =============================================================================

def set_map_extent(
    ax,
    df,
):

    x = df[
        "longitude"
    ]

    y = df[
        "latitude"
    ]

    xmin = x.min()
    xmax = x.max()

    ymin = y.min()
    ymax = y.max()

    ax.set_xlim(
        xmin - 2,
        xmax + 2,
    )

    ax.set_ylim(
        ymin - 2,
        ymax + 2,
    )

    mean_lat = (
        ymin
        +
        ymax
    ) / 2

    ax.set_aspect(
        1
        /
        np.cos(
            np.radians(
                mean_lat
            )
        )
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )

    ax.grid(
        alpha=0.12
    )


# =============================================================================
# FIGURE 1
# HISTORICAL VS RECENT RETURN LEVELS
# =============================================================================

def plot_return_level_distributions(
    city_results,
    output,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            14,
            4.8,
        ),
    )

    # -------------------------------------------------------------------------
    # Historical z100
    # -------------------------------------------------------------------------

    x = (
        city_results[
            "hist_rl100"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    axes[
        0
    ].hist(
        x,
        bins=35,
        edgecolor="white",
    )

    axes[
        0
    ].axvline(
        x.median(),
        linestyle="--",
    )

    axes[
        0
    ].set_title(
        "Historical 100-year return level",
        loc="left",
    )

    axes[
        0
    ].set_xlabel(
        "TXx (°C)"
    )

    axes[
        0
    ].set_ylabel(
        "Number of cities"
    )

    # -------------------------------------------------------------------------
    # Recent z100
    # -------------------------------------------------------------------------

    x = (
        city_results[
            "recent_rl100"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    axes[
        1
    ].hist(
        x,
        bins=35,
        edgecolor="white",
    )

    axes[
        1
    ].axvline(
        x.median(),
        linestyle="--",
    )

    axes[
        1
    ].set_title(
        "Recent 100-year return level",
        loc="left",
    )

    axes[
        1
    ].set_xlabel(
        "TXx (°C)"
    )

    axes[
        1
    ].set_ylabel(
        "Number of cities"
    )

    # -------------------------------------------------------------------------
    # Difference
    # -------------------------------------------------------------------------

    x = (
        city_results[
            "rl100_change"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    axes[
        2
    ].hist(
        x,
        bins=35,
        edgecolor="white",
    )

    axes[
        2
    ].axvline(
        0,
        linestyle="--",
    )

    axes[
        2
    ].axvline(
        x.median(),
        linewidth=1.5,
    )

    axes[
        2
    ].set_title(
        "Change in 100-year return level",
        loc="left",
    )

    axes[
        2
    ].set_xlabel(
        "Recent − historical (°C)"
    )

    axes[
        2
    ].set_ylabel(
        "Number of cities"
    )

    fig.suptitle(
        "City-specific extreme heat return levels",
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

    path = (
        output
        /
        "figure_01_return_level_distributions.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved: {path}"
    )


# =============================================================================
# FIGURE 2
# RETURN-PERIOD SHORTENING
# =============================================================================

def plot_return_period_shortening(
    city_results,
    output,
):

    x = (
        city_results[
            "recent_effective_return_period"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    x = x.clip(
        upper=RP_PLOT_CAP
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5.5,
        )
    )

    ax.hist(
        x,
        bins=40,
        edgecolor="white",
    )

    ax.axvline(
        100,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Historical definition "
            "(100 years)"
        ),
    )

    median = x.median()

    ax.axvline(
        median,
        linewidth=1.5,
        label=(
            f"City median = "
            f"{median:.1f} years"
        ),
    )

    ax.set_xlabel(
        "Recent effective return period (years)"
    )

    ax.set_ylabel(
        "Number of cities"
    )

    ax.set_title(
        (
            "How rare is the historical "
            "100-year event under recent climate?"
        ),
        loc="left",
    )

    ax.legend(
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.15,
    )

    fig.tight_layout()

    path = (
        output
        /
        "figure_02_return_period_shortening.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved: {path}"
    )


# =============================================================================
# FIGURE 3
# SYNCHRONOUS EXTREMES
# =============================================================================

def plot_synchronous_extremes(
    synchronous,
    output,
):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            11,
            7.5,
        ),
        sharex=True,
    )

    axes[
        0
    ].bar(
        synchronous[
            "year"
        ],
        synchronous[
            "n_cities_exceeding"
        ],
        width=0.8,
    )

    axes[
        0
    ].set_ylabel(
        "Number of cities"
    )

    axes[
        0
    ].set_title(
        (
            "Cities exceeding their historical "
            "100-year heat threshold"
        ),
        loc="left",
    )

    axes[
        0
    ].grid(
        axis="y",
        alpha=0.15,
    )

    axes[
        1
    ].plot(
        synchronous[
            "year"
        ],
        synchronous[
            "pct_cities_exceeding"
        ],
        linewidth=1.5,
    )

    axes[
        1
    ].set_xlabel(
        "Year"
    )

    axes[
        1
    ].set_ylabel(
        "Cities exceeding (%)"
    )

    axes[
        1
    ].set_title(
        (
            "Spatial synchronisation of "
            "historically rare extreme heat"
        ),
        loc="left",
    )

    axes[
        1
    ].grid(
        alpha=0.15,
    )

    fig.tight_layout()

    path = (
        output
        /
        "figure_03_synchronous_extreme_heat.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved: {path}"
    )


# =============================================================================
# FIGURE 4
# EXTREME HEAT SPATIAL MAPS
# =============================================================================

def plot_spatial_extremes(
    city_results,
    output,
):

    required = {
        "longitude",
        "latitude",
    }

    if not required.issubset(
        city_results.columns
    ):

        return

    plot_df = (
        city_results
        .dropna(
            subset=[
                "longitude",
                "latitude",
            ]
        )
        .copy()
    )

    if plot_df.empty:
        return

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            14,
            10,
        ),
    )

    # -------------------------------------------------------------------------
    # A. Historical z100
    # -------------------------------------------------------------------------

    ax = axes[
        0,
        0
    ]

    values = plot_df[
        "hist_rl100"
    ]

    sc = ax.scatter(
        plot_df[
            "longitude"
        ],
        plot_df[
            "latitude"
        ],
        c=values,
        s=14,
        cmap="hot",
        linewidths=0,
    )

    set_map_extent(
        ax,
        plot_df,
    )

    ax.set_title(
        (
            "Historical city-specific "
            "100-year TXx"
        ),
        loc="left",
    )

    cb = fig.colorbar(
        sc,
        ax=ax,
        shrink=0.8,
    )

    cb.set_label(
        "°C"
    )

    # -------------------------------------------------------------------------
    # B. z100 change
    # -------------------------------------------------------------------------

    ax = axes[
        0,
        1
    ]

    values = plot_df[
        "rl100_change"
    ]

    finite = np.isfinite(
        values
    )

    vmax = np.nanpercentile(
        np.abs(
            values[
                finite
            ]
        ),
        95,
    )

    if (
        not np.isfinite(
            vmax
        )
        or vmax == 0
    ):

        vmax = 1

    sc = ax.scatter(
        plot_df[
            "longitude"
        ],
        plot_df[
            "latitude"
        ],
        c=values,
        s=14,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        linewidths=0,
    )

    set_map_extent(
        ax,
        plot_df,
    )

    ax.set_title(
        (
            "Change in 100-year "
            "extreme heat level"
        ),
        loc="left",
    )

    cb = fig.colorbar(
        sc,
        ax=ax,
        shrink=0.8,
    )

    cb.set_label(
        "Recent − historical (°C)"
    )

    # -------------------------------------------------------------------------
    # C. Effective return period
    # -------------------------------------------------------------------------

    ax = axes[
        1,
        0
    ]

    values = (
        plot_df[
            "recent_effective_return_period"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .clip(
            upper=RP_PLOT_CAP
        )
    )

    sc = ax.scatter(
        plot_df[
            "longitude"
        ],
        plot_df[
            "latitude"
        ],
        c=values,
        s=14,
        cmap="viridis_r",
        linewidths=0,
    )

    set_map_extent(
        ax,
        plot_df,
    )

    ax.set_title(
        (
            "Recent return period of the "
            "historical 100-year event"
        ),
        loc="left",
    )

    cb = fig.colorbar(
        sc,
        ax=ax,
        shrink=0.8,
    )

    cb.set_label(
        "Years"
    )

    # -------------------------------------------------------------------------
    # D. Observed exceedance count
    # -------------------------------------------------------------------------

    ax = axes[
        1,
        1
    ]

    values = (
        plot_df[
            "observed_exceedance_count_recent"
        ]
        .fillna(
            0
        )
    )

    sc = ax.scatter(
        plot_df[
            "longitude"
        ],
        plot_df[
            "latitude"
        ],
        c=values,
        s=14,
        cmap="magma",
        linewidths=0,
    )

    set_map_extent(
        ax,
        plot_df,
    )

    ax.set_title(
        (
            "Observed exceedances during "
            f"{RECENT_START}-{RECENT_END}"
        ),
        loc="left",
    )

    cb = fig.colorbar(
        sc,
        ax=ax,
        shrink=0.8,
    )

    cb.set_label(
        "Number of years"
    )

    fig.suptitle(
        "Spatial pattern of rare extreme heat",
        fontsize=15,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    path = (
        output
        /
        "figure_04_spatial_extreme_heat_maps.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved: {path}"
    )


# =============================================================================
# FIGURE 5
# FIRST EXCEEDANCE YEAR
# =============================================================================

def plot_first_exceedance(
    city_results,
    output,
):

    required = {
        "longitude",
        "latitude",
        "first_exceedance_year",
    }

    if not required.issubset(
        city_results.columns
    ):

        return

    data = (
        city_results
        .dropna(
            subset=[
                "longitude",
                "latitude",
                "first_exceedance_year",
            ]
        )
        .copy()
    )

    if data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(
            10,
            8,
        )
    )

    sc = ax.scatter(
        data[
            "longitude"
        ],
        data[
            "latitude"
        ],
        c=data[
            "first_exceedance_year"
        ],
        s=18,
        cmap="viridis",
        linewidths=0,
    )

    set_map_extent(
        ax,
        data,
    )

    ax.set_title(
        (
            "First observed exceedance of the "
            "historical 100-year heat threshold"
        ),
        loc="left",
    )

    cb = fig.colorbar(
        sc,
        ax=ax,
        shrink=0.8,
    )

    cb.set_label(
        "First exceedance year"
    )

    fig.tight_layout()

    path = (
        output
        /
        "figure_05_first_exceedance_year.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved: {path}"
    )


# =============================================================================
# FIGURE 6
# HISTORICAL VS RECENT CITY z100
# =============================================================================

def plot_hist_recent_scatter(
    city_results,
    output,
):

    data = (
        city_results[
            [
                "hist_rl100",
                "recent_rl100",
            ]
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    fig, ax = plt.subplots(
        figsize=(
            7,
            6.5,
        )
    )

    ax.scatter(
        data[
            "hist_rl100"
        ],
        data[
            "recent_rl100"
        ],
        s=20,
        alpha=0.45,
    )

    lower = min(
        data[
            "hist_rl100"
        ].min(),
        data[
            "recent_rl100"
        ].min(),
    )

    upper = max(
        data[
            "hist_rl100"
        ].max(),
        data[
            "recent_rl100"
        ].max(),
    )

    ax.plot(
        [
            lower,
            upper,
        ],
        [
            lower,
            upper,
        ],
        linestyle="--",
        linewidth=1,
    )

    above = (
        data[
            "recent_rl100"
        ]
        >
        data[
            "hist_rl100"
        ]
    ).mean() * 100

    median_change = (
        data[
            "recent_rl100"
        ]
        -
        data[
            "hist_rl100"
        ]
    ).median()

    ax.text(
        0.03,
        0.97,
        (
            f"Recent > historical: "
            f"{above:.1f}% of cities\n"
            f"Median change: "
            f"{median_change:+.2f}°C"
        ),
        transform=ax.transAxes,
        va="top",
    )

    ax.set_xlabel(
        "Historical 100-year TXx (°C)"
    )

    ax.set_ylabel(
        "Recent 100-year TXx (°C)"
    )

    ax.set_title(
        "Shift in city-specific 100-year extreme heat",
        loc="left",
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    path = (
        output
        /
        "figure_06_historical_vs_recent_rl100.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved: {path}"
    )


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_key_results(
    city_results,
    synchronous,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "KEY EXTREME-HEAT RESULTS"
    )

    print(
        "=" * 80
    )

    n = len(
        city_results
    )

    print(
        f"\nCities with valid GEV fits: "
        f"{n:,}"
    )

    # -------------------------------------------------------------------------
    # Historical z100
    # -------------------------------------------------------------------------

    x = (
        city_results[
            "hist_rl100"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    print(
        "\nHistorical 100-year TXx"
    )

    print(
        "-" * 60
    )

    print(
        f"City median : "
        f"{x.median():.2f} °C"
    )

    print(
        f"P25-P75     : "
        f"{x.quantile(0.25):.2f} "
        f"to "
        f"{x.quantile(0.75):.2f} °C"
    )

    # -------------------------------------------------------------------------
    # Change
    # -------------------------------------------------------------------------

    x = (
        city_results[
            "rl100_change"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    print(
        "\nChange in 100-year TXx"
    )

    print(
        "-" * 60
    )

    print(
        f"Median change: "
        f"{x.median():+.2f} °C"
    )

    print(
        f"Cities with higher recent z100: "
        f"{(x > 0).mean() * 100:.1f}%"
    )

    # -------------------------------------------------------------------------
    # Return period shortening
    # -------------------------------------------------------------------------

    rp = (
        city_results[
            "recent_effective_return_period"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    print(
        "\nRecent effective return period "
        "of historical z100"
    )

    print(
        "-" * 60
    )

    print(
        f"Median: "
        f"{rp.median():.1f} years"
    )

    for threshold in [
        50,
        20,
        10,
        5,
    ]:

        share = (
            rp
            <= threshold
        ).mean() * 100

        print(
            f"≤ {threshold:3d} years: "
            f"{share:.1f}% of cities"
        )

    # -------------------------------------------------------------------------
    # Probability amplification
    # -------------------------------------------------------------------------

    amp = (
        city_results[
            "probability_amplification"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    print(
        "\nProbability amplification"
    )

    print(
        "-" * 60
    )

    print(
        f"Median amplification: "
        f"{amp.median():.2f}×"
    )

    # -------------------------------------------------------------------------
    # Observed exceedances
    # -------------------------------------------------------------------------

    exceed = (
        city_results[
            "observed_exceedance_count_recent"
        ]
        .fillna(
            0
        )
    )

    print(
        "\nObserved recent exceedances"
    )

    print(
        "-" * 60
    )

    print(
        f"Cities with >=1 exceedance: "
        f"{(exceed >= 1).mean() * 100:.1f}%"
    )

    print(
        f"Cities with >=2 exceedances: "
        f"{(exceed >= 2).mean() * 100:.1f}%"
    )

    print(
        f"Cities with >=3 exceedances: "
        f"{(exceed >= 3).mean() * 100:.1f}%"
    )

    # -------------------------------------------------------------------------
    # Most synchronous year
    # -------------------------------------------------------------------------

    if not synchronous.empty:

        peak = synchronous.loc[
            synchronous[
                "pct_cities_exceeding"
            ].idxmax()
        ]

        print(
            "\nMost spatially synchronous year"
        )

        print(
            "-" * 60
        )

        print(
            f"Year: "
            f"{int(peak['year'])}"
        )

        print(
            f"Cities: "
            f"{int(peak['n_cities_exceeding']):,}"
        )

        print(
            f"Share: "
            f"{peak['pct_cities_exceeding']:.2f}%"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--city-year",
        type=pathlib.Path,
        default=DEFAULT_CITY_YEAR,
    )

    parser.add_argument(
        "--spatial",
        type=pathlib.Path,
        default=DEFAULT_SPATIAL,
    )

    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
        help=(
            "Number of bootstrap samples for "
            "historical 100-year return-level CI. "
            "Default 0 for fast exploratory run."
        ),
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "EXTREME HEAT / RETURN-PERIOD ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        f"City-year input : "
        f"{args.city_year}"
    )

    print(
        f"Spatial input   : "
        f"{args.spatial}"
    )

    print(
        f"Output          : "
        f"{args.output}"
    )

    print(
        f"Bootstrap       : "
        f"{args.bootstrap}"
    )

    # =========================================================================
    # LOAD
    # =========================================================================

    city_year = load_city_year(
        args.city_year
    )

    spatial = load_spatial(
        args.spatial
    )

    # =========================================================================
    # GEV ANALYSIS
    # =========================================================================

    city_results = (
        analyse_city_extremes(
            city_year,
            n_bootstrap=args.bootstrap,
        )
    )

    # =========================================================================
    # OBSERVED EXCEEDANCES
    # =========================================================================

    (
        event_panel,
        city_results,
    ) = identify_exceedances(
        city_year,
        city_results,
    )

    # =========================================================================
    # SYNCHRONOUS EXTREMES
    # =========================================================================

    synchronous = (
        synchronous_extremes(
            event_panel
        )
    )

    # =========================================================================
    # ADD SPATIAL COORDINATES
    # =========================================================================

    city_results = (
        attach_coordinates(
            city_results,
            spatial,
        )
    )

    # =========================================================================
    # SAVE TABLES
    # =========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SAVING RESULTS"
    )

    print(
        "=" * 80
    )

    save_csv(
        city_results,
        args.output
        /
        "city_extreme_heat_results.csv",
    )

    save_csv(
        event_panel,
        args.output
        /
        "city_year_extreme_heat_events.csv",
    )

    save_csv(
        synchronous,
        args.output
        /
        "annual_synchronous_extreme_heat.csv",
    )

    # -------------------------------------------------------------------------
    # Exceedance-only event table
    # -------------------------------------------------------------------------

    exceedance_only = (
        event_panel.loc[
            event_panel[
                "exceeds_hist_rl100"
            ]
        ]
        .copy()
    )

    save_csv(
        exceedance_only,
        args.output
        /
        "historical_100yr_threshold_exceedances.csv",
    )

    # -------------------------------------------------------------------------
    # Top synchronous years
    # -------------------------------------------------------------------------

    top_years = (
        synchronous
        .sort_values(
            "pct_cities_exceeding",
            ascending=False,
        )
        .head(
            20
        )
    )

    save_csv(
        top_years,
        args.output
        /
        "top_synchronous_extreme_years.csv",
    )

    # =========================================================================
    # FIGURES
    # =========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "GENERATING FIGURES"
    )

    print(
        "=" * 80
    )

    plot_return_level_distributions(
        city_results,
        args.output,
    )

    plot_return_period_shortening(
        city_results,
        args.output,
    )

    plot_synchronous_extremes(
        synchronous,
        args.output,
    )

    plot_spatial_extremes(
        city_results,
        args.output,
    )

    plot_first_exceedance(
        city_results,
        args.output,
    )

    plot_hist_recent_scatter(
        city_results,
        args.output,
    )

    # =========================================================================
    # PRINT KEY RESULTS
    # =========================================================================

    print_key_results(
        city_results,
        synchronous,
    )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "EXTREME HEAT ANALYSIS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nMain city-level result:"
    )

    print(
        args.output
        /
        "city_extreme_heat_results.csv"
    )

    print(
        "\nActual 100-year-threshold events:"
    )

    print(
        args.output
        /
        "historical_100yr_threshold_exceedances.csv"
    )

    print(
        "\nSynchronous extreme time series:"
    )

    print(
        args.output
        /
        "annual_synchronous_extreme_heat.csv"
    )

    # =========================================================================
    # DISPLAY
    # =========================================================================

    if not args.no_show:

        print(
            "\nFigures will now be displayed."
        )

        print(
            "Close all figure windows to finish."
        )

        plt.show()


if __name__ == "__main__":
    main()