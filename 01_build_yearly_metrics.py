"""
01_build_city_year_metrics.py

Build annual and seasonal city-level temperature metrics from the
ERA5-GHSL daily city temperature dataset.

Input
-----
city_daily_temperature_1950_2025.csv

Expected columns
----------------
date_utc
city_id
city_name
t2m_min_mean
t2m_max_mean
t2m_mean_mean

Main output
-----------
city_year_metrics_1950_2025.csv
city_year_metrics_1950_2025.parquet
city_baseline_thresholds_1961_1990.csv
city_baseline_thresholds_1961_1990.parquet

Metrics
-------
1. Annual mean:
   - annual_tmean
   - annual_tmin
   - annual_tmax

2. Annual extremes:
   - txx = annual maximum daily Tmax
   - tnn = annual minimum daily Tmin

3. Absolute cold days:
   - fd0_count   : Tmin < 0 C
   - fd5_count   : Tmin < -5 C
   - fd10_count  : Tmin < -10 C
   - corresponding percentages

4. Absolute hot days:
   - hd30_count  : Tmax > 30 C
   - hd35_count  : Tmax > 35 C
   - hd40_count  : Tmax > 40 C
   - corresponding percentages

5. City-specific relative extremes based on baseline:
   - Tmax P95
   - Tmax P99
   - Tmin P05
   - Tmin P01
   - TX95p
   - TX99p
   - TN05p
   - TN01p

6. Seasonal climate:
   - winter (DJF) Tmin / Tmean / Tmax
   - spring (MAM) Tmin / Tmean / Tmax
   - summer (JJA) Tmin / Tmean / Tmax
   - autumn (SON) Tmin / Tmean / Tmax

Important
---------
DJF is assigned to the year containing January-February.

Example:
December 2000 + January 2001 + February 2001
= winter 2001.

Daily values are based on the UTC-day definition produced by the
previous ERA5 preprocessing script.

Install
-------
pip install duckdb pandas numpy matplotlib pyarrow

Run
---
python 01_build_city_year_metrics.py

or

python 01_build_city_year_metrics.py ^
    --input "E:\\your_folder\\city_daily_temperature_1950_2025.csv" ^
    --output "E:\\your_folder\\01_city_year_metrics"
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# SETTINGS
# =============================================================================

START_YEAR = 1950
END_YEAR = 2025

BASELINE_START = 1961
BASELINE_END = 1990

DEFAULT_INPUT = pathlib.Path(
    r"C:\Users\joyal\OneDrive\Desktop\oscar-team\mask_era5_china\ERA5_t2m_1950_2025\city_daily_temperature_1953_2025.csv"
)

DEFAULT_OUTPUT = pathlib.Path(
    r"C:\Users\joyal\OneDrive\Desktop\oscar-team\mask_era5_china\results_city_year_metrics_1950_2025"
)

# Absolute temperature thresholds
COLD_THRESHOLDS = {
    "fd0": 0.0,
    "fd5": -5.0,
    "fd10": -10.0,
}

HOT_THRESHOLDS = {
    "hd30": 30.0,
    "hd35": 35.0,
    "hd40": 40.0,
}


# =============================================================================
# HELPERS
# =============================================================================

def sql_path(path: pathlib.Path) -> str:
    """
    Convert a path to a DuckDB-safe SQL string.
    """
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def expected_days(year: int) -> int:
    """
    Number of calendar days in a year.
    """
    return 366 if pd.Timestamp(year, 12, 31).is_leap_year else 365


def save_dataframe(
    df: pd.DataFrame,
    csv_path: pathlib.Path,
    parquet_path: pathlib.Path,
):
    """
    Save both CSV and Parquet.
    """

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    try:
        df.to_parquet(
            parquet_path,
            index=False,
        )

        print(f"Saved parquet: {parquet_path}")

    except Exception as exc:

        print(
            "\nWARNING: Parquet output failed."
        )

        print(
            "Install pyarrow if Parquet is needed:"
        )

        print(
            "    pip install pyarrow"
        )

        print(
            f"Reason: {exc}"
        )

    print(f"Saved CSV    : {csv_path}")


# =============================================================================
# BUILD DUCKDB DAILY VIEW
# =============================================================================

def create_daily_view(
    con: duckdb.DuckDBPyConnection,
    input_csv: pathlib.Path,
):
    """
    Create a lightweight DuckDB view over the large CSV.

    The full dataset is NOT loaded into pandas memory.
    """

    csv = sql_path(input_csv)

    print("\n" + "=" * 80)
    print("CREATING DAILY DATA VIEW")
    print("=" * 80)

    query = f"""
    CREATE OR REPLACE VIEW daily AS

    SELECT
        CAST(date_utc AS DATE) AS date,
        CAST(city_id AS BIGINT) AS city_id,
        CAST(city_name AS VARCHAR) AS city_name,

        TRY_CAST(t2m_min_mean AS DOUBLE) AS tmin,
        TRY_CAST(t2m_max_mean AS DOUBLE) AS tmax,
        TRY_CAST(t2m_mean_mean AS DOUBLE) AS tmean

    FROM read_csv_auto(
        '{csv}',
        header = true,
        all_varchar = true,
        sample_size = -1
    );
    """

    con.execute(query)

    summary = con.execute(
        """
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT city_id) AS n_cities,
            MIN(date) AS first_date,
            MAX(date) AS last_date
        FROM daily
        """
    ).df()

    print(summary.to_string(index=False))


# =============================================================================
# VALIDATE INPUT
# =============================================================================

def validate_input(
    con: duckdb.DuckDBPyConnection,
    start_year: int,
    end_year: int,
):
    """
    Basic input checks.
    """

    print("\n" + "=" * 80)
    print("VALIDATING DAILY DATA")
    print("=" * 80)

    validation = con.execute(
        f"""
        SELECT
            COUNT(*) AS total_rows,

            COUNT(DISTINCT city_id) AS cities,

            COUNT(DISTINCT date) AS unique_dates,

            MIN(date) AS first_date,

            MAX(date) AS last_date,

            SUM(
                CASE WHEN tmin IS NULL
                THEN 1 ELSE 0 END
            ) AS missing_tmin,

            SUM(
                CASE WHEN tmax IS NULL
                THEN 1 ELSE 0 END
            ) AS missing_tmax,

            SUM(
                CASE WHEN tmean IS NULL
                THEN 1 ELSE 0 END
            ) AS missing_tmean

        FROM daily

        WHERE EXTRACT(YEAR FROM date)
              BETWEEN {start_year} AND {end_year}
        """
    ).df()

    print(validation.to_string(index=False))

    duplicates = con.execute(
        f"""
        SELECT
            COUNT(*) AS duplicate_city_dates

        FROM (
            SELECT
                city_id,
                date,
                COUNT(*) AS n

            FROM daily

            WHERE EXTRACT(YEAR FROM date)
                  BETWEEN {start_year} AND {end_year}

            GROUP BY
                city_id,
                date

            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    print(
        f"\nDuplicate city-date combinations: "
        f"{duplicates:,}"
    )

    if duplicates > 0:

        raise ValueError(
            "Duplicate city-date rows detected. "
            "The city-day dataset must be unique."
        )


# =============================================================================
# CITY METADATA
# =============================================================================

def build_city_metadata(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Get unique city ID/name combinations.
    """

    print("\nBuilding city metadata...")

    df = con.execute(
        """
        SELECT
            city_id,
            MIN(city_name) AS city_name

        FROM daily

        GROUP BY city_id

        ORDER BY city_id
        """
    ).df()

    return df


# =============================================================================
# BASELINE CITY-SPECIFIC THRESHOLDS
# =============================================================================

def build_baseline_thresholds(
    con: duckdb.DuckDBPyConnection,
    baseline_start: int,
    baseline_end: int,
) -> pd.DataFrame:
    """
    Calculate city-specific baseline temperature percentiles.

    Tmax:
        P95
        P99

    Tmin:
        P05
        P01
    """

    print("\n" + "=" * 80)
    print("CALCULATING CITY-SPECIFIC BASELINE THRESHOLDS")
    print("=" * 80)

    print(
        f"Baseline period: "
        f"{baseline_start}-{baseline_end}"
    )

    query = f"""
    SELECT
        city_id,

        MIN(city_name) AS city_name,

        COUNT(tmax) AS baseline_n_tmax,
        COUNT(tmin) AS baseline_n_tmin,

        quantile_cont(
            tmax,
            0.95
        ) AS tmax_p95,

        quantile_cont(
            tmax,
            0.99
        ) AS tmax_p99,

        quantile_cont(
            tmin,
            0.05
        ) AS tmin_p05,

        quantile_cont(
            tmin,
            0.01
        ) AS tmin_p01

    FROM daily

    WHERE
        EXTRACT(YEAR FROM date)
        BETWEEN {baseline_start} AND {baseline_end}

    GROUP BY city_id

    ORDER BY city_id
    """

    thresholds = con.execute(query).df()

    print(
        f"Thresholds calculated for "
        f"{len(thresholds):,} cities."
    )

    return thresholds


# =============================================================================
# ANNUAL METRICS
# =============================================================================

def build_annual_metrics(
    con: duckdb.DuckDBPyConnection,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Calculate annual temperature metrics.
    """

    print("\n" + "=" * 80)
    print("BUILDING ANNUAL CITY METRICS")
    print("=" * 80)

    query = f"""
    SELECT
        city_id,

        MIN(city_name) AS city_name,

        CAST(
            EXTRACT(YEAR FROM date)
            AS INTEGER
        ) AS year,

        COUNT(*) AS n_calendar_rows,

        COUNT(tmin) AS n_tmin_valid,
        COUNT(tmax) AS n_tmax_valid,
        COUNT(tmean) AS n_tmean_valid,

        AVG(tmean) AS annual_tmean,
        AVG(tmin) AS annual_tmin,
        AVG(tmax) AS annual_tmax,

        MAX(tmax) AS txx,
        MIN(tmin) AS tnn,

        /* -------------------------------------------------
           Cold threshold counts
           ------------------------------------------------- */

        SUM(
            CASE WHEN tmin < 0
            THEN 1 ELSE 0 END
        ) AS fd0_count,

        SUM(
            CASE WHEN tmin < -5
            THEN 1 ELSE 0 END
        ) AS fd5_count,

        SUM(
            CASE WHEN tmin < -10
            THEN 1 ELSE 0 END
        ) AS fd10_count,

        /* -------------------------------------------------
           Hot threshold counts
           ------------------------------------------------- */

        SUM(
            CASE WHEN tmax > 30
            THEN 1 ELSE 0 END
        ) AS hd30_count,

        SUM(
            CASE WHEN tmax > 35
            THEN 1 ELSE 0 END
        ) AS hd35_count,

        SUM(
            CASE WHEN tmax > 40
            THEN 1 ELSE 0 END
        ) AS hd40_count

    FROM daily

    WHERE
        EXTRACT(YEAR FROM date)
        BETWEEN {start_year} AND {end_year}

    GROUP BY
        city_id,
        EXTRACT(YEAR FROM date)

    ORDER BY
        city_id,
        year
    """

    df = con.execute(query).df()

    # -------------------------------------------------------------------------
    # Convert threshold counts to percentage of valid observations
    # -------------------------------------------------------------------------

    cold_cols = [
        "fd0",
        "fd5",
        "fd10",
    ]

    hot_cols = [
        "hd30",
        "hd35",
        "hd40",
    ]

    for name in cold_cols:

        df[f"{name}_pct"] = (
            df[f"{name}_count"]
            /
            df["n_tmin_valid"].replace(0, np.nan)
            * 100
        )

    for name in hot_cols:

        df[f"{name}_pct"] = (
            df[f"{name}_count"]
            /
            df["n_tmax_valid"].replace(0, np.nan)
            * 100
        )

    # Expected number of days
    df["expected_days"] = (
        df["year"]
        .astype(int)
        .map(expected_days)
    )

    df["calendar_coverage_fraction"] = (
        df["n_calendar_rows"]
        /
        df["expected_days"]
    )

    df["tmin_valid_fraction"] = (
        df["n_tmin_valid"]
        /
        df["expected_days"]
    )

    df["tmax_valid_fraction"] = (
        df["n_tmax_valid"]
        /
        df["expected_days"]
    )

    df["tmean_valid_fraction"] = (
        df["n_tmean_valid"]
        /
        df["expected_days"]
    )

    print(
        f"Annual city-year rows: "
        f"{len(df):,}"
    )

    return df


# =============================================================================
# RELATIVE EXTREME METRICS
# =============================================================================

def build_relative_extremes(
    con: duckdb.DuckDBPyConnection,
    thresholds: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Calculate annual exceedance frequency relative to each city's
    baseline climatology.
    """

    print("\n" + "=" * 80)
    print("BUILDING RELATIVE EXTREME METRICS")
    print("=" * 80)

    # Register pandas threshold dataframe in DuckDB
    con.register(
        "thresholds_df",
        thresholds,
    )

    query = f"""
    SELECT
        d.city_id,

        CAST(
            EXTRACT(YEAR FROM d.date)
            AS INTEGER
        ) AS year,

        COUNT(d.tmax) AS n_tmax_relative_valid,
        COUNT(d.tmin) AS n_tmin_relative_valid,

        /* Tmax > city-specific P95 */

        SUM(
            CASE
                WHEN d.tmax > t.tmax_p95
                THEN 1
                ELSE 0
            END
        ) AS tx95p_count,

        /* Tmax > city-specific P99 */

        SUM(
            CASE
                WHEN d.tmax > t.tmax_p99
                THEN 1
                ELSE 0
            END
        ) AS tx99p_count,

        /* Tmin < city-specific P05 */

        SUM(
            CASE
                WHEN d.tmin < t.tmin_p05
                THEN 1
                ELSE 0
            END
        ) AS tn05p_count,

        /* Tmin < city-specific P01 */

        SUM(
            CASE
                WHEN d.tmin < t.tmin_p01
                THEN 1
                ELSE 0
            END
        ) AS tn01p_count

    FROM daily AS d

    INNER JOIN thresholds_df AS t
        ON d.city_id = t.city_id

    WHERE
        EXTRACT(YEAR FROM d.date)
        BETWEEN {start_year} AND {end_year}

    GROUP BY
        d.city_id,
        EXTRACT(YEAR FROM d.date)

    ORDER BY
        d.city_id,
        year
    """

    df = con.execute(query).df()

    df["tx95p_pct"] = (
        df["tx95p_count"]
        /
        df[
            "n_tmax_relative_valid"
        ].replace(0, np.nan)
        * 100
    )

    df["tx99p_pct"] = (
        df["tx99p_count"]
        /
        df[
            "n_tmax_relative_valid"
        ].replace(0, np.nan)
        * 100
    )

    df["tn05p_pct"] = (
        df["tn05p_count"]
        /
        df[
            "n_tmin_relative_valid"
        ].replace(0, np.nan)
        * 100
    )

    df["tn01p_pct"] = (
        df["tn01p_count"]
        /
        df[
            "n_tmin_relative_valid"
        ].replace(0, np.nan)
        * 100
    )

    return df


# =============================================================================
# SEASONAL METRICS
# =============================================================================

def build_regular_season(
    con: duckdb.DuckDBPyConnection,
    months: tuple[int, ...],
    label: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    MAM, JJA, or SON.

    Calendar year is also the season year.
    """

    months_sql = ",".join(
        str(m)
        for m in months
    )

    query = f"""
    SELECT
        city_id,

        CAST(
            EXTRACT(YEAR FROM date)
            AS INTEGER
        ) AS year,

        COUNT(tmin) AS {label}_n_tmin,
        COUNT(tmax) AS {label}_n_tmax,
        COUNT(tmean) AS {label}_n_tmean,

        AVG(tmin) AS {label}_tmin,
        AVG(tmax) AS {label}_tmax,
        AVG(tmean) AS {label}_tmean,

        MAX(tmax) AS {label}_txx,
        MIN(tmin) AS {label}_tnn

    FROM daily

    WHERE
        EXTRACT(MONTH FROM date)
        IN ({months_sql})

        AND

        EXTRACT(YEAR FROM date)
        BETWEEN {start_year} AND {end_year}

    GROUP BY
        city_id,
        EXTRACT(YEAR FROM date)

    ORDER BY
        city_id,
        year
    """

    return con.execute(query).df()


def build_djf(
    con: duckdb.DuckDBPyConnection,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Build DJF seasons correctly across calendar years.

    Example:
        Dec 2000 + Jan 2001 + Feb 2001
        -> winter 2001
    """

    print("Calculating DJF...")

    # Need previous December for the first complete winter.
    first_needed_year = start_year - 1

    query = f"""
    WITH winter_daily AS (

        SELECT
            city_id,
            date,
            tmin,
            tmax,
            tmean,

            CASE
                WHEN EXTRACT(MONTH FROM date) = 12
                THEN CAST(
                    EXTRACT(YEAR FROM date) + 1
                    AS INTEGER
                )

                ELSE CAST(
                    EXTRACT(YEAR FROM date)
                    AS INTEGER
                )
            END AS winter_year

        FROM daily

        WHERE
            EXTRACT(MONTH FROM date)
            IN (12, 1, 2)

            AND

            EXTRACT(YEAR FROM date)
            BETWEEN {first_needed_year}
                AND {end_year}
    )

    SELECT
        city_id,

        winter_year AS year,

        COUNT(tmin) AS djf_n_tmin,
        COUNT(tmax) AS djf_n_tmax,
        COUNT(tmean) AS djf_n_tmean,

        AVG(tmin) AS djf_tmin,
        AVG(tmax) AS djf_tmax,
        AVG(tmean) AS djf_tmean,

        MAX(tmax) AS djf_txx,
        MIN(tmin) AS djf_tnn,

        SUM(
            CASE WHEN tmin < 0
            THEN 1 ELSE 0 END
        ) AS djf_fd0_count,

        SUM(
            CASE WHEN tmin < -5
            THEN 1 ELSE 0 END
        ) AS djf_fd5_count,

        SUM(
            CASE WHEN tmin < -10
            THEN 1 ELSE 0 END
        ) AS djf_fd10_count

    FROM winter_daily

    WHERE
        winter_year BETWEEN
        {start_year} AND {end_year}

    GROUP BY
        city_id,
        winter_year

    ORDER BY
        city_id,
        year
    """

    df = con.execute(query).df()

    for name in (
        "djf_fd0",
        "djf_fd5",
        "djf_fd10",
    ):

        df[f"{name}_pct"] = (
            df[f"{name}_count"]
            /
            df["djf_n_tmin"].replace(
                0,
                np.nan,
            )
            * 100
        )

    return df


def build_seasonal_metrics(
    con: duckdb.DuckDBPyConnection,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Calculate DJF, MAM, JJA and SON.
    """

    print("\n" + "=" * 80)
    print("BUILDING SEASONAL METRICS")
    print("=" * 80)

    djf = build_djf(
        con,
        start_year,
        end_year,
    )

    print("Calculating MAM...")

    mam = build_regular_season(
        con,
        (3, 4, 5),
        "mam",
        start_year,
        end_year,
    )

    print("Calculating JJA...")

    jja = build_regular_season(
        con,
        (6, 7, 8),
        "jja",
        start_year,
        end_year,
    )

    print("Calculating SON...")

    son = build_regular_season(
        con,
        (9, 10, 11),
        "son",
        start_year,
        end_year,
    )

    seasonal = (
        djf
        .merge(
            mam,
            on=["city_id", "year"],
            how="outer",
        )
        .merge(
            jja,
            on=["city_id", "year"],
            how="outer",
        )
        .merge(
            son,
            on=["city_id", "year"],
            how="outer",
        )
    )

    return seasonal


# =============================================================================
# COMBINE EVERYTHING
# =============================================================================

def combine_metrics(
    annual: pd.DataFrame,
    relative: pd.DataFrame,
    seasonal: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all city-year metrics.
    """

    print("\n" + "=" * 80)
    print("COMBINING CITY-YEAR METRICS")
    print("=" * 80)

    threshold_cols = [
        "city_id",
        "tmax_p95",
        "tmax_p99",
        "tmin_p05",
        "tmin_p01",
    ]

    final = (
        annual
        .merge(
            relative,
            on=["city_id", "year"],
            how="left",
        )
        .merge(
            seasonal,
            on=["city_id", "year"],
            how="left",
        )
        .merge(
            thresholds[
                threshold_cols
            ],
            on="city_id",
            how="left",
        )
    )

    final = final.sort_values(
        [
            "city_id",
            "year",
        ]
    ).reset_index(drop=True)

    print(
        f"Final rows   : "
        f"{len(final):,}"
    )

    print(
        f"Final cities : "
        f"{final['city_id'].nunique():,}"
    )

    print(
        f"Years        : "
        f"{final['year'].min()}-"
        f"{final['year'].max()}"
    )

    return final


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def build_national_summary(
    city_year: pd.DataFrame,
) -> pd.DataFrame:
    """
    National summary based on the distribution across cities.

    IMPORTANT:
    This is NOT an area-weighted China mean.

    It describes the median / quartiles of city-level conditions.
    """

    metrics = [
        "annual_tmean",
        "annual_tmin",
        "annual_tmax",
        "djf_tmin",
        "jja_tmax",
        "fd0_pct",
        "fd5_pct",
        "fd10_pct",
        "hd35_pct",
        "hd40_pct",
        "tx95p_pct",
        "tx99p_pct",
    ]

    available = [
        x for x in metrics
        if x in city_year.columns
    ]

    pieces = []

    for year, group in city_year.groupby(
        "year"
    ):

        row = {
            "year": int(year),
            "n_cities": int(
                group["city_id"].nunique()
            ),
        }

        for metric in available:

            values = (
                group[metric]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
            )

            if len(values) == 0:

                row[
                    f"{metric}_median"
                ] = np.nan

                row[
                    f"{metric}_p25"
                ] = np.nan

                row[
                    f"{metric}_p75"
                ] = np.nan

            else:

                row[
                    f"{metric}_median"
                ] = values.median()

                row[
                    f"{metric}_p25"
                ] = values.quantile(0.25)

                row[
                    f"{metric}_p75"
                ] = values.quantile(0.75)

        pieces.append(row)

    return pd.DataFrame(pieces)


# =============================================================================
# PLOTS
# =============================================================================

def plot_city_temperature_summary(
    summary: pd.DataFrame,
    output_dir: pathlib.Path,
):
    """
    Plot three major city-level temperature indicators.
    """

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 10),
        sharex=True,
    )

    plot_specs = [
        (
            "annual_tmean",
            "Annual mean temperature",
        ),
        (
            "djf_tmin",
            "Winter (DJF) mean Tmin",
        ),
        (
            "jja_tmax",
            "Summer (JJA) mean Tmax",
        ),
    ]

    for ax, (
        metric,
        title,
    ) in zip(
        axes,
        plot_specs,
    ):

        median = (
            f"{metric}_median"
        )

        p25 = (
            f"{metric}_p25"
        )

        p75 = (
            f"{metric}_p75"
        )

        if median not in summary.columns:
            continue

        ax.plot(
            summary["year"],
            summary[median],
            linewidth=1.6,
        )

        ax.fill_between(
            summary["year"],
            summary[p25],
            summary[p75],
            alpha=0.20,
        )

        ax.set_ylabel("Temperature (°C)")

        ax.set_title(
            title,
            loc="left",
            fontsize=11,
        )

        ax.grid(
            alpha=0.2,
        )

    axes[-1].set_xlabel("Year")

    fig.suptitle(
        "Distribution of annual city temperature conditions",
        fontsize=14,
    )

    fig.text(
        0.5,
        0.01,
        (
            "Line = median across cities; "
            "shaded area = interquartile range (P25–P75)"
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=[0, 0.03, 1, 0.97]
    )

    path = (
        output_dir
        / "figure_01_city_temperature_summary.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved figure: {path}"
    )


def plot_cold_extremes(
    summary: pd.DataFrame,
    output_dir: pathlib.Path,
):
    """
    Plot median percentage of cold days across cities.
    """

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    specs = [
        (
            "fd0_pct_median",
            "Tmin < 0°C",
        ),
        (
            "fd5_pct_median",
            "Tmin < -5°C",
        ),
        (
            "fd10_pct_median",
            "Tmin < -10°C",
        ),
    ]

    for col, label in specs:

        if col in summary.columns:

            ax.plot(
                summary["year"],
                summary[col],
                linewidth=1.6,
                label=label,
            )

    ax.set_xlabel("Year")

    ax.set_ylabel(
        "Median percentage of days (%)"
    )

    ax.set_title(
        "Cold-day frequency across Chinese urban centres",
        loc="left",
    )

    ax.legend(
        frameon=False,
    )

    ax.grid(
        alpha=0.2,
    )

    fig.tight_layout()

    path = (
        output_dir
        / "figure_02_cold_extreme_frequency.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved figure: {path}"
    )


def plot_hot_extremes(
    summary: pd.DataFrame,
    output_dir: pathlib.Path,
):
    """
    Plot absolute and relative hot extremes.
    """

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    specs = [
        (
            "hd35_pct_median",
            "Tmax > 35°C",
        ),
        (
            "hd40_pct_median",
            "Tmax > 40°C",
        ),
        (
            "tx95p_pct_median",
            "Tmax > city P95",
        ),
        (
            "tx99p_pct_median",
            "Tmax > city P99",
        ),
    ]

    for col, label in specs:

        if col in summary.columns:

            ax.plot(
                summary["year"],
                summary[col],
                linewidth=1.6,
                label=label,
            )

    ax.set_xlabel("Year")

    ax.set_ylabel(
        "Median percentage of days (%)"
    )

    ax.set_title(
        "Hot-day frequency across Chinese urban centres",
        loc="left",
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    ax.grid(
        alpha=0.2,
    )

    fig.tight_layout()

    path = (
        output_dir
        / "figure_03_hot_extreme_frequency.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved figure: {path}"
    )


def plot_threshold_distributions(
    thresholds: pd.DataFrame,
    output_dir: pathlib.Path,
):
    """
    Show how city-specific extreme thresholds differ spatially
    between cities.

    This is useful before formal spatial mapping.
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
    )

    specs = [
        (
            "tmax_p95",
            "City Tmax P95",
        ),
        (
            "tmax_p99",
            "City Tmax P99",
        ),
        (
            "tmin_p05",
            "City Tmin P05",
        ),
        (
            "tmin_p01",
            "City Tmin P01",
        ),
    ]

    for ax, (
        column,
        title,
    ) in zip(
        axes.ravel(),
        specs,
    ):

        values = (
            thresholds[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        ax.hist(
            values,
            bins=35,
            edgecolor="white",
            linewidth=0.4,
        )

        median = values.median()

        ax.axvline(
            median,
            linestyle="--",
            linewidth=1.2,
        )

        ax.set_title(
            title,
            loc="left",
        )

        ax.set_xlabel("Temperature (°C)")

        ax.set_ylabel("Number of cities")

        ax.grid(
            axis="y",
            alpha=0.15,
        )

    fig.suptitle(
        (
            "Distribution of city-specific baseline "
            "temperature thresholds"
        ),
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    path = (
        output_dir
        / "figure_04_city_threshold_distributions.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved figure: {path}"
    )


def show_all_plots():
    """
    Display plots interactively after all files are saved.
    """

    print(
        "\nClose the figure windows "
        "to finish the script."
    )

    plt.show()


# =============================================================================
# QUALITY CONTROL
# =============================================================================

def make_qc_table(
    city_year: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise data completeness.
    """

    qc = (
        city_year
        .groupby("year")
        .agg(
            n_cities=(
                "city_id",
                "nunique",
            ),

            median_calendar_coverage=(
                "calendar_coverage_fraction",
                "median",
            ),

            min_calendar_coverage=(
                "calendar_coverage_fraction",
                "min",
            ),

            median_tmean_valid_fraction=(
                "tmean_valid_fraction",
                "median",
            ),

            min_tmean_valid_fraction=(
                "tmean_valid_fraction",
                "min",
            ),
        )
        .reset_index()
    )

    return qc


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
        help=(
            "Combined daily city temperature CSV."
        ),
    )

    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=END_YEAR,
    )

    parser.add_argument(
        "--baseline-start",
        type=int,
        default=BASELINE_START,
    )

    parser.add_argument(
        "--baseline-end",
        type=int,
        default=BASELINE_END,
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help=(
            "Save figures but do not open "
            "interactive figure windows."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Validate arguments
    # -------------------------------------------------------------------------

    if not args.input.exists():

        raise FileNotFoundError(
            f"Input file does not exist:\n"
            f"{args.input}"
        )

    if (
        args.start_year
        >
        args.end_year
    ):

        raise ValueError(
            "start year must be <= end year"
        )

    if (
        args.baseline_start
        >
        args.baseline_end
    ):

        raise ValueError(
            "baseline start must be <= baseline end"
        )

    if (
        args.baseline_start
        < args.start_year
        or
        args.baseline_end
        > args.end_year
    ):

        raise ValueError(
            "Baseline period must lie inside "
            "the requested analysis period."
        )

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 80)
    print("CITY-YEAR TEMPERATURE METRIC BUILDER")
    print("=" * 80)

    print(
        f"Input          : {args.input}"
    )

    print(
        f"Output         : {args.output}"
    )

    print(
        f"Period         : "
        f"{args.start_year}-"
        f"{args.end_year}"
    )

    print(
        f"Baseline       : "
        f"{args.baseline_start}-"
        f"{args.baseline_end}"
    )

    # -------------------------------------------------------------------------
    # DuckDB connection
    # -------------------------------------------------------------------------

    con = duckdb.connect(
        database=":memory:"
    )

    try:

        # =====================================================================
        # 1. DAILY VIEW
        # =====================================================================

        create_daily_view(
            con,
            args.input,
        )

        validate_input(
            con,
            args.start_year,
            args.end_year,
        )

        # =====================================================================
        # 2. CITY METADATA
        # =====================================================================

        city_metadata = (
            build_city_metadata(
                con
            )
        )

        city_metadata.to_csv(
            args.output
            / "city_metadata.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # =====================================================================
        # 3. BASELINE THRESHOLDS
        # =====================================================================

        thresholds = (
            build_baseline_thresholds(
                con,
                args.baseline_start,
                args.baseline_end,
            )
        )

        save_dataframe(
            thresholds,

            args.output
            / (
                f"city_baseline_thresholds_"
                f"{args.baseline_start}_"
                f"{args.baseline_end}.csv"
            ),

            args.output
            / (
                f"city_baseline_thresholds_"
                f"{args.baseline_start}_"
                f"{args.baseline_end}.parquet"
            ),
        )

        # =====================================================================
        # 4. ANNUAL METRICS
        # =====================================================================

        annual = (
            build_annual_metrics(
                con,
                args.start_year,
                args.end_year,
            )
        )

        # =====================================================================
        # 5. RELATIVE EXTREMES
        # =====================================================================

        relative = (
            build_relative_extremes(
                con,
                thresholds,
                args.start_year,
                args.end_year,
            )
        )

        # =====================================================================
        # 6. SEASONAL METRICS
        # =====================================================================

        seasonal = (
            build_seasonal_metrics(
                con,
                args.start_year,
                args.end_year,
            )
        )

        # =====================================================================
        # 7. COMBINE
        # =====================================================================

        city_year = (
            combine_metrics(
                annual,
                relative,
                seasonal,
                thresholds,
            )
        )

        final_csv = (
            args.output
            / (
                f"city_year_metrics_"
                f"{args.start_year}_"
                f"{args.end_year}.csv"
            )
        )

        final_parquet = (
            args.output
            / (
                f"city_year_metrics_"
                f"{args.start_year}_"
                f"{args.end_year}.parquet"
            )
        )

        save_dataframe(
            city_year,
            final_csv,
            final_parquet,
        )

        # =====================================================================
        # 8. QC
        # =====================================================================

        qc = make_qc_table(
            city_year
        )

        qc.to_csv(
            args.output
            / "city_year_data_qc.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # =====================================================================
        # 9. NATIONAL CITY DISTRIBUTION SUMMARY
        # =====================================================================

        summary = (
            build_national_summary(
                city_year
            )
        )

        summary.to_csv(
            args.output
            / "national_city_distribution_by_year.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # =====================================================================
        # 10. PLOTS
        # =====================================================================

        print("\n" + "=" * 80)
        print("GENERATING FIGURES")
        print("=" * 80)

        plot_city_temperature_summary(
            summary,
            args.output,
        )

        plot_cold_extremes(
            summary,
            args.output,
        )

        plot_hot_extremes(
            summary,
            args.output,
        )

        plot_threshold_distributions(
            thresholds,
            args.output,
        )

        # =====================================================================
        # 11. FINAL SUMMARY
        # =====================================================================

        print("\n" + "=" * 80)
        print("FINISHED")
        print("=" * 80)

        print(
            f"Cities: "
            f"{city_year['city_id'].nunique():,}"
        )

        print(
            f"City-year observations: "
            f"{len(city_year):,}"
        )

        print(
            "\nMain output:"
        )

        print(
            final_csv
        )

        print(
            "\nRecommended file for later analysis:"
        )

        print(
            final_parquet
        )

        print(
            "\nNext scripts should use this "
            "city-year dataset rather than "
            "re-reading the large daily CSV."
        )

        if not args.no_show:

            show_all_plots()

    finally:

        con.close()


if __name__ == "__main__":

    main()