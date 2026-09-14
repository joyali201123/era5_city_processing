"""
03_spatial_analysis.py

Spatial analysis of city-level ERA5 temperature changes.

INPUTS
------
1. city_temporal_trends_wide.csv
   produced by 02_temporal_analysis.py

2. city_year_metrics_1950_2025.csv
   produced by 01_build_city_year_metrics.py

3. GHSL Urban Centre Database geopackage

MAIN ANALYSES
-------------
A. Spatial distribution of temporal trends
   - annual Tmean trend
   - winter DJF Tmin trend
   - summer JJA Tmax trend
   - Tmin < -5 C frequency trend
   - city-specific P99 hot-day trend

B. Spatial autocorrelation
   - k-nearest-neighbour spatial weights
   - Global Moran's I
   - permutation significance
   - Local Moran's I / LISA
   - FDR correction of local p-values

C. Geographic aggregation
   - latitude-band comparison
   - median / IQR by latitude band

D. Data-driven city typology
   - long-term thermal characteristics
   - temporal trend characteristics
   - standardisation
   - PCA using NumPy SVD
   - K-means using SciPy
   - cluster map
   - cluster profile heatmap

NO EXTRA PACKAGES REQUIRED
--------------------------
numpy
pandas
scipy
matplotlib
geopandas
shapely

OUTPUTS
-------
CSV tables + PNG figures.

Run
---
python 03_spatial_analysis.py
"""

from __future__ import annotations

import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from scipy import sparse
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2


# =============================================================================
# SETTINGS
# =============================================================================

ROOT = pathlib.Path(
    __file__
).resolve().parent


# -----------------------------------------------------------------------------
# INPUTS
# -----------------------------------------------------------------------------

DEFAULT_TRENDS = (
    ROOT
    / "results_temporal_analysis_1950_2025"
    / "city_temporal_trends_wide.csv"
)

DEFAULT_CITY_YEAR = (
    ROOT
    / "results_city_year_metrics_1950_2025"
    / "city_year_metrics_1950_2025.csv"
)


# -----------------------------------------------------------------------------
# GHSL
# -----------------------------------------------------------------------------

LAYER = (
    "GHSL_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"
)

DEFAULT_GPKG = (
    ROOT.parent
    / "GHS_UCDB_REGION_EASTERN_AND_SOUTH_EASTERN_ASIA_R2024A_V1_2"
    / "GHS_UCDB_REGION_EASTERN_AND_SOUTH_EASTERN_ASIA_R2024A.gpkg"
)


# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT = (
    ROOT
    / "results_spatial_analysis_1950_2025"
)


# -----------------------------------------------------------------------------
# SPATIAL STATISTICS
# -----------------------------------------------------------------------------

K_NEIGHBORS = 8

GLOBAL_PERMUTATIONS = 999

LOCAL_PERMUTATIONS = 999

ALPHA = 0.05


# -----------------------------------------------------------------------------
# CLUSTERING
# -----------------------------------------------------------------------------

N_CLUSTERS = 5

KMEANS_RESTARTS = 20

RANDOM_SEED = 42


# =============================================================================
# CORE SPATIAL METRICS
# =============================================================================

SPATIAL_METRICS = {

    "annual_tmean_slope_decade": {
        "label":
            "Annual mean temperature",

        "unit":
            "°C/decade",
    },

    "djf_tmin_slope_decade": {
        "label":
            "Winter (DJF) Tmin",

        "unit":
            "°C/decade",
    },

    "jja_tmax_slope_decade": {
        "label":
            "Summer (JJA) Tmax",

        "unit":
            "°C/decade",
    },

    "fd5_pct_slope_decade": {
        "label":
            "Tmin < -5°C frequency",

        "unit":
            "percentage points/decade",
    },

    "tx99p_pct_slope_decade": {
        "label":
            "Tmax > city P99 frequency",

        "unit":
            "percentage points/decade",
    },
}


# =============================================================================
# CLUSTER VARIABLES
# =============================================================================

CLIMATOLOGY_METRICS = {

    "annual_tmean":
        "Long-term annual Tmean",

    "djf_tmin":
        "Long-term winter Tmin",

    "jja_tmax":
        "Long-term summer Tmax",

    "fd5_pct":
        "Long-term Tmin < -5°C",

    "hd35_pct":
        "Long-term Tmax > 35°C",

    "tx99p_pct":
        "Long-term Tmax > city P99",
}


TREND_CLUSTER_METRICS = {

    "annual_tmean_slope_decade":
        "Annual Tmean trend",

    "djf_tmin_slope_decade":
        "Winter Tmin trend",

    "jja_tmax_slope_decade":
        "Summer Tmax trend",

    "fd5_pct_slope_decade":
        "Tmin < -5°C trend",

    "tx99p_pct_slope_decade":
        "P99 hot-day trend",
}


# =============================================================================
# HELPERS
# =============================================================================

def save_csv(
    df,
    path,
):

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved: {path}"
    )


def benjamini_hochberg(
    pvalues,
):

    p = np.asarray(
        pvalues,
        dtype=float,
    )

    adjusted = np.full(
        len(p),
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(
        p
    )

    pv = p[
        valid
    ]

    if len(pv) == 0:
        return adjusted

    order = np.argsort(
        pv
    )

    ranked = pv[
        order
    ]

    n = len(
        ranked
    )

    q = (
        ranked
        * n
        /
        np.arange(
            1,
            n + 1,
        )
    )

    q = (
        np.minimum.accumulate(
            q[::-1]
        )[::-1]
    )

    q = np.clip(
        q,
        0,
        1,
    )

    restored = np.empty_like(
        q
    )

    restored[
        order
    ] = q

    adjusted[
        valid
    ] = restored

    return adjusted


# =============================================================================
# LOAD TEMPORAL TRENDS
# =============================================================================

def load_trends(
    path,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LOADING TEMPORAL TREND RESULTS"
    )

    print(
        "=" * 80
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Trend file does not exist:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "city_id",
        "city_name",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    df["city_id"] = (
        pd.to_numeric(
            df[
                "city_id"
            ],
            errors="coerce",
        )
    )

    df = df.dropna(
        subset=[
            "city_id"
        ]
    )

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

    print(
        f"Cities in trend table: "
        f"{len(df):,}"
    )

    for metric in SPATIAL_METRICS:

        if metric not in df.columns:

            print(
                f"WARNING: missing metric: "
                f"{metric}"
            )

    return df


# =============================================================================
# LOAD CITY-YEAR METRICS
# =============================================================================

def load_city_year(
    path,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LOADING CITY-YEAR METRICS"
    )

    print(
        "=" * 80
    )

    if not path.exists():

        raise FileNotFoundError(
            f"City-year file does not exist:\n"
            f"{path}"
        )

    wanted = {
        "city_id",
        "city_name",
        "year",
        *CLIMATOLOGY_METRICS.keys(),
    }

    df = pd.read_csv(
        path,
        usecols=lambda c:
            c in wanted,
    )

    df[
        "city_id"
    ] = pd.to_numeric(
        df[
            "city_id"
        ],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "city_id"
        ]
    )

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

    print(
        f"Rows   : {len(df):,}"
    )

    print(
        f"Cities : "
        f"{df['city_id'].nunique():,}"
    )

    if "year" in df.columns:

        print(
            f"Years  : "
            f"{df['year'].min()}-"
            f"{df['year'].max()}"
        )

    return df


# =============================================================================
# LONG-TERM CITY CLIMATOLOGY
# =============================================================================

def build_city_climatology(
    city_year,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BUILDING LONG-TERM CITY PROFILES"
    )

    print(
        "=" * 80
    )

    available = [
        x
        for x
        in CLIMATOLOGY_METRICS
        if x
        in city_year.columns
    ]

    grouped = (
        city_year
        .groupby(
            "city_id"
        )[
            available
        ]
        .median()
        .reset_index()
    )

    rename = {
        metric:
            f"clim_{metric}"
        for metric
        in available
    }

    grouped = grouped.rename(
        columns=rename
    )

    print(
        f"Long-term profiles: "
        f"{len(grouped):,} cities"
    )

    return grouped


# =============================================================================
# LOAD GHSL
# =============================================================================

def load_ghsl(
    path,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LOADING GHSL CHINA URBAN CENTRES"
    )

    print(
        "=" * 80
    )

    if not path.exists():

        raise FileNotFoundError(
            f"GHSL file does not exist:\n"
            f"{path}"
        )

    cities = gpd.read_file(
        path,
        layer=LAYER,
    )

    required = {
        "ID_UC_G0",
        "GC_UCN_MAI_2025",
        "GC_CNT_GAD_2025",
        "geometry",
    }

    missing = (
        required
        - set(
            cities.columns
        )
    )

    if missing:

        raise ValueError(
            f"Missing GHSL columns: "
            f"{sorted(missing)}"
        )

    cities = (
        cities.loc[
            cities[
                "GC_CNT_GAD_2025"
            ]
            == "China"
        ]
        .copy()
        .to_crs(
            4326
        )
    )

    cities[
        "city_id"
    ] = pd.to_numeric(
        cities[
            "ID_UC_G0"
        ],
        errors="coerce",
    )

    cities = cities.dropna(
        subset=[
            "city_id"
        ]
    )

    cities[
        "city_id"
    ] = (
        cities[
            "city_id"
        ]
        .astype(
            int
        )
    )

    if cities[
        "city_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate GHSL city IDs found."
        )

    print(
        f"GHSL China urban centres: "
        f"{len(cities):,}"
    )

    return cities


# =============================================================================
# JOIN DATA
# =============================================================================

def build_spatial_table(
    ghsl,
    trends,
    climatology,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "JOINING CLIMATE RESULTS TO GHSL"
    )

    print(
        "=" * 80
    )

    gdf = (
        ghsl
        .merge(
            trends,
            on="city_id",
            how="inner",
            suffixes=(
                "_ghsl",
                "_trend",
            ),
        )
        .merge(
            climatology,
            on="city_id",
            how="left",
        )
    )

    # -------------------------------------------------------------------------
    # Representative point
    # -------------------------------------------------------------------------

    points = (
        gdf[
            "geometry"
        ]
        .representative_point()
    )

    gdf[
        "longitude"
    ] = points.x

    gdf[
        "latitude"
    ] = points.y

    print(
        f"Matched cities: "
        f"{len(gdf):,}"
    )

    print(
        f"Trend table cities: "
        f"{len(trends):,}"
    )

    if len(gdf) < len(
        trends
    ):

        print(
            "WARNING: some trend-table cities "
            "were not matched to GHSL."
        )

    return gdf


# =============================================================================
# LATITUDE BANDS
# =============================================================================

def add_latitude_bands(
    gdf,
):

    bins = [
        -90,
        25,
        30,
        35,
        40,
        45,
        90,
    ]

    labels = [
        "<25°N",
        "25–30°N",
        "30–35°N",
        "35–40°N",
        "40–45°N",
        "≥45°N",
    ]

    gdf[
        "latitude_band"
    ] = pd.cut(
        gdf[
            "latitude"
        ],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    return gdf


# =============================================================================
# SPHERICAL COORDINATES
# =============================================================================

def lonlat_to_xyz(
    lon,
    lat,
):

    lon_rad = np.radians(
        lon
    )

    lat_rad = np.radians(
        lat
    )

    x = (
        np.cos(
            lat_rad
        )
        *
        np.cos(
            lon_rad
        )
    )

    y = (
        np.cos(
            lat_rad
        )
        *
        np.sin(
            lon_rad
        )
    )

    z = np.sin(
        lat_rad
    )

    return np.column_stack(
        [
            x,
            y,
            z,
        ]
    )


# =============================================================================
# KNN SPATIAL WEIGHTS
# =============================================================================

def build_knn_weights(
    lon,
    lat,
    k=8,
):

    print(
        "\nBuilding "
        f"{k}-nearest-neighbour "
        "spatial weights..."
    )

    xyz = lonlat_to_xyz(
        lon,
        lat,
    )

    n = len(
        xyz
    )

    if n <= k:

        raise ValueError(
            "Number of cities must be "
            "larger than k."
        )

    tree = cKDTree(
        xyz
    )

    distances, neighbors = (
        tree.query(
            xyz,
            k=k + 1,
        )
    )

    # First neighbour is self
    neighbors = neighbors[
        :,
        1:
    ]

    rows = np.repeat(
        np.arange(
            n
        ),
        k,
    )

    cols = neighbors.ravel()

    data = np.ones(
        len(rows),
        dtype=float,
    )

    W = sparse.csr_matrix(
        (
            data,
            (
                rows,
                cols,
            ),
        ),
        shape=(
            n,
            n,
        ),
    )

    # -------------------------------------------------------------------------
    # Make symmetric
    # -------------------------------------------------------------------------

    W = W.maximum(
        W.T
    )

    W.setdiag(
        0
    )

    W.eliminate_zeros()

    # -------------------------------------------------------------------------
    # Row-standardise
    # -------------------------------------------------------------------------

    row_sum = np.asarray(
        W.sum(
            axis=1
        )
    ).ravel()

    inverse = np.zeros_like(
        row_sum
    )

    valid = (
        row_sum
        > 0
    )

    inverse[
        valid
    ] = (
        1
        /
        row_sum[
            valid
        ]
    )

    D = sparse.diags(
        inverse
    )

    W = (
        D
        @ W
    ).tocsr()

    print(
        f"Spatial weights created "
        f"for {n:,} cities."
    )

    return W


# =============================================================================
# SUBSET WEIGHTS
# =============================================================================

def subset_weights(
    W,
    indices,
):

    W_sub = (
        W[
            indices,
            :
        ][
            :,
            indices
        ]
        .tocsr()
    )

    row_sum = np.asarray(
        W_sub.sum(
            axis=1
        )
    ).ravel()

    inverse = np.zeros_like(
        row_sum
    )

    valid = (
        row_sum
        > 0
    )

    inverse[
        valid
    ] = (
        1
        /
        row_sum[
            valid
        ]
    )

    W_sub = (
        sparse.diags(
            inverse
        )
        @ W_sub
    ).tocsr()

    return W_sub


# =============================================================================
# GLOBAL MORAN'S I
# =============================================================================

def moran_global(
    values,
    W,
    permutations=999,
    seed=42,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    valid = np.isfinite(
        x
    )

    indices = np.flatnonzero(
        valid
    )

    x = x[
        valid
    ]

    Wv = subset_weights(
        W,
        indices,
    )

    n = len(
        x
    )

    if n < 10:

        return {
            "n":
                n,

            "moran_i":
                np.nan,

            "p_permutation":
                np.nan,
        }

    z = (
        x
        -
        np.mean(
            x
        )
    )

    denominator = np.sum(
        z ** 2
    )

    S0 = Wv.sum()

    if (
        denominator <= 0
        or
        S0 <= 0
    ):

        return {
            "n":
                n,

            "moran_i":
                np.nan,

            "p_permutation":
                np.nan,
        }

    observed = (
        n
        /
        S0
        *
        (
            z
            @ (
                Wv
                @ z
            )
        )
        /
        denominator
    )

    rng = np.random.default_rng(
        seed
    )

    simulated = np.empty(
        permutations,
        dtype=float,
    )

    for p in range(
        permutations
    ):

        zp = rng.permutation(
            z
        )

        simulated[
            p
        ] = (
            n
            /
            S0
            *
            (
                zp
                @ (
                    Wv
                    @ zp
                )
            )
            /
            denominator
        )

    p_value = (
        1
        +
        np.sum(
            np.abs(
                simulated
            )
            >=
            abs(
                observed
            )
        )
    ) / (
        permutations
        + 1
    )

    return {
        "n":
            int(
                n
            ),

        "moran_i":
            float(
                observed
            ),

        "p_permutation":
            float(
                p_value
            ),
    }


# =============================================================================
# LOCAL MORAN / LISA
# =============================================================================

def moran_local(
    values,
    W,
    permutations=999,
    seed=42,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    valid = np.isfinite(
        x
    )

    indices = np.flatnonzero(
        valid
    )

    xv = x[
        valid
    ]

    Wv = subset_weights(
        W,
        indices,
    )

    n = len(
        xv
    )

    result = pd.DataFrame(
        {
            "row_index":
                np.arange(
                    len(x)
                ),

            "local_i":
                np.nan,

            "local_p":
                np.nan,

            "local_p_fdr":
                np.nan,

            "spatial_lag_z":
                np.nan,

            "lisa_class":
                "Missing",

            "lisa_class_fdr":
                "Missing",
        }
    )

    if n < 10:

        return result

    z = (
        xv
        -
        np.mean(
            xv
        )
    )

    m2 = np.mean(
        z ** 2
    )

    if m2 <= 0:

        return result

    lag = (
        Wv
        @ z
    )

    local_i = (
        z
        *
        lag
        /
        m2
    )

    rng = np.random.default_rng(
        seed
    )

    extreme_count = np.zeros(
        n,
        dtype=int,
    )

    for _ in range(
        permutations
    ):

        zp = rng.permutation(
            z
        )

        lag_p = (
            Wv
            @ zp
        )

        ip = (
            zp
            *
            lag_p
            /
            m2
        )

        extreme_count += (
            np.abs(
                ip
            )
            >=
            np.abs(
                local_i
            )
        )

    p = (
        1
        +
        extreme_count
    ) / (
        permutations
        + 1
    )

    p_fdr = benjamini_hochberg(
        p
    )

    classes = np.full(
        n,
        "Not significant",
        dtype=object,
    )

    classes_fdr = np.full(
        n,
        "Not significant",
        dtype=object,
    )

    # -------------------------------------------------------------------------
    # Raw p classification
    # -------------------------------------------------------------------------

    sig = (
        p
        < ALPHA
    )

    hh = (
        sig
        &
        (z > 0)
        &
        (lag > 0)
    )

    ll = (
        sig
        &
        (z < 0)
        &
        (lag < 0)
    )

    hl = (
        sig
        &
        (z > 0)
        &
        (lag < 0)
    )

    lh = (
        sig
        &
        (z < 0)
        &
        (lag > 0)
    )

    classes[
        hh
    ] = "High-High"

    classes[
        ll
    ] = "Low-Low"

    classes[
        hl
    ] = "High-Low"

    classes[
        lh
    ] = "Low-High"

    # -------------------------------------------------------------------------
    # FDR classification
    # -------------------------------------------------------------------------

    sig_fdr = (
        p_fdr
        < ALPHA
    )

    hh = (
        sig_fdr
        &
        (z > 0)
        &
        (lag > 0)
    )

    ll = (
        sig_fdr
        &
        (z < 0)
        &
        (lag < 0)
    )

    hl = (
        sig_fdr
        &
        (z > 0)
        &
        (lag < 0)
    )

    lh = (
        sig_fdr
        &
        (z < 0)
        &
        (lag > 0)
    )

    classes_fdr[
        hh
    ] = "High-High"

    classes_fdr[
        ll
    ] = "Low-Low"

    classes_fdr[
        hl
    ] = "High-Low"

    classes_fdr[
        lh
    ] = "Low-High"

    # -------------------------------------------------------------------------
    # Put back into complete table
    # -------------------------------------------------------------------------

    result.loc[
        indices,
        "local_i",
    ] = local_i

    result.loc[
        indices,
        "local_p",
    ] = p

    result.loc[
        indices,
        "local_p_fdr",
    ] = p_fdr

    result.loc[
        indices,
        "spatial_lag_z",
    ] = lag

    result.loc[
        indices,
        "lisa_class",
    ] = classes

    result.loc[
        indices,
        "lisa_class_fdr",
    ] = classes_fdr

    return result


# =============================================================================
# RUN SPATIAL AUTOCORRELATION
# =============================================================================

def spatial_autocorrelation(
    gdf,
    W,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SPATIAL AUTOCORRELATION"
    )

    print(
        "=" * 80
    )

    rows = []

    lisa_outputs = {}

    for number, (
        metric,
        info,
    ) in enumerate(
        SPATIAL_METRICS.items(),
        start=1,
    ):

        if metric not in gdf.columns:
            continue

        print(
            f"\n{info['label']}"
        )

        global_result = (
            moran_global(
                gdf[
                    metric
                ].to_numpy(),
                W,
                permutations=(
                    GLOBAL_PERMUTATIONS
                ),
                seed=(
                    RANDOM_SEED
                    + number
                ),
            )
        )

        rows.append(
            {
                "metric":
                    metric,

                "label":
                    info[
                        "label"
                    ],

                "unit":
                    info[
                        "unit"
                    ],

                **global_result,
            }
        )

        print(
            f"Global Moran's I = "
            f"{global_result['moran_i']:.4f}"
        )

        print(
            f"Permutation p    = "
            f"{global_result['p_permutation']:.4g}"
        )

        lisa = moran_local(
            gdf[
                metric
            ].to_numpy(),
            W,
            permutations=(
                LOCAL_PERMUTATIONS
            ),
            seed=(
                RANDOM_SEED
                + 100
                + number
            ),
        )

        lisa_outputs[
            metric
        ] = lisa

    return (
        pd.DataFrame(
            rows
        ),
        lisa_outputs,
    )


# =============================================================================
# ADD LISA RESULTS TO CITY TABLE
# =============================================================================

def attach_lisa(
    gdf,
    lisa_outputs,
):

    result = gdf.copy()

    for metric, lisa in (
        lisa_outputs.items()
    ):

        prefix = metric.replace(
            "_slope_decade",
            "",
        )

        result[
            f"{prefix}_local_moran_i"
        ] = lisa[
            "local_i"
        ].to_numpy()

        result[
            f"{prefix}_local_p"
        ] = lisa[
            "local_p"
        ].to_numpy()

        result[
            f"{prefix}_local_p_fdr"
        ] = lisa[
            "local_p_fdr"
        ].to_numpy()

        result[
            f"{prefix}_lisa_class"
        ] = lisa[
            "lisa_class_fdr"
        ].to_numpy()

    return result


# =============================================================================
# LATITUDE-BAND SUMMARY
# =============================================================================

def latitude_band_summary(
    gdf,
):

    rows = []

    for band, group in gdf.groupby(
        "latitude_band",
        observed=True,
    ):

        for metric, info in (
            SPATIAL_METRICS.items()
        ):

            if metric not in group.columns:
                continue

            x = (
                group[
                    metric
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

            if len(x) == 0:
                continue

            rows.append(
                {
                    "latitude_band":
                        str(
                            band
                        ),

                    "metric":
                        metric,

                    "label":
                        info[
                            "label"
                        ],

                    "n_cities":
                        len(
                            x
                        ),

                    "mean":
                        x.mean(),

                    "median":
                        x.median(),

                    "p25":
                        x.quantile(
                            0.25
                        ),

                    "p75":
                        x.quantile(
                            0.75
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# PREPARE CLUSTER FEATURES
# =============================================================================

def prepare_cluster_features(
    gdf,
):

    feature_columns = []

    feature_labels = []

    # -------------------------------------------------------------------------
    # Long-term climatology
    # -------------------------------------------------------------------------

    for metric, label in (
        CLIMATOLOGY_METRICS.items()
    ):

        column = (
            f"clim_{metric}"
        )

        if column in gdf.columns:

            feature_columns.append(
                column
            )

            feature_labels.append(
                label
            )

    # -------------------------------------------------------------------------
    # Temporal trends
    # -------------------------------------------------------------------------

    for metric, label in (
        TREND_CLUSTER_METRICS.items()
    ):

        if metric in gdf.columns:

            feature_columns.append(
                metric
            )

            feature_labels.append(
                label
            )

    if len(
        feature_columns
    ) < 4:

        raise ValueError(
            "Too few cluster features."
        )

    X = (
        gdf[
            feature_columns
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .copy()
    )

    # -------------------------------------------------------------------------
    # Median imputation
    # -------------------------------------------------------------------------

    for column in feature_columns:

        median = X[
            column
        ].median()

        X[
            column
        ] = (
            X[
                column
            ]
            .fillna(
                median
            )
        )

    # -------------------------------------------------------------------------
    # Standardise
    # -------------------------------------------------------------------------

    means = X.mean(
        axis=0
    )

    stds = X.std(
        axis=0,
        ddof=0,
    )

    stds = stds.replace(
        0,
        1,
    )

    Z = (
        X
        -
        means
    ) / stds

    return (
        X,
        Z,
        feature_columns,
        feature_labels,
    )


# =============================================================================
# PCA
# =============================================================================

def run_pca(
    Z,
):

    matrix = Z.to_numpy(
        dtype=float
    )

    # Already centred because Z is standardised
    U, S, Vt = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    scores = (
        U
        *
        S
    )

    variance = (
        S ** 2
    )

    explained = (
        variance
        /
        variance.sum()
    )

    cumulative = np.cumsum(
        explained
    )

    # Use enough PCs to explain at least 80%,
    # but not more than 5 PCs.
    threshold = np.flatnonzero(
        cumulative
        >= 0.80
    )

    if len(
        threshold
    ) > 0:

        n_pc = (
            int(
                threshold[
                    0
                ]
            )
            + 1
        )

    else:

        n_pc = 5

    n_pc = max(
        2,
        min(
            n_pc,
            5,
            scores.shape[
                1
            ],
        ),
    )

    return (
        scores,
        explained,
        Vt,
        n_pc,
    )


# =============================================================================
# K-MEANS
# =============================================================================

def robust_kmeans(
    X,
    k,
    restarts=20,
    seed=42,
):

    best_labels = None

    best_centroids = None

    best_sse = np.inf

    for run in range(
        restarts
    ):

        np.random.seed(
            seed
            + run
        )

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            centroids, labels = (
                kmeans2(
                    X,
                    k,
                    iter=100,
                    minit="++",
                )
            )

        residual = (
            X
            -
            centroids[
                labels
            ]
        )

        sse = np.sum(
            residual ** 2
        )

        if sse < best_sse:

            best_sse = sse

            best_centroids = (
                centroids.copy()
            )

            best_labels = (
                labels.copy()
            )

    return (
        best_centroids,
        best_labels,
        best_sse,
    )


# =============================================================================
# CITY CLUSTERING
# =============================================================================

def cluster_cities(
    gdf,
    n_clusters,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DATA-DRIVEN CITY CLASSIFICATION"
    )

    print(
        "=" * 80
    )

    (
        raw,
        Z,
        feature_columns,
        feature_labels,
    ) = prepare_cluster_features(
        gdf
    )

    (
        scores,
        explained,
        loadings,
        n_pc,
    ) = run_pca(
        Z
    )

    print(
        f"Cluster features : "
        f"{len(feature_columns)}"
    )

    print(
        f"PCA components   : "
        f"{n_pc}"
    )

    print(
        f"Variance retained: "
        f"{explained[:n_pc].sum() * 100:.1f}%"
    )

    clustering_matrix = scores[
        :,
        :n_pc
    ]

    (
        centroids,
        labels,
        sse,
    ) = robust_kmeans(
        clustering_matrix,
        n_clusters,
        restarts=(
            KMEANS_RESTARTS
        ),
        seed=(
            RANDOM_SEED
        ),
    )

    # -------------------------------------------------------------------------
    # Reorder clusters from colder to warmer based on annual Tmean
    # -------------------------------------------------------------------------

    if (
        "clim_annual_tmean"
        in raw.columns
    ):

        temp = raw[
            "clim_annual_tmean"
        ].to_numpy()

        cluster_mean_temp = {
            cluster:
                np.mean(
                    temp[
                        labels
                        == cluster
                    ]
                )
            for cluster
            in np.unique(
                labels
            )
        }

        ordered = sorted(
            cluster_mean_temp,
            key=cluster_mean_temp.get,
        )

        mapping = {
            old:
                new + 1
            for new, old
            in enumerate(
                ordered
            )
        }

    else:

        mapping = {
            old:
                old + 1
            for old
            in np.unique(
                labels
            )
        }

    final_labels = np.array(
        [
            mapping[
                x
            ]
            for x
            in labels
        ],
        dtype=int,
    )

    result = gdf.copy()

    result[
        "thermal_cluster"
    ] = final_labels

    # -------------------------------------------------------------------------
    # PCA scores
    # -------------------------------------------------------------------------

    pca_table = pd.DataFrame(
        {
            "city_id":
                gdf[
                    "city_id"
                ].to_numpy()
        }
    )

    for i in range(
        min(
            5,
            scores.shape[
                1
            ],
        )
    ):

        pca_table[
            f"PC{i + 1}"
        ] = scores[
            :,
            i
        ]

    pca_table[
        "thermal_cluster"
    ] = final_labels

    # -------------------------------------------------------------------------
    # Cluster z-score profiles
    # -------------------------------------------------------------------------

    Z_with_cluster = (
        Z.copy()
    )

    Z_with_cluster[
        "thermal_cluster"
    ] = final_labels

    profile_z = (
        Z_with_cluster
        .groupby(
            "thermal_cluster"
        )
        .mean()
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Raw profiles
    # -------------------------------------------------------------------------

    raw_with_cluster = (
        raw.copy()
    )

    raw_with_cluster[
        "thermal_cluster"
    ] = final_labels

    profile_raw = (
        raw_with_cluster
        .groupby(
            "thermal_cluster"
        )
        .mean()
        .reset_index()
    )

    counts = (
        pd.Series(
            final_labels
        )
        .value_counts()
        .sort_index()
    )

    print(
        "\nCluster sizes:"
    )

    for cluster, count in (
        counts.items()
    ):

        print(
            f"  Cluster {cluster}: "
            f"{count:,} cities"
        )

    return {
        "gdf":
            result,

        "pca":
            pca_table,

        "profile_z":
            profile_z,

        "profile_raw":
            profile_raw,

        "feature_columns":
            feature_columns,

        "feature_labels":
            feature_labels,

        "explained":
            explained,

        "loadings":
            loadings,

        "n_pc":
            n_pc,

        "sse":
            sse,
    }


# =============================================================================
# MAP HELPER
# =============================================================================

def set_china_map_extent(
    ax,
    gdf,
):

    xmin = gdf[
        "longitude"
    ].min()

    xmax = gdf[
        "longitude"
    ].max()

    ymin = gdf[
        "latitude"
    ].min()

    ymax = gdf[
        "latitude"
    ].max()

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
# FIGURE 1 — TREND MAPS
# =============================================================================

def plot_trend_maps(
    gdf,
    output,
):

    metrics = [
        x
        for x
        in SPATIAL_METRICS
        if x
        in gdf.columns
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            16,
            10,
        ),
    )

    axes = axes.ravel()

    for ax, metric in zip(
        axes,
        metrics,
    ):

        values = gdf[
            metric
        ].to_numpy(
            dtype=float
        )

        finite = np.isfinite(
            values
        )

        if not finite.any():
            continue

        # Symmetric colour range for trends
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
            gdf.loc[
                finite,
                "longitude",
            ],
            gdf.loc[
                finite,
                "latitude",
            ],
            c=values[
                finite
            ],
            s=12,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            linewidths=0,
        )

        info = SPATIAL_METRICS[
            metric
        ]

        ax.set_title(
            (
                f"{info['label']}\n"
                f"({info['unit']})"
            ),
            loc="left",
            fontsize=10,
        )

        set_china_map_extent(
            ax,
            gdf,
        )

        cbar = fig.colorbar(
            sc,
            ax=ax,
            shrink=0.75,
        )

        cbar.set_label(
            info[
                "unit"
            ]
        )

    # Remove unused panel
    for ax in axes[
        len(metrics):
    ]:

        ax.axis(
            "off"
        )

    fig.suptitle(
        (
            "Spatial distribution of city-level "
            "temperature trends"
        ),
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
        "figure_01_city_temperature_trend_maps.png"
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
# FIGURE 2 — GLOBAL MORAN
# =============================================================================

def plot_global_moran(
    table,
    output,
):

    if table.empty:
        return

    fig, ax = plt.subplots(
        figsize=(
            9,
            5.5,
        )
    )

    labels = table[
        "label"
    ].tolist()

    y = np.arange(
        len(
            table
        )
    )

    ax.barh(
        y,
        table[
            "moran_i"
        ],
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Global Moran's I"
    )

    ax.set_title(
        (
            "Spatial autocorrelation of "
            "city temperature trends"
        ),
        loc="left",
    )

    for i, row in (
        table.reset_index(
            drop=True
        )
        .iterrows()
    ):

        ax.text(
            row[
                "moran_i"
            ],
            i,
            (
                f"  p="
                f"{row['p_permutation']:.3g}"
            ),
            va="center",
            fontsize=9,
        )

    ax.grid(
        axis="x",
        alpha=0.15,
    )

    fig.tight_layout()

    path = (
        output
        /
        "figure_02_global_moran_i.png"
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
# FIGURE 3 — LISA MAPS
# =============================================================================

def plot_lisa_maps(
    gdf,
    output,
):

    metrics = [
        "annual_tmean",
        "fd5_pct",
        "tx99p_pct",
    ]

    titles = {
        "annual_tmean":
            "Annual mean temperature trend",

        "fd5_pct":
            "Tmin < -5°C frequency trend",

        "tx99p_pct":
            "P99 hot-day frequency trend",
    }

    colours = {
        "High-High":
            "#d73027",

        "Low-Low":
            "#4575b4",

        "High-Low":
            "#fdae61",

        "Low-High":
            "#74add1",

        "Not significant":
            "#d9d9d9",

        "Missing":
            "#ffffff",
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            17,
            5.8,
        )
    )

    for ax, prefix in zip(
        axes,
        metrics,
    ):

        column = (
            f"{prefix}_lisa_class"
        )

        if column not in gdf.columns:

            ax.axis(
                "off"
            )

            continue

        for category in [
            "Not significant",
            "Low-Low",
            "High-High",
            "Low-High",
            "High-Low",
        ]:

            subset = gdf.loc[
                gdf[
                    column
                ]
                == category
            ]

            if subset.empty:
                continue

            ax.scatter(
                subset[
                    "longitude"
                ],
                subset[
                    "latitude"
                ],
                s=(
                    9
                    if category
                    ==
                    "Not significant"
                    else 16
                ),
                c=colours[
                    category
                ],
                label=category,
                linewidths=0,
            )

        ax.set_title(
            titles[
                prefix
            ],
            loc="left",
            fontsize=10,
        )

        set_china_map_extent(
            ax,
            gdf,
        )

    handles, labels = (
        axes[
            -1
        ].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
    )

    fig.suptitle(
        (
            "Local Moran's I (LISA) "
            "clusters after FDR correction"
        ),
        fontsize=14,
    )

    fig.tight_layout(
        rect=[
            0,
            0.08,
            1,
            0.94,
        ]
    )

    path = (
        output
        /
        "figure_03_lisa_cluster_maps.png"
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
# FIGURE 4 — LATITUDE BANDS
# =============================================================================

def plot_latitude_band_changes(
    gdf,
    output,
):

    metrics = [
        "annual_tmean_slope_decade",
        "djf_tmin_slope_decade",
        "jja_tmax_slope_decade",
        "fd5_pct_slope_decade",
        "tx99p_pct_slope_decade",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            15,
            9,
        ),
    )

    axes = axes.ravel()

    categories = [
        x
        for x
        in gdf[
            "latitude_band"
        ].cat.categories
    ]

    for ax, metric in zip(
        axes,
        metrics,
    ):

        data = []

        labels = []

        for category in categories:

            x = (
                gdf.loc[
                    gdf[
                        "latitude_band"
                    ]
                    == category,
                    metric,
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

            if len(x) == 0:
                continue

            data.append(
                x.to_numpy()
            )

            labels.append(
                str(
                    category
                )
            )

        ax.boxplot(
            data,
            tick_labels=labels,
            showfliers=False,
        )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        info = SPATIAL_METRICS[
            metric
        ]

        ax.set_title(
            info[
                "label"
            ],
            loc="left",
            fontsize=10,
        )

        ax.set_ylabel(
            info[
                "unit"
            ]
        )

        ax.tick_params(
            axis="x",
            rotation=35,
        )

        ax.grid(
            axis="y",
            alpha=0.15,
        )

    axes[
        -1
    ].axis(
        "off"
    )

    fig.suptitle(
        (
            "City temperature trends "
            "across latitude bands"
        ),
        fontsize=14,
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
        "figure_04_latitude_band_comparison.png"
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
# FIGURE 5 — CLUSTER MAP
# =============================================================================

def plot_cluster_map(
    gdf,
    output,
):

    fig, ax = plt.subplots(
        figsize=(
            10,
            8,
        )
    )

    clusters = sorted(
        gdf[
            "thermal_cluster"
        ].dropna().unique()
    )

    cmap = plt.get_cmap(
        "tab10"
    )

    for i, cluster in enumerate(
        clusters
    ):

        subset = gdf.loc[
            gdf[
                "thermal_cluster"
            ]
            == cluster
        ]

        ax.scatter(
            subset[
                "longitude"
            ],
            subset[
                "latitude"
            ],
            s=15,
            label=(
                f"Cluster {cluster} "
                f"(n={len(subset)})"
            ),
            c=[
                cmap(
                    i
                    %
                    10
                )
            ],
            linewidths=0,
        )

    set_china_map_extent(
        ax,
        gdf,
    )

    ax.set_title(
        (
            "Data-driven thermal-change "
            "classification of Chinese cities"
        ),
        loc="left",
        fontsize=13,
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    fig.tight_layout()

    path = (
        output
        /
        "figure_05_city_thermal_cluster_map.png"
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
# FIGURE 6 — PCA
# =============================================================================

def plot_pca(
    pca,
    explained,
    output,
):

    fig, ax = plt.subplots(
        figsize=(
            8,
            6.5,
        )
    )

    clusters = sorted(
        pca[
            "thermal_cluster"
        ].unique()
    )

    cmap = plt.get_cmap(
        "tab10"
    )

    for i, cluster in enumerate(
        clusters
    ):

        sub = pca.loc[
            pca[
                "thermal_cluster"
            ]
            == cluster
        ]

        ax.scatter(
            sub[
                "PC1"
            ],
            sub[
                "PC2"
            ],
            s=20,
            alpha=0.6,
            c=[
                cmap(
                    i
                    %
                    10
                )
            ],
            label=(
                f"Cluster {cluster}"
            ),
        )

    ax.set_xlabel(
        (
            f"PC1 "
            f"({explained[0] * 100:.1f}%)"
        )
    )

    ax.set_ylabel(
        (
            f"PC2 "
            f"({explained[1] * 100:.1f}%)"
        )
    )

    ax.set_title(
        (
            "PCA representation of "
            "city thermal profiles"
        ),
        loc="left",
    )

    ax.axhline(
        0,
        linewidth=0.8,
        linestyle="--",
    )

    ax.axvline(
        0,
        linewidth=0.8,
        linestyle="--",
    )

    ax.legend(
        frameon=False,
    )

    ax.grid(
        alpha=0.12
    )

    fig.tight_layout()

    path = (
        output
        /
        "figure_06_pca_city_clusters.png"
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
# FIGURE 7 — CLUSTER PROFILE HEATMAP
# =============================================================================

def plot_cluster_profiles(
    profile_z,
    feature_columns,
    feature_labels,
    output,
):

    table = (
        profile_z
        .set_index(
            "thermal_cluster"
        )
    )

    table = table[
        feature_columns
    ]

    matrix = table.to_numpy(
        dtype=float
    )

    vmax = np.nanmax(
        np.abs(
            matrix
        )
    )

    if (
        not np.isfinite(
            vmax
        )
        or vmax == 0
    ):

        vmax = 1

    fig, ax = plt.subplots(
        figsize=(
            13,
            5.5,
        )
    )

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(
        np.arange(
            len(
                feature_labels
            )
        )
    )

    ax.set_xticklabels(
        feature_labels,
        rotation=55,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(
                table.index
            )
        )
    )

    ax.set_yticklabels(
        [
            f"Cluster {x}"
            for x
            in table.index
        ]
    )

    ax.set_title(
        (
            "Standardised thermal profile "
            "of each city cluster"
        ),
        loc="left",
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        shrink=0.8,
    )

    cbar.set_label(
        "Cluster mean z-score"
    )

    fig.tight_layout()

    path = (
        output
        /
        "figure_07_cluster_profile_heatmap.png"
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
# FIGURE 8 — SIGNIFICANT TREND MAP
# =============================================================================

def plot_significant_warming_map(
    gdf,
    output,
):

    metric = (
        "annual_tmean_slope_decade"
    )

    significance = (
        "annual_tmean_significant"
    )

    if (
        metric not in gdf.columns
        or
        significance not in gdf.columns
    ):

        return

    sig = gdf[
        significance
    ].copy()

    # CSV can sometimes read bool as string
    if sig.dtype == object:

        sig = (
            sig.astype(
                str
            )
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        )

    fig, ax = plt.subplots(
        figsize=(
            10,
            8,
        )
    )

    # Non-significant
    non = gdf.loc[
        ~sig
    ]

    ax.scatter(
        non[
            "longitude"
        ],
        non[
            "latitude"
        ],
        s=8,
        alpha=0.35,
        label="Not significant",
    )

    significant = gdf.loc[
        sig
    ]

    values = significant[
        metric
    ]

    vmax = np.nanpercentile(
        np.abs(
            values
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
        significant[
            "longitude"
        ],
        significant[
            "latitude"
        ],
        c=values,
        s=18,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        linewidths=0,
        label="FDR-significant",
    )

    set_china_map_extent(
        ax,
        gdf,
    )

    ax.set_title(
        (
            "FDR-significant trends in "
            "annual mean temperature"
        ),
        loc="left",
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        shrink=0.75,
    )

    cbar.set_label(
        "°C/decade"
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    path = (
        output
        /
        "figure_08_significant_annual_warming_map.png"
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
# PRINT RESULTS
# =============================================================================

def print_key_results(
    gdf,
    moran_table,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "KEY SPATIAL RESULTS"
    )

    print(
        "=" * 80
    )

    for metric, info in (
        SPATIAL_METRICS.items()
    ):

        if metric not in gdf.columns:
            continue

        x = (
            gdf[
                metric
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
            f"\n{info['label']}"
        )

        print(
            "-" * 60
        )

        print(
            f"Median city trend : "
            f"{x.median():+.4f} "
            f"{info['unit']}"
        )

        print(
            f"P25-P75           : "
            f"{x.quantile(0.25):+.4f} "
            f"to "
            f"{x.quantile(0.75):+.4f}"
        )

        row = moran_table.loc[
            moran_table[
                "metric"
            ]
            == metric
        ]

        if not row.empty:

            row = row.iloc[
                0
            ]

            print(
                f"Global Moran's I  : "
                f"{row['moran_i']:.4f}"
            )

            print(
                f"Permutation p     : "
                f"{row['p_permutation']:.4g}"
            )

    if (
        "thermal_cluster"
        in gdf.columns
    ):

        print(
            "\nThermal clusters"
        )

        print(
            "-" * 60
        )

        counts = (
            gdf[
                "thermal_cluster"
            ]
            .value_counts()
            .sort_index()
        )

        for cluster, count in (
            counts.items()
        ):

            print(
                f"Cluster {cluster}: "
                f"{count:,} cities "
                f"({count / len(gdf) * 100:.1f}%)"
            )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--trends",
        type=pathlib.Path,
        default=DEFAULT_TRENDS,
    )

    parser.add_argument(
        "--city-year",
        type=pathlib.Path,
        default=DEFAULT_CITY_YEAR,
    )

    parser.add_argument(
        "--gpkg",
        type=pathlib.Path,
        default=DEFAULT_GPKG,
    )

    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=K_NEIGHBORS,
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=N_CLUSTERS,
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    if args.k_neighbors < 1:

        raise ValueError(
            "--k-neighbors must be >= 1"
        )

    if args.clusters < 2:

        raise ValueError(
            "--clusters must be >= 2"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SPATIAL ANALYSIS OF CITY TEMPERATURE"
    )

    print(
        "=" * 80
    )

    print(
        f"Trend input : {args.trends}"
    )

    print(
        f"City-year   : {args.city_year}"
    )

    print(
        f"GHSL        : {args.gpkg}"
    )

    print(
        f"Output      : {args.output}"
    )

    # =========================================================================
    # LOAD
    # =========================================================================

    trends = load_trends(
        args.trends
    )

    city_year = load_city_year(
        args.city_year
    )

    climatology = build_city_climatology(
        city_year
    )

    ghsl = load_ghsl(
        args.gpkg
    )

    gdf = build_spatial_table(
        ghsl,
        trends,
        climatology,
    )

    gdf = add_latitude_bands(
        gdf
    )

    # =========================================================================
    # SPATIAL WEIGHTS
    # =========================================================================

    W = build_knn_weights(
        gdf[
            "longitude"
        ].to_numpy(),
        gdf[
            "latitude"
        ].to_numpy(),
        k=args.k_neighbors,
    )

    # =========================================================================
    # GLOBAL + LOCAL MORAN
    # =========================================================================

    (
        moran_table,
        lisa_outputs,
    ) = spatial_autocorrelation(
        gdf,
        W,
    )

    save_csv(
        moran_table,
        args.output
        /
        "global_moran_results.csv",
    )

    gdf = attach_lisa(
        gdf,
        lisa_outputs,
    )

    # =========================================================================
    # LATITUDE AGGREGATION
    # =========================================================================

    latitude_summary = (
        latitude_band_summary(
            gdf
        )
    )

    save_csv(
        latitude_summary,
        args.output
        /
        "latitude_band_summary.csv",
    )

    # =========================================================================
    # CITY CLUSTERING
    # =========================================================================

    cluster_result = cluster_cities(
        gdf,
        n_clusters=args.clusters,
    )

    gdf = cluster_result[
        "gdf"
    ]

    save_csv(
        cluster_result[
            "pca"
        ],
        args.output
        /
        "city_pca_scores.csv",
    )

    save_csv(
        cluster_result[
            "profile_z"
        ],
        args.output
        /
        "thermal_cluster_profiles_zscore.csv",
    )

    save_csv(
        cluster_result[
            "profile_raw"
        ],
        args.output
        /
        "thermal_cluster_profiles_raw.csv",
    )

    # =========================================================================
    # FINAL CITY TABLE
    # =========================================================================

    drop_geometry = (
        gdf.drop(
            columns=[
                "geometry"
            ]
        )
    )

    save_csv(
        drop_geometry,
        args.output
        /
        "city_spatial_analysis_results.csv",
    )

    # -------------------------------------------------------------------------
    # Optional GeoPackage output
    # -------------------------------------------------------------------------

    try:

        gpkg_output = (
            args.output
            /
            "city_spatial_analysis_results.gpkg"
        )

        gdf.to_file(
            gpkg_output,
            layer="city_spatial_results",
            driver="GPKG",
        )

        print(
            f"Saved optional GeoPackage: "
            f"{gpkg_output}"
        )

    except Exception as exc:

        print(
            "\nWARNING: GeoPackage output "
            "was skipped."
        )

        print(
            f"Reason: {exc}"
        )

        print(
            "CSV output is complete and "
            "is sufficient for later analysis."
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

    plot_trend_maps(
        gdf,
        args.output,
    )

    plot_global_moran(
        moran_table,
        args.output,
    )

    plot_lisa_maps(
        gdf,
        args.output,
    )

    plot_latitude_band_changes(
        gdf,
        args.output,
    )

    plot_cluster_map(
        gdf,
        args.output,
    )

    plot_pca(
        cluster_result[
            "pca"
        ],
        cluster_result[
            "explained"
        ],
        args.output,
    )

    plot_cluster_profiles(
        cluster_result[
            "profile_z"
        ],
        cluster_result[
            "feature_columns"
        ],
        cluster_result[
            "feature_labels"
        ],
        args.output,
    )

    plot_significant_warming_map(
        gdf,
        args.output,
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print_key_results(
        gdf,
        moran_table,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SPATIAL ANALYSIS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nMain output:"
    )

    print(
        args.output
        /
        "city_spatial_analysis_results.csv"
    )

    print(
        "\nGlobal spatial statistics:"
    )

    print(
        args.output
        /
        "global_moran_results.csv"
    )

    print(
        "\nCluster profiles:"
    )

    print(
        args.output
        /
        "thermal_cluster_profiles_raw.csv"
    )

    # =========================================================================
    # SHOW
    # =========================================================================

    if not args.no_show:

        print(
            "\nFigures will now be displayed."
        )

        print(
            "Close all figure windows "
            "to finish the script."
        )

        plt.show()


if __name__ == "__main__":
    main()