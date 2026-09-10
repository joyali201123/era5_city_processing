"""
ERA5 hourly 2-m air temperature -> daily China city statistics
Period: 1950-2025

Input:
    E:\\ERA5_temp_t2m_yearly

Main workflow:
    1. Find ERA5 NetCDF files.
    2. Optionally inspect ONE file.
    3. Index input files by year.
    4. Normalize ERA5 t2m coordinates and units.
    5. Validate ERA5 grid consistency.
    6. Build GHSL China city-grid intersection weights ONCE.
       - Invalid GHSL geometries are repaired using shapely.make_valid().
       - Unrepairable geometries are reported and skipped.
    7. For each year:
           hourly t2m
               ->
           daily Tmin / Tmax / Tmean
               ->
           city intersection-area weighted statistics
    8. Save one CSV per year.
    9. Combine yearly CSVs.
    10. Save metadata and processing summary.

Install:
    conda install numpy pandas xarray netcdf4 dask geopandas shapely pyproj

Requires:
    shapely >= 2.0
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import csv
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from pyproj import Transformer

from shapely import make_valid
from shapely.geometry import box
from shapely.ops import transform, unary_union

from xarray import SerializationWarning
from dask.diagnostics import ProgressBar


# =============================================================================
# SETTINGS
# =============================================================================

ROOT = pathlib.Path(__file__).resolve().parent

START_YEAR = 1950
END_YEAR = 2025

DEFAULT_INPUT = pathlib.Path(
    r"E:\ERA5_temp_t2m_yearly"
)

LAYER = (
    "GHSL_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"
)

GPKG = (
    ROOT.parent
    / "GHS_UCDB_REGION_EASTERN_AND_SOUTH_EASTERN_ASIA_R2024A_V1_2"
    / "GHS_UCDB_REGION_EASTERN_AND_SOUTH_EASTERN_ASIA_R2024A.gpkg"
)


# =============================================================================
# INSPECT ONE FILE
# =============================================================================

def inspect_one_file(path):
    """
    Open ONE ERA5 NetCDF file and print its structure.

    No processing or modification is performed.

    Reports:
        - dataset structure
        - data variables
        - dimensions
        - coordinates
        - t2m presence
        - units
        - extra dimensions
        - time range
        - number of timestamps
        - latitude/longitude ranges
        - grid spacing
    """

    path = pathlib.Path(path)

    print("\n" + "=" * 80)
    print("ERA5 SINGLE-FILE INSPECTION")
    print("=" * 80)

    print(f"\nFile:\n{path}")

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist:\n{path}"
        )

    with xr.open_dataset(
        path,
        engine="netcdf4",
    ) as ds:

        # ---------------------------------------------------------------------
        # Dataset
        # ---------------------------------------------------------------------

        print("\n--- Dataset summary ---")
        print(ds)

        # ---------------------------------------------------------------------
        # Variables
        # ---------------------------------------------------------------------

        print("\n--- Data variables ---")

        for var in ds.data_vars:

            da = ds[var]

            print(
                f"{var}: "
                f"dims={da.dims}, "
                f"shape={da.shape}, "
                f"dtype={da.dtype}, "
                f"units={da.attrs.get('units', 'UNKNOWN')}"
            )

        # ---------------------------------------------------------------------
        # Coordinates
        # ---------------------------------------------------------------------

        print("\n--- Coordinates ---")

        for coord in ds.coords:

            c = ds[coord]

            print(
                f"{coord}: "
                f"dims={c.dims}, "
                f"shape={c.shape}, "
                f"dtype={c.dtype}"
            )

        # ---------------------------------------------------------------------
        # Detect coordinate names
        # ---------------------------------------------------------------------

        time_name = None
        lat_name = None
        lon_name = None

        for candidate in (
            "time",
            "valid_time",
        ):
            if candidate in ds.coords or candidate in ds.dims:
                time_name = candidate
                break

        for candidate in (
            "latitude",
            "lat",
        ):
            if candidate in ds.coords or candidate in ds.dims:
                lat_name = candidate
                break

        for candidate in (
            "longitude",
            "lon",
        ):
            if candidate in ds.coords or candidate in ds.dims:
                lon_name = candidate
                break

        print("\n--- Detected coordinate names ---")

        print(f"time      : {time_name}")
        print(f"latitude  : {lat_name}")
        print(f"longitude : {lon_name}")

        # ---------------------------------------------------------------------
        # t2m
        # ---------------------------------------------------------------------

        print("\n--- Temperature variable ---")

        if "t2m" in ds.data_vars:

            a = ds["t2m"]

            print("t2m found : YES")
            print(f"dimensions: {a.dims}")
            print(f"shape     : {a.shape}")
            print(
                f"units     : "
                f"{a.attrs.get('units', 'UNKNOWN')}"
            )
            print(
                f"long_name : "
                f"{a.attrs.get('long_name', 'UNKNOWN')}"
            )

            recognized_dims = {
                x
                for x in (
                    time_name,
                    lat_name,
                    lon_name,
                )
                if x is not None
            }

            extra_dims = [
                dim
                for dim in a.dims
                if dim not in recognized_dims
            ]

            print(f"extra dims: {extra_dims}")

            for dim in extra_dims:
                print(
                    f"    {dim}: "
                    f"size={a.sizes[dim]}"
                )

        else:

            print("t2m found : NO")
            print(
                f"Available variables: "
                f"{list(ds.data_vars)}"
            )

        # ---------------------------------------------------------------------
        # Time
        # ---------------------------------------------------------------------

        if time_name is not None:

            idx = pd.DatetimeIndex(
                ds[time_name].values
            )

            print("\n--- Time information ---")

            print(
                f"Number timestamps : {len(idx)}"
            )

            if len(idx) > 0:

                print(
                    f"First timestamp   : {idx[0]}"
                )

                print(
                    f"Last timestamp    : {idx[-1]}"
                )

                print(
                    f"Duplicate times   : "
                    f"{idx.has_duplicates}"
                )

                print(
                    f"Monotonic         : "
                    f"{idx.is_monotonic_increasing}"
                )

                print(
                    f"Years present     : "
                    f"{sorted(set(idx.year))}"
                )

        # ---------------------------------------------------------------------
        # Grid
        # ---------------------------------------------------------------------

        if (
            lat_name is not None
            and lon_name is not None
        ):

            lat = np.asarray(
                ds[lat_name].values
            )

            lon = np.asarray(
                ds[lon_name].values
            )

            print("\n--- Spatial grid ---")

            print(
                f"Latitude count    : {len(lat)}"
            )

            print(
                f"Longitude count   : {len(lon)}"
            )

            print(
                f"Latitude range    : "
                f"{lat.min()} to {lat.max()}"
            )

            print(
                f"Longitude range   : "
                f"{lon.min()} to {lon.max()}"
            )

            if len(lat) > 1:

                dlat = np.abs(
                    np.diff(lat)
                )

                print(
                    f"Latitude spacing  : "
                    f"{dlat.min()} to {dlat.max()}"
                )

            if len(lon) > 1:

                dlon = np.abs(
                    np.diff(lon)
                )

                print(
                    f"Longitude spacing : "
                    f"{dlon.min()} to {dlon.max()}"
                )

    print("\nInspection complete.")
    print("=" * 80)


# =============================================================================
# FIND INPUT FILES
# =============================================================================

def find_nc_files(input_dir):
    """
    Recursively find NetCDF files.
    """

    input_dir = pathlib.Path(
        input_dir
    )

    files = sorted(
        set(
            list(
                input_dir.rglob("*.nc")
            )
            +
            list(
                input_dir.rglob("*.nc4")
            )
        )
    )

    if not files:

        raise FileNotFoundError(
            f"No .nc or .nc4 files found under:\n"
            f"{input_dir}"
        )

    return files


# =============================================================================
# NORMALIZE ERA5 T2M
# =============================================================================

def normalize_t2m(ds):
    """
    Normalize ERA5 t2m coordinate names and dimensions.

    Final dimensions:
        time
        latitude
        longitude

    Returns
    -------
    a : xarray.DataArray
        ERA5 t2m.

    idx : pandas.DatetimeIndex
        Time coordinate.

    conversion : str
        'kelvin' or 'celsius'.
    """

    rename = {}

    aliases = [
        ("time", "valid_time"),
        ("latitude", "lat"),
        ("longitude", "lon"),
    ]

    for target, alias in aliases:

        if (
            target not in ds.coords
            and target not in ds.dims
            and (
                alias in ds.coords
                or alias in ds.dims
            )
        ):

            rename[alias] = target

    if rename:
        ds = ds.rename(rename)

    # -------------------------------------------------------------------------
    # Check variable
    # -------------------------------------------------------------------------

    if "t2m" not in ds.data_vars:

        raise ValueError(
            f"'t2m' absent. "
            f"Available variables: "
            f"{list(ds.data_vars)}"
        )

    a = ds["t2m"]

    # -------------------------------------------------------------------------
    # Extra dimensions
    # -------------------------------------------------------------------------

    for dim in list(a.dims):

        if dim not in (
            "time",
            "latitude",
            "longitude",
        ):

            if a.sizes[dim] != 1:

                raise ValueError(
                    f"Unresolved dimension: "
                    f"{dim}, "
                    f"size={a.sizes[dim]}"
                )

            print(
                f"Removing singleton "
                f"dimension: {dim}"
            )

            a = a.isel(
                {dim: 0},
                drop=True,
            )

    # -------------------------------------------------------------------------
    # Required dimensions
    # -------------------------------------------------------------------------

    required = {
        "time",
        "latitude",
        "longitude",
    }

    if not required.issubset(
        set(a.dims)
    ):

        raise ValueError(
            f"Expected dimensions "
            f"{required}; "
            f"found {a.dims}"
        )

    a = a.transpose(
        "time",
        "latitude",
        "longitude",
    )

    # -------------------------------------------------------------------------
    # Grid
    # -------------------------------------------------------------------------

    for coord in (
        "latitude",
        "longitude",
    ):

        values = np.asarray(
            a[coord].values
        )

        if values.ndim != 1:

            raise ValueError(
                f"{coord} must be 1-D"
            )

        if len(values) < 2:

            raise ValueError(
                f"{coord} contains "
                f"fewer than two points"
            )

        spacing = np.abs(
            np.diff(values)
        )

        if not np.allclose(
            spacing,
            0.25,
            atol=1e-6,
            rtol=0,
        ):

            raise ValueError(
                f"{coord}: expected regular "
                f"0.25 degree ERA5 grid; "
                f"spacing range = "
                f"{spacing.min()} "
                f"to {spacing.max()}"
            )

    # -------------------------------------------------------------------------
    # Units
    # -------------------------------------------------------------------------

    units = str(
        a.attrs.get(
            "units",
            "",
        )
    ).strip().lower()

    kelvin_units = {
        "k",
        "kelvin",
        "degrees_kelvin",
        "degree_kelvin",
    }

    celsius_units = {
        "degc",
        "c",
        "celsius",
        "degree_celsius",
        "degrees_celsius",
    }

    if units in kelvin_units:

        conversion = "kelvin"

    elif units in celsius_units:

        conversion = "celsius"

    else:

        raise ValueError(
            f"Unexpected t2m units: "
            f"'{units}'"
        )

    # -------------------------------------------------------------------------
    # Time
    # -------------------------------------------------------------------------

    idx = pd.DatetimeIndex(
        a.time.values
    )

    if idx.empty:

        raise ValueError(
            "Time coordinate is empty"
        )

    if idx.has_duplicates:

        raise ValueError(
            "Duplicate timestamps detected"
        )

    if not idx.is_monotonic_increasing:

        raise ValueError(
            "Time coordinate is not "
            "monotonically increasing"
        )

    return a, idx, conversion


# =============================================================================
# INDEX FILES BY YEAR
# =============================================================================

def index_files_by_year(
    files,
    start_year,
    end_year,
):
    """
    Read only metadata/time coordinates and determine which
    files contain each requested year.

    Each file is inspected once.

    Returns
    -------
    dict
        {
            1950: [file1, ...],
            1951: [file2, ...],
            ...
        }
    """

    print("\n" + "=" * 80)
    print("INDEXING INPUT FILES BY YEAR")
    print("=" * 80)

    file_index = {
        year: []
        for year in range(
            start_year,
            end_year + 1,
        )
    }

    for n, path in enumerate(
        files,
        start=1,
    ):

        with xr.open_dataset(
            path,
            engine="netcdf4",
        ) as ds:

            # Detect time coordinate
            if "time" in ds.coords:

                time_name = "time"

            elif "valid_time" in ds.coords:

                time_name = "valid_time"

            else:

                raise ValueError(
                    f"No time/valid_time "
                    f"coordinate found:\n{path}"
                )

            idx = pd.DatetimeIndex(
                ds[time_name].values
            )

            years = sorted(
                set(idx.year)
            )

            relevant = [
                year
                for year in years
                if (
                    start_year
                    <= year
                    <= end_year
                )
            ]

            for year in relevant:

                file_index[
                    year
                ].append(path)

        print(
            f"{n}/{len(files)}: "
            f"{path.name} -> "
            f"{years}"
        )

    print("\nInput file index:")

    for year in range(
        start_year,
        end_year + 1,
    ):

        print(
            f"{year}: "
            f"{len(file_index[year])} file(s)"
        )

    return file_index


# =============================================================================
# GET REFERENCE GRID
# =============================================================================

def obtain_reference_grid(files):
    """
    Obtain ERA5 latitude/longitude grid from first valid file.
    """

    for path in files:

        try:

            with xr.open_dataset(
                path,
                engine="netcdf4",
            ) as ds:

                a, _, _ = normalize_t2m(
                    ds
                )

                lat = np.asarray(
                    a.latitude.values
                ).copy()

                lon = np.asarray(
                    a.longitude.values
                ).copy()

                print("\nReference grid from:")
                print(path)

                print(
                    f"Grid shape: "
                    f"{len(lat)} latitude x "
                    f"{len(lon)} longitude"
                )

                return lat, lon

        except Exception as exc:

            print(
                f"Cannot use "
                f"{path.name}: {exc}"
            )

    raise ValueError(
        "No valid ERA5 t2m file found."
    )


# =============================================================================
# VALIDATE ALL GRIDS
# =============================================================================

def validate_all_grids(
    files,
    reference_grid,
):
    """
    Confirm all ERA5 files use exactly the same spatial grid.
    """

    ref_lat, ref_lon = reference_grid

    print("\n" + "=" * 80)
    print("CHECKING GRID CONSISTENCY")
    print("=" * 80)

    for n, path in enumerate(
        files,
        start=1,
    ):

        with xr.open_dataset(
            path,
            engine="netcdf4",
        ) as ds:

            a, _, _ = normalize_t2m(
                ds
            )

            lat = np.asarray(
                a.latitude.values
            )

            lon = np.asarray(
                a.longitude.values
            )

            if not np.array_equal(
                lat,
                ref_lat,
            ):

                raise ValueError(
                    f"Latitude grid changed:\n"
                    f"{path}"
                )

            if not np.array_equal(
                lon,
                ref_lon,
            ):

                raise ValueError(
                    f"Longitude grid changed:\n"
                    f"{path}"
                )

        if (
            n % 10 == 0
            or n == len(files)
        ):

            print(
                f"Grid checked: "
                f"{n}/{len(files)} files"
            )

    print("All ERA5 grids are consistent.")


# =============================================================================
# POLYGON REPAIR
# =============================================================================

def extract_polygonal_geometry(
    geometry,
):
    """
    Keep only Polygon/MultiPolygon components from a geometry.

    make_valid() may return GeometryCollection.
    """

    if geometry is None:
        return None

    if geometry.is_empty:
        return None

    if geometry.geom_type in (
        "Polygon",
        "MultiPolygon",
    ):

        return geometry

    if hasattr(
        geometry,
        "geoms",
    ):

        pieces = []

        for g in geometry.geoms:

            cleaned = (
                extract_polygonal_geometry(
                    g
                )
            )

            if (
                cleaned is not None
                and not cleaned.is_empty
            ):

                pieces.append(
                    cleaned
                )

        if not pieces:

            return None

        return unary_union(
            pieces
        )

    return None


def repair_city_geometry(
    city,
    city_id,
    city_name,
):
    """
    Validate and, when necessary, repair a GHSL city geometry.

    Returns
    -------
    geometry or None

    status : str
        'valid'
        'repaired'
        'skipped'
    """

    if (
        city is None
        or city.is_empty
    ):

        print(
            f"WARNING: empty geometry; "
            f"skipping "
            f"{city_id} ({city_name})"
        )

        return None, "skipped"

    if city.is_valid:

        return city, "valid"

    print(
        f"\nWARNING: invalid geometry: "
        f"{city_id} ({city_name})"
    )

    print(
        "Attempting make_valid()..."
    )

    try:

        repaired = make_valid(
            city
        )

        repaired = (
            extract_polygonal_geometry(
                repaired
            )
        )

    except Exception as exc:

        print(
            f"WARNING: repair failed for "
            f"{city_id} ({city_name}): "
            f"{exc}"
        )

        return None, "skipped"

    if (
        repaired is None
        or repaired.is_empty
    ):

        print(
            f"WARNING: no polygonal geometry "
            f"after repair; skipping "
            f"{city_id} ({city_name})"
        )

        return None, "skipped"

    if not repaired.is_valid:

        print(
            f"WARNING: repaired geometry "
            f"is still invalid; skipping "
            f"{city_id} ({city_name})"
        )

        return None, "skipped"

    print(
        f"Geometry repaired successfully: "
        f"{city_id} ({city_name})"
    )

    return repaired, "repaired"


# =============================================================================
# BUILD CITY WEIGHTS
# =============================================================================

def build_city_weights(
    args,
    grid,
):
    """
    Build ERA5-grid / GHSL-city intersection-area weights.

    Invalid GHSL geometries are repaired where possible.

    Weight:
        physical area (m2) of city polygon intersecting each
        ERA5 0.25-degree grid cell.
    """

    lat, lon_original = grid

    # Convert ERA5 0-360 longitude, if necessary,
    # to -180 ... 180 for GHSL geometry operations.
    lon = (
        lon_original + 180
    ) % 360 - 180

    print("\n" + "=" * 80)
    print("BUILDING CITY INTERSECTION WEIGHTS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Read GHSL
    # -------------------------------------------------------------------------

    cities = gpd.read_file(
        args.gpkg,
        layer=LAYER,
    )

    required_columns = {
        "GC_CNT_GAD_2025",
        "ID_UC_G0",
        "GC_UCN_MAI_2025",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(cities.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing GHSL columns: "
            f"{sorted(missing_columns)}"
        )

    cities = (
        cities.loc[
            cities[
                "GC_CNT_GAD_2025"
            ] == "China"
        ]
        .copy()
        .to_crs(4326)
    )

    if cities.empty:

        raise ValueError(
            "China city selection is empty."
        )

    if cities[
        "ID_UC_G0"
    ].duplicated().any():

        raise ValueError(
            "Duplicate GHSL city IDs detected."
        )

    print(
        f"China GHSL urban centres: "
        f"{len(cities)}"
    )

    # -------------------------------------------------------------------------
    # Containers
    # -------------------------------------------------------------------------

    weights = []

    repaired_cities = []
    skipped_cities = []

    n_lon = len(lon)

    # -------------------------------------------------------------------------
    # Process cities
    # -------------------------------------------------------------------------

    for n, row in enumerate(
        cities.itertuples(
            index=False
        ),
        start=1,
    ):

        city_id = int(
            row.ID_UC_G0
        )

        city_name = str(
            row.GC_UCN_MAI_2025
        )

        city = row.geometry

        # ---------------------------------------------------------------------
        # Validate / repair geometry
        # ---------------------------------------------------------------------

        city, geometry_status = (
            repair_city_geometry(
                city,
                city_id,
                city_name,
            )
        )

        if geometry_status == "repaired":

            repaired_cities.append(
                {
                    "city_id": city_id,
                    "city_name": city_name,
                }
            )

        elif geometry_status == "skipped":

            skipped_cities.append(
                {
                    "city_id": city_id,
                    "city_name": city_name,
                }
            )

            continue

        # ---------------------------------------------------------------------
        # Candidate ERA5 cells
        # ---------------------------------------------------------------------

        xmin, ymin, xmax, ymax = (
            city.bounds
        )

        ys = np.flatnonzero(
            (lat + 0.125 > ymin)
            &
            (lat - 0.125 < ymax)
        )

        xs = np.flatnonzero(
            (lon + 0.125 > xmin)
            &
            (lon - 0.125 < xmax)
        )

        # ---------------------------------------------------------------------
        # Local equal-area projection
        # ---------------------------------------------------------------------

        center = (
            city.representative_point()
        )

        project = (
            Transformer
            .from_crs(
                4326,
                (
                    f"+proj=laea "
                    f"+lat_0={center.y} "
                    f"+lon_0={center.x} "
                    f"+datum=WGS84 "
                    f"+units=m"
                ),
                always_xy=True,
            )
            .transform
        )

        def area_m2(g):
            """
            Calculate projected area after densifying
            geographic polygon boundaries.
            """

            return transform(
                project,
                g.segmentize(
                    0.01
                ),
            ).area

        # ---------------------------------------------------------------------
        # Grid intersections
        # ---------------------------------------------------------------------

        ids = []
        areas = []

        for y in ys:

            for x in xs:

                cell = box(
                    lon[x] - 0.125,
                    lat[y] - 0.125,
                    lon[x] + 0.125,
                    lat[y] + 0.125,
                )

                try:

                    overlap = (
                        city.intersection(
                            cell
                        )
                    )

                except Exception as exc:

                    raise ValueError(
                        f"Intersection failed for "
                        f"city {city_id} "
                        f"({city_name}), "
                        f"grid y={y}, x={x}: "
                        f"{exc}"
                    )

                if overlap.is_empty:
                    continue

                w = area_m2(
                    overlap
                )

                if w > 0:

                    flat_id = int(
                        y * n_lon + x
                    )

                    ids.append(
                        flat_id
                    )

                    areas.append(
                        w
                    )

        ids = np.asarray(
            ids,
            dtype=np.int64,
        )

        areas = np.asarray(
            areas,
            dtype=np.float64,
        )

        total_city_area = (
            area_m2(
                city
            )
        )

        if (
            not np.isfinite(
                total_city_area
            )
            or total_city_area <= 0
        ):

            print(
                f"WARNING: invalid calculated area; "
                f"skipping {city_id} ({city_name})"
            )

            skipped_cities.append(
                {
                    "city_id": city_id,
                    "city_name": city_name,
                }
            )

            continue

        weights.append(
            (
                city_id,
                city_name,
                ids,
                areas,
                total_city_area,
            )
        )

        if (
            n % 100 == 0
            or n == len(cities)
        ):

            print(
                f"City weights: "
                f"{n}/{len(cities)}"
            )

    # -------------------------------------------------------------------------
    # Check results
    # -------------------------------------------------------------------------

    if not weights:

        raise ValueError(
            "No valid city weights were created."
        )

    # -------------------------------------------------------------------------
    # Prepare compressed arrays
    # -------------------------------------------------------------------------

    lengths = [
        len(w[2])
        for w in weights
    ]

    offsets = np.cumsum(
        [0] + lengths
    )

    if sum(lengths) > 0:

        all_indices = (
            np.concatenate(
                [
                    w[2]
                    for w in weights
                ]
            )
        )

        all_areas = (
            np.concatenate(
                [
                    w[3]
                    for w in weights
                ]
            )
        )

    else:

        all_indices = np.array(
            [],
            dtype=np.int64,
        )

        all_areas = np.array(
            [],
            dtype=np.float64,
        )

    # -------------------------------------------------------------------------
    # Save reusable weights
    # -------------------------------------------------------------------------

    np.savez_compressed(
        args.output
        / "city_weights.npz",

        city_id=np.asarray(
            [
                w[0]
                for w in weights
            ],
            dtype=np.int64,
        ),

        city_name=np.asarray(
            [
                str(w[1])
                for w in weights
            ]
        ),

        offsets=offsets,

        indices=all_indices,

        area_m2=all_areas,

        city_area_m2=np.asarray(
            [
                w[4]
                for w in weights
            ],
            dtype=np.float64,
        ),

        latitude=lat,

        longitude=lon_original,
    )

    # -------------------------------------------------------------------------
    # Save geometry QC
    # -------------------------------------------------------------------------

    geometry_qc = {
        "ghsl_china_city_count":
            int(len(cities)),

        "processed_city_count":
            int(len(weights)),

        "repaired_city_count":
            int(len(repaired_cities)),

        "skipped_city_count":
            int(len(skipped_cities)),

        "repaired_cities":
            repaired_cities,

        "skipped_cities":
            skipped_cities,
    }

    (
        args.output
        / "city_geometry_qc.json"
    ).write_text(
        json.dumps(
            geometry_qc,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("CITY GEOMETRY SUMMARY")
    print("-" * 80)

    print(
        f"Original China cities : "
        f"{len(cities)}"
    )

    print(
        f"Processed cities      : "
        f"{len(weights)}"
    )

    print(
        f"Repaired geometries   : "
        f"{len(repaired_cities)}"
    )

    print(
        f"Skipped geometries    : "
        f"{len(skipped_cities)}"
    )

    return (
        weights,
        geometry_qc,
    )


# =============================================================================
# WEIGHTED CITY STATISTICS
# =============================================================================

def weighted_stats(
    values,
    weights,
):
    """
    Calculate city-level spatial statistics using
    ERA5-grid / city-polygon intersection area.

    Returns:
        weighted mean
        spatial maximum
        spatial minimum
        weighted P25
        weighted P75
        valid area fraction
    """

    values = np.asarray(
        values
    )

    weights = np.asarray(
        weights
    )

    valid = (
        np.isfinite(values)
        &
        np.isfinite(weights)
        &
        (weights > 0)
    )

    if not valid.any():

        return [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.0,
        ]

    x = values[
        valid
    ]

    a = weights[
        valid
    ]

    # -------------------------------------------------------------------------
    # Weighted mean
    # -------------------------------------------------------------------------

    weighted_mean = np.average(
        x,
        weights=a,
    )

    # -------------------------------------------------------------------------
    # Spatial extrema
    # -------------------------------------------------------------------------

    vmax = x.max()
    vmin = x.min()

    # -------------------------------------------------------------------------
    # Weighted empirical quantiles
    # -------------------------------------------------------------------------

    order = np.argsort(
        x
    )

    x_sorted = x[
        order
    ]

    a_sorted = a[
        order
    ]

    cumulative = (
        np.cumsum(
            a_sorted
        )
        /
        a_sorted.sum()
    )

    def weighted_quantile(p):

        index = np.searchsorted(
            cumulative,
            p,
            side="left",
        )

        index = min(
            index,
            len(x_sorted) - 1,
        )

        return x_sorted[
            index
        ]

    p25 = weighted_quantile(
        0.25
    )

    p75 = weighted_quantile(
        0.75
    )

    # -------------------------------------------------------------------------
    # Valid-area fraction
    # -------------------------------------------------------------------------

    total_weight = (
        weights[
            np.isfinite(weights)
            &
            (weights > 0)
        ].sum()
    )

    if total_weight > 0:

        valid_area_fraction = (
            a.sum()
            /
            total_weight
        )

    else:

        valid_area_fraction = 0.0

    return [
        weighted_mean,
        vmax,
        vmin,
        p25,
        p75,
        valid_area_fraction,
    ]


# =============================================================================
# PROCESS ONE YEAR
# =============================================================================

def process_year(
    year,
    year_files,
    weights,
    output_csv,
    target_chunk_hours=744,
):
    """
    Process ONE year.

    Main scientific workflow:

        ERA5 hourly t2m
              ↓
        Kelvin -> degC
              ↓
        validate complete hourly sequence
              ↓
        daily Tmin / Tmax / Tmean
              ↓
        compute daily grids
              ↓
        city spatial aggregation

    xarray/Dask performs temporal processing.
    Python loops are used mainly for city output.
    """

    print("\n" + "=" * 80)
    print(f"PROCESSING YEAR {year}")
    print("=" * 80)

    if not year_files:

        raise ValueError(
            f"No input files found for {year}"
        )

    print(
        f"Source files: "
        f"{len(year_files)}"
    )

    for path in year_files:

        print(
            f"  {path.name}"
        )

    # -------------------------------------------------------------------------
    # GET MAIN DATA
    # -------------------------------------------------------------------------

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            category=SerializationWarning,
        )

        # ---------------------------------------------------------------------
        # Open source files.
        #
        # Important:
        # Dataset remains open until after daily.compute().
        # This avoids lazy Dask arrays referencing closed NetCDF files.
        # ---------------------------------------------------------------------

        ds = xr.open_mfdataset(
            [
                str(path)
                for path in year_files
            ],
            engine="netcdf4",
            combine="by_coords",
            parallel=False,
            chunks={},
        )

        try:

            a, idx, conversion = (
                normalize_t2m(
                    ds
                )
            )

            # -----------------------------------------------------------------
            # Select requested year
            # -----------------------------------------------------------------

            year_mask = (
                idx.year == year
            )

            if not year_mask.any():

                raise ValueError(
                    f"Files assigned to {year} "
                    f"contain no {year} timestamps."
                )

            positions = np.flatnonzero(
                year_mask
            )

            hourly = a.isel(
                time=positions
            )

            # -----------------------------------------------------------------
            # Rechunk along time
            # -----------------------------------------------------------------

            hourly = hourly.chunk(
                {
                    "time":
                        target_chunk_hours
                }
            )

            # -----------------------------------------------------------------
            # Kelvin -> Celsius
            # -----------------------------------------------------------------

            if conversion == "kelvin":

                hourly = (
                    hourly - 273.15
                )

            hourly.attrs[
                "units"
            ] = "degC"

            # -----------------------------------------------------------------
            # TIME VALIDATION
            # -----------------------------------------------------------------

            idx_year = pd.DatetimeIndex(
                hourly.time.values
            )

            if idx_year.has_duplicates:

                duplicates = (
                    idx_year[
                        idx_year.duplicated()
                    ]
                )

                raise ValueError(
                    f"{year}: duplicate timestamps. "
                    f"Examples: "
                    f"{list(duplicates[:5])}"
                )

            leap = pd.Timestamp(
                year=year,
                month=12,
                day=31,
            ).is_leap_year

            expected_hours = (
                8784
                if leap
                else 8760
            )

            print(
                f"Hourly timestamps: "
                f"{len(idx_year)} "
                f"/ expected "
                f"{expected_hours}"
            )

            if (
                len(idx_year)
                != expected_hours
            ):

                raise ValueError(
                    f"{year}: expected "
                    f"{expected_hours} hourly "
                    f"timestamps but found "
                    f"{len(idx_year)}."
                )

            expected_index = (
                pd.date_range(
                    start=(
                        f"{year}-01-01 "
                        f"00:00:00"
                    ),
                    end=(
                        f"{year}-12-31 "
                        f"23:00:00"
                    ),
                    freq="h",
                )
            )

            if not idx_year.equals(
                expected_index
            ):

                # Find first mismatch
                mismatch_message = ""

                if (
                    len(idx_year)
                    ==
                    len(expected_index)
                ):

                    mismatch = np.flatnonzero(
                        idx_year.values
                        !=
                        expected_index.values
                    )

                    if len(mismatch) > 0:

                        i = mismatch[0]

                        mismatch_message = (
                            f" First mismatch: "
                            f"observed={idx_year[i]}, "
                            f"expected="
                            f"{expected_index[i]}."
                        )

                raise ValueError(
                    f"{year}: hourly timestamps "
                    f"are not the expected complete "
                    f"UTC sequence."
                    f"{mismatch_message}"
                )

            print(
                "Hourly sequence verified."
            )

            # -----------------------------------------------------------------
            # PROCESS TIME
            # -----------------------------------------------------------------

            print(
                "Calculating daily "
                "Tmin / Tmax / Tmean..."
            )

            daily_min = (
                hourly
                .resample(
                    time="1D"
                )
                .min(
                    dim="time",
                    skipna=False,
                )
            )

            daily_max = (
                hourly
                .resample(
                    time="1D"
                )
                .max(
                    dim="time",
                    skipna=False,
                )
            )

            daily_mean = (
                hourly
                .resample(
                    time="1D"
                )
                .mean(
                    dim="time",
                    skipna=False,
                )
            )

            daily = xr.Dataset(
                {
                    "t2m_min":
                        daily_min,

                    "t2m_max":
                        daily_max,

                    "t2m_mean":
                        daily_mean,
                }
            )

            # -----------------------------------------------------------------
            # Metadata
            # -----------------------------------------------------------------

            for variable in (
                "t2m_min",
                "t2m_max",
                "t2m_mean",
            ):

                daily[
                    variable
                ].attrs[
                    "units"
                ] = "degC"

            daily[
                "t2m_min"
            ].attrs[
                "long_name"
            ] = (
                "daily minimum "
                "2-m air temperature"
            )

            daily[
                "t2m_max"
            ].attrs[
                "long_name"
            ] = (
                "daily maximum "
                "2-m air temperature"
            )

            daily[
                "t2m_mean"
            ].attrs[
                "long_name"
            ] = (
                "daily mean "
                "2-m air temperature"
            )

            daily.attrs.update(
                {
                    "source":
                        "ERA5",

                    "variable":
                        "2-m air temperature",

                    "year":
                        int(year),

                    "timezone":
                        "UTC",

                    "input_frequency":
                        "hourly",

                    "output_frequency":
                        "daily",

                    "units":
                        "degC",

                    "source_files":
                        ",".join(
                            [
                                str(p)
                                for p
                                in year_files
                            ]
                        ),

                    "missing_policy":
                        (
                            "Daily statistics use "
                            "skipna=False; a missing "
                            "hour produces missing "
                            "daily value for that "
                            "grid cell."
                        ),
                }
            )

            # -----------------------------------------------------------------
            # COMPUTE DASK GRAPH
            # -----------------------------------------------------------------

            print(
                "Computing daily grids..."
            )

            with ProgressBar():

                daily = (
                    daily.compute()
                )

        finally:

            ds.close()

    # -------------------------------------------------------------------------
    # Check number of days
    # -------------------------------------------------------------------------

    expected_days = (
        366
        if leap
        else 365
    )

    if (
        daily.sizes["time"]
        != expected_days
    ):

        raise ValueError(
            f"{year}: expected "
            f"{expected_days} daily values "
            f"but generated "
            f"{daily.sizes['time']}."
        )

    print(
        f"Daily grids complete: "
        f"{daily.sizes['time']} days"
    )

    # -------------------------------------------------------------------------
    # PROCESS SPACE
    # -------------------------------------------------------------------------

    variables = (
        "t2m_min",
        "t2m_max",
        "t2m_mean",
    )

    labels = (
        "mean",
        "max",
        "min",
        "p25",
        "p75",
        "valid_area_fraction",
    )

    header = [
        "date_utc",
        "city_id",
        "city_name",
        "grid_overlap_count",
        "grid_coverage_fraction",
    ]

    header.extend(
        [
            f"{variable}_{label}"
            for variable in variables
            for label in labels
        ]
    )

    n_rows = 0

    print(
        "Calculating city statistics..."
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            header
        )

        for n in range(
            daily.sizes["time"]
        ):

            day = pd.Timestamp(
                daily.time.values[n]
            )

            date_text = (
                day.strftime(
                    "%Y-%m-%d"
                )
            )

            # -----------------------------------------------------------------
            # Flatten daily grids once for this day
            # -----------------------------------------------------------------

            arrays = {
                variable: (
                    daily[
                        variable
                    ]
                    .isel(
                        time=n
                    )
                    .values
                    .ravel()
                )
                for variable
                in variables
            }

            # -----------------------------------------------------------------
            # Cities
            # -----------------------------------------------------------------

            for (
                city_id,
                city_name,
                ids,
                areas,
                total_city_area,
            ) in weights:

                if len(
                    areas
                ) > 0:

                    grid_coverage_fraction = (
                        areas.sum()
                        /
                        total_city_area
                    )

                else:

                    grid_coverage_fraction = (
                        0.0
                    )

                row = [
                    date_text,
                    city_id,
                    city_name,
                    len(ids),
                    grid_coverage_fraction,
                ]

                for variable in variables:

                    if len(ids) == 0:

                        result = [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            0.0,
                        ]

                    else:

                        values = (
                            arrays[
                                variable
                            ][ids]
                        )

                        result = (
                            weighted_stats(
                                values,
                                areas,
                            )
                        )

                    row.extend(
                        result
                    )

                writer.writerow(
                    row
                )

                n_rows += 1

            # -----------------------------------------------------------------
            # Progress
            # -----------------------------------------------------------------

            if (
                (n + 1) % 30 == 0
                or
                n + 1
                == daily.sizes["time"]
            ):

                print(
                    f"{year}: "
                    f"{n + 1}/"
                    f"{daily.sizes['time']} "
                    f"days completed"
                )

    print(
        f"{year} finished: "
        f"{n_rows:,} rows"
    )

    return (
        n_rows,
        [
            str(path)
            for path
            in year_files
        ],
    )


# =============================================================================
# COMBINE YEARLY OUTPUTS
# =============================================================================

def combine_yearly_csvs(
    yearly_paths,
    combined_path,
):
    """
    Combine yearly CSVs into one final CSV without
    loading the entire dataset into memory.
    """

    if not yearly_paths:

        raise ValueError(
            "No yearly CSV files "
            "available to combine."
        )

    print("\n" + "=" * 80)
    print("COMBINING YEARLY CSV FILES")
    print("=" * 80)

    with combined_path.open(
        "wb"
    ) as destination:

        for n, path in enumerate(
            yearly_paths
        ):

            with path.open(
                "rb"
            ) as source:

                content = (
                    source.read()
                )

            # First file:
            # retain BOM and header.
            if n == 0:

                destination.write(
                    content
                )

                continue

            # Other files:
            # remove BOM.
            bom = b"\xef\xbb\xbf"

            if content.startswith(
                bom
            ):

                content = content[
                    len(bom):
                ]

            # Remove header.
            first_newline = (
                content.find(
                    b"\n"
                )
            )

            if first_newline >= 0:

                content = content[
                    first_newline + 1:
                ]

            destination.write(
                content
            )

        print(
            f"Combined "
            f"{n + 1}/{len(yearly_paths)}: "
            f"{path.name}"
        )

    print(
        f"\nCombined output:\n"
        f"{combined_path}"
    )


# =============================================================================
# MAIN PROCESSING WRAPPER
# =============================================================================

def process_data(args):
    """
    Main processing wrapper.

    Workflow:

        find files
            ↓
        optional inspection
            ↓
        index files by year
            ↓
        reference grid
            ↓
        validate grids
            ↓
        build city weights
            ↓
        process each year
            ↓
        continue past failed years
            ↓
        combine successful outputs
            ↓
        metadata + summary
    """

    # -------------------------------------------------------------------------
    # GET FILES
    # -------------------------------------------------------------------------

    files = find_nc_files(
        args.input
    )

    print(
        f"\nFound "
        f"{len(files)} NetCDF files."
    )

    # -------------------------------------------------------------------------
    # TEST / INSPECTION
    # -------------------------------------------------------------------------

    if args.inspect_first:

        inspect_one_file(
            files[0]
        )

    # -------------------------------------------------------------------------
    # INDEX FILES
    # -------------------------------------------------------------------------

    file_index = (
        index_files_by_year(
            files,
            args.start_year,
            args.end_year,
        )
    )

    # -------------------------------------------------------------------------
    # REFERENCE GRID
    # -------------------------------------------------------------------------

    grid = obtain_reference_grid(
        files
    )

    # -------------------------------------------------------------------------
    # GRID CONSISTENCY
    # -------------------------------------------------------------------------

    if not args.skip_grid_check:

        validate_all_grids(
            files,
            grid,
        )

    # -------------------------------------------------------------------------
    # CITY WEIGHTS
    # -------------------------------------------------------------------------

    (
        weights,
        geometry_qc,
    ) = build_city_weights(
        args,
        grid,
    )

    # -------------------------------------------------------------------------
    # PROCESS YEARS
    # -------------------------------------------------------------------------

    yearly_outputs = []

    completed_years = []

    failed_years = []

    source_files_by_year = {}

    total_rows = 0

    for year in range(
        args.start_year,
        args.end_year + 1,
    ):

        output_csv = (
            args.output
            / "yearly"
            / f"city_daily_{year}.csv"
        )

        try:

            year_files = (
                file_index[
                    year
                ]
            )

            rows, source_files = (
                process_year(
                    year=year,
                    year_files=year_files,
                    weights=weights,
                    output_csv=output_csv,
                    target_chunk_hours=(
                        args.chunk_hours
                    ),
                )
            )

            yearly_outputs.append(
                output_csv
            )

            completed_years.append(
                year
            )

            source_files_by_year[
                str(year)
            ] = source_files

            total_rows += rows

            print(
                f"\nCOMPLETED: {year}"
            )

        except Exception as exc:

            print(
                f"\nFAILED: {year}"
            )

            print(
                f"Reason: {exc}"
            )

            failed_years.append(
                {
                    "year":
                        int(year),

                    "error":
                        str(exc),
                }
            )

            # Remove incomplete yearly output
            # if an exception occurred during writing.
            if output_csv.exists():

                try:

                    output_csv.unlink()

                except Exception:

                    pass

    # -------------------------------------------------------------------------
    # COMBINE SUCCESSFUL YEARS
    # -------------------------------------------------------------------------

    if yearly_outputs:

        combined = (
            args.output
            / (
                f"city_daily_temperature_"
                f"{args.start_year}_"
                f"{args.end_year}.csv"
            )
        )

        combine_yearly_csvs(
            yearly_outputs,
            combined,
        )

    else:

        combined = None

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------

    metadata = {

        "dataset":
            "ERA5",

        "variable":
            "t2m",

        "long_name":
            "2-m air temperature",

        "requested_start_year":
            int(args.start_year),

        "requested_end_year":
            int(args.end_year),

        "input_frequency":
            "hourly",

        "output_frequency":
            "daily",

        "timezone":
            "UTC",

        "input_units":
            (
                "Kelvin converted to degC; "
                "existing Celsius also accepted."
            ),

        "output_units":
            "degC",

        "temporal_processing": {

            "t2m_min":
                (
                    "minimum of the "
                    "24 UTC hourly values"
                ),

            "t2m_max":
                (
                    "maximum of the "
                    "24 UTC hourly values"
                ),

            "t2m_mean":
                (
                    "arithmetic mean of the "
                    "24 UTC hourly values"
                ),
        },

        "time_validation":
            (
                "Each year must contain the "
                "complete ordered UTC hourly "
                "sequence: 8760 hours in a "
                "normal year or 8784 hours "
                "in a leap year."
            ),

        "daily_missing_policy":
            (
                "skipna=False. "
                "Any missing hourly grid-cell "
                "value causes the corresponding "
                "daily grid-cell statistic "
                "to be missing."
            ),

        "city_boundary":
            (
                "GHSL Urban Centre Database "
                "R2024A, fixed 2025 boundary; "
                "country filter = China."
            ),

        "city_geometry_handling":
            (
                "Invalid GHSL geometries are "
                "repaired using shapely.make_valid. "
                "Polygonal components are retained; "
                "unrepairable geometries are skipped "
                "and recorded."
            ),

        "ghsl_china_city_count":
            geometry_qc[
                "ghsl_china_city_count"
            ],

        "processed_city_count":
            geometry_qc[
                "processed_city_count"
            ],

        "repaired_city_count":
            geometry_qc[
                "repaired_city_count"
            ],

        "skipped_city_count":
            geometry_qc[
                "skipped_city_count"
            ],

        "spatial_grid":
            (
                "Regular ERA5 "
                "0.25-degree grid."
            ),

        "spatial_weight":
            (
                "Physical intersection area "
                "between ERA5 grid cell and "
                "GHSL urban-centre polygon."
            ),

        "area_method":
            (
                "City-specific Lambert "
                "Azimuthal Equal Area (LAEA) "
                "projection. Geographic polygon "
                "boundaries are densified at "
                "0.01 degrees before projection."
            ),

        "city_statistics": [
            "area-weighted mean",
            "spatial maximum",
            "spatial minimum",
            "area-weighted P25",
            "area-weighted P75",
            "valid-area fraction",
        ],

        "quantile_method":
            (
                "Weighted empirical inverse CDF "
                "using city-grid intersection area."
            ),

        "grid_coverage_fraction":
            (
                "Sum of ERA5-cell/city "
                "intersection area divided by "
                "complete city polygon area."
            ),

        "dask_time_chunk_hours":
            int(args.chunk_hours),

        "input_directory":
            str(args.input),

        "number_of_input_files":
            int(len(files)),

        "completed_years":
            completed_years,

        "failed_years":
            failed_years,

        "source_files_by_year":
            source_files_by_year,

        "total_city_day_rows":
            int(total_rows),

        "combined_output":
            (
                str(combined)
                if combined is not None
                else None
            ),
    }

    (
        args.output
        / "method.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PROCESS RESULTS")
    print("=" * 80)

    print(
        f"Requested years : "
        f"{args.start_year}-"
        f"{args.end_year}"
    )

    print(
        f"Completed years : "
        f"{len(completed_years)}"
    )

    print(
        f"Failed years    : "
        f"{len(failed_years)}"
    )

    print(
        f"Cities used     : "
        f"{len(weights)}"
    )

    print(
        f"Total rows      : "
        f"{total_rows:,}"
    )

    if failed_years:

        print("\nFAILED YEAR SUMMARY")

        for item in failed_years:

            print(
                f"{item['year']}: "
                f"{item['error']}"
            )

    if combined is not None:

        print(
            f"\nMain combined output:\n"
            f"{combined}"
        )

    print(
        f"\nGeometry QC:\n"
        f"{args.output / 'city_geometry_qc.json'}"
    )

    print(
        f"\nMethod metadata:\n"
        f"{args.output / 'method.json'}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # -------------------------------------------------------------------------
    # Input
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=DEFAULT_INPUT,
        help=(
            "Directory containing "
            "ERA5 t2m NetCDF files."
        ),
    )

    # -------------------------------------------------------------------------
    # GHSL
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--gpkg",
        type=pathlib.Path,
        default=GPKG,
        help=(
            "GHSL UCDB geopackage."
        ),
    )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=(
            ROOT
            / "results_t2m_1950_2025"
        ),
        help=(
            "NEW output directory."
        ),
    )

    # -------------------------------------------------------------------------
    # Years
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Dask
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--chunk-hours",
        type=int,
        default=744,
        help=(
            "Dask time chunk size. "
            "744 hours is approximately "
            "31 days."
        ),
    )

    # -------------------------------------------------------------------------
    # Grid
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--skip-grid-check",
        action="store_true",
        help=(
            "Skip complete input-grid "
            "consistency checking."
        ),
    )

    # -------------------------------------------------------------------------
    # Inspection
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--inspect-first",
        action="store_true",
        help=(
            "Inspect first input NetCDF "
            "before processing."
        ),
    )

    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help=(
            "Inspect first NetCDF and exit "
            "without doing any processing."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Validate input
    # -------------------------------------------------------------------------

    if not args.input.exists():

        raise FileNotFoundError(
            f"Input directory "
            f"does not exist:\n"
            f"{args.input}"
        )

    if (
        args.start_year
        >
        args.end_year
    ):

        raise ValueError(
            "--start-year must be "
            "<= --end-year"
        )

    if args.chunk_hours <= 0:

        raise ValueError(
            "--chunk-hours must be > 0"
        )

    # -------------------------------------------------------------------------
    # Inspection only
    # -------------------------------------------------------------------------

    if args.inspect_only:

        files = find_nc_files(
            args.input
        )

        inspect_one_file(
            files[0]
        )

        return

    # -------------------------------------------------------------------------
    # GHSL needed for processing
    # -------------------------------------------------------------------------

    if not args.gpkg.exists():

        raise FileNotFoundError(
            f"GHSL geopackage "
            f"does not exist:\n"
            f"{args.gpkg}"
        )

    # -------------------------------------------------------------------------
    # Never overwrite existing output
    # -------------------------------------------------------------------------

    if args.output.exists():

        raise FileExistsError(
            f"\nOutput directory "
            f"already exists:\n"
            f"{args.output}\n\n"
            f"Choose a new directory "
            f"or delete the previous "
            f"test output first."
        )

    args.output.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        args.output
        / "yearly"
    ).mkdir()

    # -------------------------------------------------------------------------
    # PROCESS
    # -------------------------------------------------------------------------

    process_data(
        args
    )


if __name__ == "__main__":
    main()