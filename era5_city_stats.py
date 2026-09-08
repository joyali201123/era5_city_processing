"""UTC daily d2m, then China city intersection-area statistics.
Install: pip install numpy pandas xarray netCDF4 geopandas shapely pyproj
Run: python era5_city_stats.py --archive INPUT --gpkg UCDB --output NEW_DIRECTORY
For existing daily files: --resume-daily --gpkg UCDB --output EXISTING_DIRECTORY
Archive is streamed once; one NC member is temporarily staged at a time.
Requires shapely >= 2.0. No input files are modified.
"""
import argparse, contextlib, csv, json, pathlib, shutil, tarfile, tempfile, zipfile
import time
from datetime import datetime, timezone

STARTED = time.monotonic()
def log(*items, **kwargs):
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp} elapsed={time.monotonic()-STARTED:.0f}s]", *items, flush=True)
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform, unary_union
from shapely import make_valid
from shapely.validation import explain_validity

LAYER = 'GHSL_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A'

@contextlib.contextmanager
def members(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            yield ((m.filename, z.open(m)) for m in z.infolist() if not m.is_dir() and m.filename.lower().endswith('.nc'))
    else:
        with tarfile.open(path, 'r|*') as t:
            yield ((m.name, t.extractfile(m)) for m in t if m.isfile() and m.name.lower().endswith('.nc'))

def normalize(ds):
    rename = {}
    for target, alias in [('time','valid_time'), ('latitude','lat'), ('longitude','lon')]:
        if target not in ds.coords and alias in ds.coords:
            rename[alias] = target
    ds = ds.rename(rename)
    if 'd2m' not in ds:
        raise ValueError(f'd2m absent: {list(ds.data_vars)}')
    a = ds.d2m
    for dim in list(a.dims):
        if dim not in ('time','latitude','longitude'):
            if a.sizes[dim] != 1:
                raise ValueError(f'Unresolved dimension: {dim}')
            a = a.isel({dim:0}, drop=True)
    a = a.transpose('time','latitude','longitude')
    for coord in ('latitude','longitude'):
        v = a[coord].values
        if v.ndim != 1 or len(v)<2 or not np.allclose(np.abs(np.diff(v)), .25):
            raise ValueError(f'{coord}: expected regular 0.25 degree grid')
    if str(a.attrs.get('units','')).lower() not in ('k','kelvin','degrees_kelvin'):
        raise ValueError('Expected d2m units Kelvin')
    idx = pd.DatetimeIndex(a.time.values)
    if idx.empty or idx.has_duplicates or not idx.is_monotonic_increasing:
        raise ValueError('Time must be nonempty, unique and increasing')
    return a, idx

def daily_stage(args):
    paths, seen, grid = [], set(), None
    with members(args.archive) as stream:
        for name, source in stream:
            log('Reading:', name, flush=True)
            with tempfile.TemporaryDirectory(dir=args.scratch or args.output) as tmp:
                local = pathlib.Path(tmp)/'member.nc'
                with source, local.open('wb') as dest:
                    copied=0
                    last_report=time.monotonic()
                    log('Extracting member to temporary disk:',name)
                    while True:
                        chunk=source.read(8*1024*1024)
                        if not chunk:
                            break
                        dest.write(chunk)
                        copied+=len(chunk)
                        if time.monotonic()-last_report>=60:
                            log(f'Extraction: {name}, {copied/1024**2:.0f} MiB written')
                            last_report=time.monotonic()
                    log(f'Extraction complete: {copied/1024**2:.0f} MiB')
                with xr.open_dataset(local, engine='netcdf4') as ds:
                    a, idx = normalize(ds)
                    log("NetCDF verified:",dict(a.sizes),"UTC time range:",idx[0],idx[-1])
                    current = (a.latitude.values, a.longitude.values)
                    if grid is None:
                        grid = tuple(v.copy() for v in current)
                    elif not all(np.array_equal(x,y) for x,y in zip(grid,current)):
                        raise ValueError('Grid changes across input files')
                    dates = idx.normalize()
                    for day_number,day in enumerate(dates.unique(),1):
                        log(f"Daily aggregation {day_number}/{len(dates.unique())}: {day.date()}")
                        pos = np.flatnonzero(dates == day)
                        if not idx[pos].equals(pd.date_range(day, periods=24, freq='h')):
                            raise ValueError(f'{name}: incomplete/split UTC day {day}; requires 24 hourly samples in one member')
                        key = day.strftime('%Y-%m-%d')
                        if key in seen:
                            raise ValueError(f'Duplicate day {key}')
                        seen.add(key)
                        b = a.isel(time=slice(pos[0],pos[-1]+1)).load()-273.15
                        result = xr.Dataset({f'd2m_{s}':getattr(b,s)('time',skipna=False) for s in ('min','max','mean')}).expand_dims(time=[day.to_datetime64()])
                        result.attrs.update(timezone='UTC',source_member=name,hourly_count=24,missing_policy='Any missing hour yields missing daily value')
                        for v in result:
                            result[v].attrs['units']='degC'
                        path=args.output/'daily'/f'{key}.nc'
                        result.to_netcdf(path,engine='netcdf4',encoding={v:dict(zlib=True,complevel=1,dtype='float32') for v in result})
                        paths.append(path)
                        log("Daily file saved:",path.name)
            if args.first_member_only:
                break
    if not paths:
        raise ValueError('No NC input files')
    return sorted(paths), grid

def repair_polygon(city):
    """Repair in memory; discard only non-area remnants from make_valid."""
    if city is None or city.is_empty:
        raise ValueError('Missing or empty city geometry')
    reason = explain_validity(city)
    if city.is_valid:
        return city, None
    def polygons(g):
        if g.geom_type == 'Polygon':
            return [g]
        return [p for part in getattr(g, 'geoms', []) for p in polygons(part)]
    fixed = unary_union(polygons(make_valid(city)))
    if fixed.is_empty or not fixed.is_valid or fixed.area <= 0:
        raise ValueError(f'Geometry repair failed: {reason}')
    return fixed, reason


def resume_daily(args):
    paths = sorted((args.output/'daily').glob('*.nc'))
    if not paths:
        raise ValueError('No existing daily/*.nc files')
    dates = pd.DatetimeIndex([pd.Timestamp(p.stem) for p in paths])
    if dates.has_duplicates or not dates.equals(pd.date_range(dates[0], dates[-1], freq='D')):
        raise ValueError('Daily filenames contain gaps or duplicate dates')
    with xr.open_dataset(paths[0], engine='netcdf4') as ds:
        grid = (ds.latitude.values.copy(), ds.longitude.values.copy())
    log(f'Resuming {len(paths)} daily files: {dates[0].date()} through {dates[-1].date()}; hourly archive skipped.', flush=True)
    return paths, grid


def weights_stage(args,grid):
    lat,lon0=grid
    lon=(lon0+180)%360-180
    log('Loading city boundaries:',args.gpkg)
    cities=gpd.read_file(args.gpkg,layer=LAYER)
    cities=cities.loc[cities.GC_CNT_GAD_2025=='China'].copy()
    repairs=[]
    for index,row in cities.iterrows():
        fixed,reason=repair_polygon(row.geometry)
        if reason:
            repairs.append(dict(city_id=int(row.ID_UC_G0),city_name=row.GC_UCN_MAI_2025,reason=reason,
                                original_area=float(row.geometry.area),repaired_area=float(fixed.area),crs=str(cities.crs)))
            cities.at[index,'geometry']=fixed
            log(f'Repaired city {row.ID_UC_G0}: {reason}',flush=True)
    (args.output/'geometry_repairs.json').write_text(json.dumps(repairs,ensure_ascii=False,indent=2),encoding='utf-8')
    cities=cities.to_crs(4326)
    if cities.empty or cities.ID_UC_G0.duplicated().any():
        raise ValueError('Empty China selection or duplicate IDs')
    weights=[]
    for number,row in enumerate(cities.itertuples(),1):
        if number == 1 or number % 100 == 0 or number == len(cities):
            log(f"City mask {number}/{len(cities)}",flush=True)
        city=row.geometry
        if city is None or city.is_empty or not city.is_valid:
            raise ValueError(f'Invalid city geometry {row.ID_UC_G0}')
        xmin,ymin,xmax,ymax=city.bounds
        ys=np.flatnonzero((lat+.125>ymin)&(lat-.125<ymax))
        xs=np.flatnonzero((lon+.125>xmin)&(lon-.125<xmax))
        center=city.centroid
        project=Transformer.from_crs(4326,f'+proj=laea +lat_0={center.y} +lon_0={center.x} +datum=WGS84 +units=m',always_xy=True).transform
        def area(g):
            return transform(project,g.segmentize(.01)).area
        ids,areas=[],[]
        for y in ys:
            for x in xs:
                overlap=city.intersection(box(lon[x]-.125,lat[y]-.125,lon[x]+.125,lat[y]+.125))
                if not overlap.is_empty:
                    w=area(overlap)
                    if w>0:
                        ids.append(int(y*len(lon)+x)); areas.append(w)
        weights.append((int(row.ID_UC_G0),row.GC_UCN_MAI_2025,np.asarray(ids,dtype=np.int64),np.asarray(areas),area(city)))
    np.savez_compressed(args.output/'city_weights.npz',city_id=[w[0] for w in weights],city_name=[w[1] for w in weights],
        offsets=np.cumsum([0]+[len(w[2]) for w in weights]),indices=np.concatenate([w[2] for w in weights]),
        area_m2=np.concatenate([w[3] for w in weights]),city_area_m2=[w[4] for w in weights],latitude=lat,longitude=lon0)
    log('Mask weights saved; cities:',len(weights))
    return weights

def stats(v,w):
    valid=np.isfinite(v)
    if not valid.any():
        return [np.nan]*5+[0.0]
    x,a=v[valid],w[valid]
    order=np.argsort(x)
    cumulative=np.cumsum(a[order])/a.sum()
    q=[x[order][min(np.searchsorted(cumulative,p),len(x)-1)] for p in (.25,.75)]
    return [np.average(x,weights=a),x.max(),x.min(),*q,a.sum()/w.sum()]

def cities_stage(args,paths,weights,grid):
    variables=('d2m_min','d2m_max','d2m_mean')
    labels=('mean','max','min','p25','p75','valid_area_fraction')
    with (args.output/'city_daily.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['date_utc','city_id','city_name','grid_overlap_count','grid_coverage_fraction']+[f'{v}_{s}' for v in variables for s in labels])
        for number,path in enumerate(paths,1):
            if number == 1 or number % 30 == 0 or number == len(paths):
                log(f'City statistics day {number}/{len(paths)}: {path.stem}')
            with xr.open_dataset(path,engine='netcdf4') as ds:
                if not all(np.array_equal(ds[c].values,g) for c,g in zip(('latitude','longitude'),grid)):
                    raise ValueError(f'Grid mismatch: {path}')
                if ds.sizes.get('time') != 1 or pd.Timestamp(ds.time.values[0]) != pd.Timestamp(path.stem):
                    raise ValueError(f'Daily timestamp mismatch: {path}')
                for v in variables:
                    if ds[v].dims != ('time','latitude','longitude') or ds[v].attrs.get('units') != 'degC':
                        raise ValueError(f'Unexpected variable structure or units: {path}, {v}')
                arrays={v:ds[v].values.ravel() for v in variables}
                for identity,name,ids,areas,total in weights:
                    row=[path.stem,identity,name,len(ids),areas.sum()/total]
                    for v in variables:
                        row.extend(stats(arrays[v][ids],areas))
                    writer.writerow(row)
            f.flush()
            if number == 1 or number % 30 == 0 or number == len(paths):
                log(f'City statistics saved through {path.stem}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=pathlib.Path,help='Hourly archive; required unless --resume-daily')
    p.add_argument('--gpkg',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--scratch',type=pathlib.Path,help='Temporary extraction directory, preferably node-local scratch')
    p.add_argument('--first-member-only',action='store_true')
    p.add_argument('--resume-daily',action='store_true',help='Reuse existing daily files; never open the hourly archive')
    args=p.parse_args()
    if not args.resume_daily and (args.archive is None or not args.archive.is_file()):
        p.error('--archive must name an existing archive for a fresh run')
    if not args.gpkg.is_file():
        p.error('--gpkg must name an existing GeoPackage')
    if args.scratch:
        args.scratch.mkdir(parents=True,exist_ok=True)
    log('Configuration:',vars(args))
    if args.resume_daily:
        if any((args.output/name).exists() for name in ('city_daily.csv','method.json')):
            raise FileExistsError('City output already exists; preserve it before retrying')
        paths,grid=resume_daily(args)
    else:
        if args.output.exists():
            raise FileExistsError('Choose a NEW --output directory or use --resume-daily')
        args.output.mkdir(parents=True)
        (args.output/'daily').mkdir()
        paths,grid=daily_stage(args)
    log('Daily grids complete; now constructing city masks.',flush=True)
    weights=weights_stage(args,grid)
    cities_stage(args,paths,weights,grid)
    (args.output/'method.json').write_text(json.dumps(dict(timezone='UTC',units='degC',variable='d2m: dew point, not air temperature',
        boundary='UCDB fixed 2025',country_filter='China',days=len(paths),cities=len(weights),
        area='Local equal-area LAEA; geographic boundaries densified at 0.01 degrees',
        quantile='Weighted empirical inverse CDF, P25/P75',extrema='Finite grid values with positive overlap',
        missing='Require 24 hours; propagate hourly NaN to daily; spatial statistics renormalize finite areas; report coverage'),indent=2),encoding='utf-8')
    log('Finished:',args.output,flush=True)

if __name__=='__main__':
    main()
