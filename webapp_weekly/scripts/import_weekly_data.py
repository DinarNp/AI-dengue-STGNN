"""
Import Historical Weekly Data from the Canonical Revision-Experiments CSV

Backfills the weekly-cadence database (dengue_cases, climate_data, ndvi_data)
from revision_experiments/data/fix/data_weekly_5kab_2021_2025_ndvi_neocorrected.csv
— the same, real (EWARS + OpenWeather/NASA POWER + NASA NEO NDVI) dataset the
paper's validated STGNN model was trained/evaluated on. Epi-week/epi-year are
computed with the `epiweeks` package (already a dependency) rather than trusting
the CSV's own Year/Week columns literally, so the database's week boundaries are
self-consistent regardless of how the source CSV's week numbers were derived.

Usage:
    python3 scripts/import_weekly_data.py [path/to/csv]

Defaults to the canonical revision_experiments dataset via a relative path
(../../revision_experiments/data/fix/data_weekly_5kab_2021_2025_ndvi_neocorrected.csv).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import date

from app import create_app
from app.models import db, Regency, DengueCase, ClimateData, NDVIData

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'revision_experiments', 'data', 'fix',
    'data_weekly_5kab_2021_2025_ndvi_neocorrected.csv',
)


def _epiweek_for(year: int, week: int):
    """Resolve a (Year, Week) pair from the CSV into an unambiguous epi_year/epi_week
    pair, using the ISO week's midpoint date so trailing/leading weeks (week 53,
    or a week that straddles a year boundary) land in the correct epi-year."""
    from epiweeks import Week
    from datetime import date as _date
    try:
        wk = Week(year, week, system='iso')
    except ValueError:
        # Fall back to the last valid ISO week of that year if the CSV's week
        # number overshoots (e.g. a stray 53rd week in a 52-week year).
        # Dec 28 always falls in the year's last ISO week (ISO 8601 rule).
        wk = Week.fromdate(_date(year, 12, 28), system='iso')
    return wk.year, wk.week


def import_csv_to_database(csv_path: str):
    print(f"\n{'=' * 60}")
    print(f"Importing weekly data from: {csv_path}")
    print(f"{'=' * 60}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")

    regencies = {r.name: r.id for r in Regency.query.all()}
    print(f"Found {len(regencies)} regencies in database")

    counts = {'dengue': 0, 'climate': 0, 'ndvi': 0, 'skipped': 0}

    for idx, row in df.iterrows():
        regency_name = row['Region']
        if regency_name not in regencies:
            print(f"Warning: Regency '{regency_name}' not found in database. Skipping row {idx + 1}")
            counts['skipped'] += 1
            continue

        regency_id = regencies[regency_name]
        epi_year, epi_week = _epiweek_for(int(row['Year']), int(row['Week']))
        month = date.fromisocalendar(epi_year, epi_week, 1).month

        # 1. Dengue cases
        case = DengueCase.query.filter_by(regency_id=regency_id, epi_year=epi_year, epi_week=epi_week).first()
        if case is None:
            case = DengueCase(regency_id=regency_id, epi_year=epi_year, epi_week=epi_week)
            db.session.add(case)
        case.month = month
        case.cases = int(row['Cases'])
        case.data_source = 'ewars_backfill'
        counts['dengue'] += 1

        # 2. Climate
        climate = ClimateData.query.filter_by(regency_id=regency_id, epi_year=epi_year, epi_week=epi_week).first()
        if climate is None:
            climate = ClimateData(regency_id=regency_id, epi_year=epi_year, epi_week=epi_week)
            db.session.add(climate)
        climate.month = month
        climate.temperature_min = float(row['Temperature_Min'])
        climate.temperature_max = float(row['Temperature_Max'])
        climate.temperature_avg = float(row['Temperature_Avg'])
        climate.humidity = float(row['Humidity'])
        climate.precipitation_total = float(row['Precipitation_Total'])
        climate.pressure = float(row['Pressure'])
        climate.wind_speed = float(row['Wind_Speed'])
        climate.wind_direction = float(row['Wind_Direction'])
        climate.cloud_cover = float(row['Cloud_Cover'])
        climate.data_source = 'openweather_backfill'
        counts['climate'] += 1

        # 3. NDVI
        ndvi = NDVIData.query.filter_by(regency_id=regency_id, epi_year=epi_year, epi_week=epi_week).first()
        if ndvi is None:
            ndvi = NDVIData(regency_id=regency_id, epi_year=epi_year, epi_week=epi_week)
            db.session.add(ndvi)
        ndvi.month = month
        ndvi.ndvi_value = float(row['NDVI'])
        ndvi.data_source = 'modis_neo_csv'
        ndvi.is_imputed = False
        counts['ndvi'] += 1

        if (idx + 1) % 200 == 0:
            db.session.commit()
            print(f"  ...committed {idx + 1} rows")

    db.session.commit()

    print(f"\n{'=' * 60}")
    print("Import complete:")
    print(f"  Dengue case rows written: {counts['dengue']}")
    print(f"  Climate rows written:     {counts['climate']}")
    print(f"  NDVI rows written:        {counts['ndvi']}")
    print(f"  Skipped (unknown region): {counts['skipped']}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)

    app = create_app('development')
    with app.app_context():
        db.create_all()
        import_csv_to_database(csv_path)
