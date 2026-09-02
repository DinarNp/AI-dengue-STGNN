"""
Automated weekly NDVI ingestion via the NASA NEO CSV archive.

Reuses the exact extraction logic (5 kabupaten centroids -- including the
Gunung Kidul coastal-pixel fix -- and the CSV-archive download/parse) from
revision_experiments/data/ndvi_neo_processed/extract_neo_csv_ndvi.py rather
than duplicating it, so this stays in sync with the corrected NDVI pipeline
the paper's numbers are based on.

MOD_NDVI_16 is a 16-day composite product, reset to day-of-year 1 each
calendar year. For any given week, the "current" composite is the one whose
16-day window most recently started on or before that week -- this matches
the carry-forward logic in ndvi_to_weekly.py (nearest composite at or before
the target date, not interpolation). This replaces the old app's manual
GeoTIFF-upload flow with something that can be triggered automatically.
"""
import os
import sys
from datetime import date, datetime, timedelta

REVISION_EXPERIMENTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
    'revision_experiments',
)
_NDVI_MODULE_DIR = os.path.join(REVISION_EXPERIMENTS, 'data', 'ndvi_neo_processed')
if _NDVI_MODULE_DIR not in sys.path:
    sys.path.insert(0, _NDVI_MODULE_DIR)

from extract_neo_csv_ndvi import process_date, latlon_to_rc, COORDS  # noqa: E402
from epiweeks import Week

from ..models import db, Regency, NDVIData


def _composite_date_on_or_before(target: date) -> date:
    """Most recent MOD_NDVI_16 16-day composite start date <= target, within
    target's calendar year (the product resets to day-of-year 1 each year)."""
    day_of_year = target.timetuple().tm_yday
    composite_day = ((day_of_year - 1) // 16) * 16 + 1
    return date(target.year, 1, 1) + timedelta(days=composite_day - 1)


def fetch_ndvi_for_week(epi_year: int, epi_week: int) -> dict:
    """Fetch (or carry-forward) NDVI for every active regency for one epi-week."""
    wk = Week(epi_year, epi_week, system='iso')
    composite_date = _composite_date_on_or_before(wk.startdate())
    date_str = composite_date.strftime('%Y-%m-%d')
    is_imputed = composite_date != wk.startdate()  # true whenever we're carrying forward

    targets = {name: latlon_to_rc(lat, lon) for name, (lat, lon) in COORDS.items()}
    result = process_date(date_str, targets)  # {region_name: (ndvi_value_or_None, fallback_radius)}

    regencies = {r.name: r for r in Regency.query.filter_by(is_active=True).all()}
    written, failed = 0, []

    for region_name, (ndvi_value, _radius) in result.items():
        regency = regencies.get(region_name)
        if regency is None or ndvi_value is None:
            failed.append(region_name)
            continue

        existing = NDVIData.query.filter_by(regency_id=regency.id, epi_year=epi_year, epi_week=epi_week).first()
        if existing is None:
            existing = NDVIData(regency_id=regency.id, epi_year=epi_year, epi_week=epi_week)
            db.session.add(existing)
        existing.month = wk.startdate().month
        existing.ndvi_value = float(ndvi_value)
        existing.data_source = 'modis_neo_csv'
        existing.is_imputed = is_imputed
        existing.processing_date = datetime.utcnow()
        written += 1

    db.session.commit()
    return {
        'success': len(failed) == 0,
        'composite_date': date_str,
        'is_imputed': is_imputed,
        'written': written,
        'failed_regions': failed,
    }
