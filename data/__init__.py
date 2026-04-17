from data.loader import load_bars, validate_schema
from data.validate import validate as validate_data
from data.features import compute_features, FeatureConfig
from data.labels import compute_labels, LabelConfig
from data.splits import time_based_split, save_splits, SplitConfig
from data.build_dataset import build_dataset, DatasetConfig

# databento_fetcher requires the 'databento' package.
# Import lazily so the rest of the project works without it installed.
import logging as _logging
try:
    from data.databento_fetcher import fetch_and_save, fetch_bars
except ImportError:
    _logging.getLogger(__name__).debug("databento package not available — fetch_and_save/fetch_bars unavailable")
