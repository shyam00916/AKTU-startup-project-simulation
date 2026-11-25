"""Data simulation and leakage-free temporal feature extraction modules."""

from .features import (
    PreparedDatasets,
    SequenceDatasetArrays,
    build_customer_behavior_table,
    compute_point_in_time_records,
    prepare_datasets,
)
from .generator import (
    ARCHETYPES,
    INDUSTRIES,
    REGIONS,
    ReceivablesSimulator,
    generate_receivables_dataset,
)

__all__ = [
    "ARCHETYPES",
    "INDUSTRIES",
    "REGIONS",
    "PreparedDatasets",
    "ReceivablesSimulator",
    "SequenceDatasetArrays",
    "build_customer_behavior_table",
    "compute_point_in_time_records",
    "generate_receivables_dataset",
    "prepare_datasets",
]
