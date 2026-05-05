"""Global constants for the nuT standalone package."""
import os.path

# Root directory of this package (nuT_no_graphnet/)
NUT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory (contains geometry tables, test data, example data)
DATA_DIR = os.path.join(NUT_ROOT_DIR, "data")

# Test data
TEST_DATA_DIR = os.path.join(DATA_DIR, "tests")
TEST_OUTPUT_DIR = os.path.join(TEST_DATA_DIR, "output")

_test_dataset_name = "oscNext_genie_level7_v02"
_test_dataset_file = f"{_test_dataset_name}_first_5_frames"
TEST_SQLITE_DATA = os.path.join(
    TEST_DATA_DIR, "sqlite", _test_dataset_name, f"{_test_dataset_file}.db"
)
TEST_PARQUET_DATA = os.path.join(
    TEST_DATA_DIR, "parquet", _test_dataset_name, "merged"
)

# Example data
EXAMPLE_DATA_DIR = os.path.join(DATA_DIR, "examples")
EXAMPLE_OUTPUT_DIR = os.path.join(EXAMPLE_DATA_DIR, "output")

# Configuration files
CONFIG_DIR = os.path.join(NUT_ROOT_DIR, "configs")
DATASETS_CONFIG_DIR = os.path.join(CONFIG_DIR, "datasets")
MODEL_CONFIG_DIR = os.path.join(CONFIG_DIR, "models")

# Geometry tables for each supported detector type
GEOMETRY_TABLE_DIR = os.path.join(DATA_DIR, "geometry_tables")
ICECUBE_GEOMETRY_TABLE_DIR = os.path.join(GEOMETRY_TABLE_DIR, "icecube")
PROMETHEUS_GEOMETRY_TABLE_DIR = os.path.join(GEOMETRY_TABLE_DIR, "prometheus")
LIQUIDO_GEOMETRY_TABLE_DIR = os.path.join(GEOMETRY_TABLE_DIR, "liquid-o")
KM3NeT_GEOMETRY_TABLE_DIR = os.path.join(GEOMETRY_TABLE_DIR, "km3net")
