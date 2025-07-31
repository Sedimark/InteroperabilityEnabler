import pandas as pd
import numpy as np
import pytest
from InteroperabilityEnabler.utils.data_formatter import data_to_dataframe
from InteroperabilityEnabler.utils.annotation_dataset import add_quality_annotations_to_df
from io import StringIO
from InteroperabilityEnabler.utils.merge_data import merge_predicted_data
from InteroperabilityEnabler.utils.extract_data import extract_columns
from InteroperabilityEnabler.utils.add_metadata import add_metadata_to_predictions_from_dataframe
from InteroperabilityEnabler.utils.data_mapper import data_mapper
import json


FILE_PATH_JSON = "example_json.json"


MOCK_CSV = """
UnixTime,temperature,windSpeed # urn:ngsi-ld:Dataset:Open-Meteo:10MTR,windSpeed # urn:ngsi-ld:Dataset:Open-Meteo:80MTR,windSpeed # urn:ngsi-ld:Dataset:Open-Meteo:120MTR,windSpeed # urn:ngsi-ld:Dataset:Open-Meteo:180MTR
1713312000,-7.6,9.7,12.5,13.1,14.1
1713315600,-7.7,8,10.2,10.7,12.1
1713319200,-8.3,9,11.4,12.3,13
1713322800,-8.2,11.4,14.3,15.3,16.6
1713326400,-8.6,10.4,13.2,14.5,16.2
"""


PREDICTED_CSV = """
-7.6,9.7
-7.7,8
-8.3,9
-8.2,11.4
-8.6,10.4
-9.1,8.4
-8.4,14.9
-7.6,19.4
-6.6,21.3
"""


@pytest.mark.parametrize("file_path", [FILE_PATH_JSON])
def test_data_formatter(file_path):
    # Load the JSON file from disk
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Run the formatter
    context_df, time_series_df = data_to_dataframe(json_data, sep="__")

    # Assertions (use `assert`, not `assertEqual`)
    assert context_df.iloc[0]["id"] == "urn:sedimark:station:1"
    assert context_df.iloc[0]["type"] == "MonitoringSite"
    assert "pm10__value" in time_series_df.columns
    assert "pnci__value" in time_series_df.columns
    assert time_series_df["pm10__value"].iloc[3] == 22.9



@pytest.mark.parametrize("file_path", [FILE_PATH_JSON])
def test_instance_level_annotation(file_path):
    # Load JSON data
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Convert to DataFrames
    context_df, time_series_df = data_to_dataframe(json_data, sep="__")

    # Apply instance-level annotation
    updated_context_df, updated_time_series_df = add_quality_annotations_to_df(
            context_df, time_series_df, sep="__", assessed_attrs=None
    )

    # Assertions for context-level quality annotation
    assert "hasQuality__type" in updated_context_df.columns
    assert "hasQuality__object" in updated_context_df.columns

    assert updated_context_df.loc[0, "hasQuality__type"] == "Relationship"
    assert updated_context_df.loc[0, "hasQuality__object"] == (
        "urn:ngsi-ld:DataQualityAssessment:MonitoringSite:urn:sedimark:station:1"
    )

    # Time-series DataFrame should remain unchanged
    assert "pm10__hasQuality__type" not in updated_time_series_df.columns



@pytest.mark.parametrize("file_path", [FILE_PATH_JSON])
def test_attribute_level_annotation(file_path):
    # Load JSON data from file
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Convert to DataFrames
    context_df, time_series_df = data_to_dataframe(json_data, sep="__")

    # Apply attribute-level annotation on 'pm10'
    updated_context_df, updated_time_series_df = add_quality_annotations_to_df(
        context_df,
        time_series_df,
        sep="__",
        assessed_attrs=["pm10"]
    )

    # Check that new quality columns are added for 'pm10'
    assert "pm10__hasQuality__type" in updated_time_series_df.columns
    assert "pm10__hasQuality__object" in updated_time_series_df.columns

    # Ensure all annotated rows have correct values
    expected_object_uri = (
        "urn:ngsi-ld:DataQualityAssessment:MonitoringSite:urn:sedimark:station:1:pm10"
    )

    for i in range(len(updated_time_series_df)):
        has_value = pd.notna(updated_time_series_df.loc[i, "pm10__value"])
        expected_type = "Relationship" if has_value else None
        expected_obj = expected_object_uri if has_value else None

        assert updated_time_series_df.loc[i, "pm10__hasQuality__type"] == expected_type
        assert updated_time_series_df.loc[i, "pm10__hasQuality__object"] == expected_obj

    # Confirm context_df is unchanged (no instance-level fields)
    assert "hasQuality__type" not in updated_context_df.columns



@pytest.mark.parametrize("file_path", [FILE_PATH_JSON])
def test_data_mapper(file_path):
    # Load JSON data
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Format data
    context_df, time_series_df = data_to_dataframe(json_data, sep="__")

    # Apply attribute-level annotation on 'no2'
    context_df, time_series_df = add_quality_annotations_to_df(
        context_df,
        time_series_df,
        sep="__",
        assessed_attrs=["no2"]
    )

    # Map back to JSON structure
    mapped_data = data_mapper(context_df, time_series_df, sep="__")

    # Assertions
    assert isinstance(mapped_data, dict)
    assert mapped_data["id"] == "urn:sedimark:station:1"
    assert mapped_data["type"] == "MonitoringSite"
    assert "no2" in mapped_data

    # Check at least one annotation exists for 'no2'
    no2_values = mapped_data["no2"]
    assert isinstance(no2_values, list)

    found_annotated = any(
        "hasQuality" in item and
        item["hasQuality"]["type"] == "Relationship" and
        item["hasQuality"]["object"].endswith(":no2")
        for item in no2_values
    )
    assert found_annotated, "No attribute-level annotation found for no2"


def test_extract_columns_valid_indices():
    """
    Data Extractor component tests.
    Columns extraction from CSV format with valid indices.
    """
    print(
        "\nData Extractor component tests: extract columns test from CSV format with valid indices."
    )
    df = pd.read_csv(StringIO(MOCK_CSV))
    selected_df, col_names = extract_columns(df, [0, 2, 4])
    assert selected_df.shape[1] == 3
    assert col_names == [
        "UnixTime",
        "windSpeed # urn:ngsi-ld:Dataset:Open-Meteo:10MTR",
        "windSpeed # urn:ngsi-ld:Dataset:Open-Meteo:120MTR",
    ]
    assert selected_df.iloc[0, 0] == 1713312000
    assert selected_df.iloc[0, 1] == 9.7
    assert selected_df.iloc[0, 2] == 13.1


def test_extract_columns_invalid_indices():
    """
    Data Extractor component tests.
    Columns extraction from CSV format with invalid indices.
    """
    print(
        "\nData Extractor component tests: extract columns test from CSV file with invalid index."
    )
    df = pd.read_csv(StringIO(MOCK_CSV))
    selected_df, col_names = extract_columns(df, [0, 100])  # Invalid index
    assert selected_df.empty
    assert col_names == []


def test_add_metadata_correct_columns():
    """
    Metadata Restorer component tests.
    Correct columns test.
    """
    print("\nMetadata Restorer component tests: correct columns.")
    df = pd.read_csv(StringIO(PREDICTED_CSV), header=None)
    column_names = ["temperature", "windSpeed"]
    result_df = add_metadata_to_predictions_from_dataframe(df, column_names)
    assert list(result_df.columns) == column_names
    assert result_df.shape == (9, 2)
    assert result_df.loc[0, "temperature"] == -7.6
    assert result_df.loc[0, "windSpeed"] == 9.7


def test_add_metadata_column_mismatch():
    """
    Metadata Restorer component tests.
    Column mismatch test.
    """
    print("\nMetadata Restorer component tests: column mismatch.")
    df = pd.read_csv(StringIO(PREDICTED_CSV), header=None)
    column_names = ["temperature"]  # only one column name, mismatch
    result_df = add_metadata_to_predictions_from_dataframe(df, column_names)
    assert result_df.empty


def test_add_metadata_empty_input():
    """
    Metadata Restorer component tests.
    Empty input test.
    """
    print("\nMetadata Restorer component tests: empty input.")
    df = pd.DataFrame()
    column_names = []
    result_df = add_metadata_to_predictions_from_dataframe(df, column_names)
    assert result_df.empty


def test_merge_predicted_data_matching_columns():
    """
    Data Merger component tests.
    Merge predicted data with matching columns.
    """
    print("\nData Merger component tests: merge predicted data.")
    df_initial = pd.DataFrame({"temperature": [-7.6, -7.7], "windSpeed": [9.7, 8]})
    predicted_df = pd.DataFrame(
        {"temperature": [-6.6, -6.1], "windSpeed": [10.2, 11.3]}
    )
    merged = merge_predicted_data(df_initial, predicted_df)
    assert merged.shape == (4, 2)
    assert list(merged.columns) == ["temperature", "windSpeed"]
    assert merged.iloc[2]["temperature"] == -6.6


def test_merge_predicted_data_predicted_missing_column():
    """
    Data Merger component tests.
    Merge predicted data with missing column.
    """
    print("\nData Merger component tests: merge predicted data with missing column.")
    df_initial = pd.DataFrame({"temperature": [-7.6, -7.7], "windSpeed": [9.7, 8]})
    predicted_df = pd.DataFrame({"temperature": [-6.6, -6.1]})  # missing windSpeed
    merged = merge_predicted_data(df_initial, predicted_df)
    assert "windSpeed" in merged.columns
    # Proper way to check for NaN values
    assert pd.isna(merged.iloc[2]["windSpeed"]) # This checks if the value is NaN