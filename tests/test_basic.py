import pandas as pd
import pytest
from ie.utils.data_formatter import data_to_dataframe
from ie.utils.annotation_dataset import add_quality_annotations_to_df
from ie.utils.data_mapper import data_conversion, restore_ngsi_ld_structure
from io import StringIO
from ie.utils.merge_data import merge_predicted_data
from ie.utils.extract_data import extract_columns
from ie.utils.add_metadata import add_metadata_to_predictions_from_dataframe


FILE_PATH_JSON = "tests/example_json.json"

# Expected values to validate
DATA = {
    "id": "urn:ngsild:Vehicle:vehicle:MobilityManagement:196671",
    "type": "Vehicle",
    "category.type": "Property",
    "category.value": "tracked",
    "vehicleNumber.type": "Property",
    "vehicleNumber.value": "379131",
    "battery[0].type": "Property",
    "battery[0].value": 1,
    "battery[0].observedAt": "2024-09-25T04:30:06Z",
    "battery[0].unitCode": "P1",
    "battery[1].type": "Property",
    "battery[1].value": 0.98,
    "battery[1].observedAt": "2024-09-24T16:42:24Z",
    "battery[1].unitCode": "P1",
    "location[0].type": "GeoProperty",
    "location[0].value.type": "Point",
    "location[0].value.coordinates": [43.460405, -3.853312],
    "location[0].observedAt": "2024-09-24T15:45:58Z",
    "location[1].type": "GeoProperty",
    "location[1].value.type": "Point",
    "location[1].value.coordinates": [43.459994, -3.820141],
    "location[1].observedAt": "2024-09-24T15:09:14Z",
    "@context": [
        "https://raw.githubusercontent.com/smart-data-models/dataModel.ERA/master/context.jsonld",
        "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context-v1.7.jsonld",
    ],
}


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
def test_json_to_dataframe(file_path):
    """
    Data Formatter component tests: JSON to DataFrame
    """
    print("\nJData Formatter component tests: JSON to DataFrame.")
    df = data_to_dataframe(file_path)
    assert isinstance(df, pd.DataFrame), f"{file_path} did not return a DataFrame"
    assert not df.empty, f"{file_path} returned an empty DataFrame"
    row = df.loc[0]
    for key, expected_value in DATA.items():
        assert key in row, f"Missing key '{key}' in DataFrame from {file_path}"
        assert (
            row[key] == expected_value
        ), f"Mismatch for '{key}' in {file_path}: expected {expected_value}, got {row[key]}"


def test_instance_level_annotation():
    """
    Data Quality Annotation component tests.
    Entire instance level annotation.
    """
    print(
        "\nData Quality Annotation component tests: entire instance level annotation."
    )
    df = pd.DataFrame(DATA)
    result = add_quality_annotations_to_df(
        data=df,
        entity_type="Vehicle",
        assessed_attrs=None,
    )
    assert "hasQuality.type" in result.columns
    assert "hasQuality.object" in result.columns
    assert result.loc[0, "hasQuality.type"] == "Relationship"
    assert (
        result.loc[0, "hasQuality.object"]
        == "urn:ngsi-ld:DataQualityAssessment:Vehicle:urn:ngsild:Vehicle:vehicle:MobilityManagement:196671"
    )


def test_attribute_level_annotation():
    """
    Data Quality Annotation component tests.
    Attribute level annotation.
    """
    print("\nData Quality Annotation component tests: attribute level annotation.")
    df = pd.DataFrame(DATA)
    result = add_quality_annotations_to_df(
        data=df, entity_type="Vehicle", assessed_attrs=["battery"]
    )

    for i in [0, 1]:  # because your example has battery[0] and battery[1]
        type_col = f"battery[{i}].hasQuality.type"
        object_col = f"battery[{i}].hasQuality.object"
        assert type_col in result.columns
        assert object_col in result.columns
        assert result.loc[0, type_col] == "Relationship"
        assert result.loc[0, object_col] == (
            f"urn:ngsi-ld:DataQualityAssessment:Vehicle:urn:ngsild:Vehicle:vehicle:MobilityManagement:196671:battery"
        )


def test_granular_level_annotation():
    """
    Data Quality Annotation component tests.
    Granular level annotation.
    """
    print("\nData Quality Annotation component tests: granular level annotation.")
    df = pd.DataFrame(DATA)
    result = add_quality_annotations_to_df(
        data=df, entity_type="Vehicle", assessed_attrs=["battery[0]"]
    )
    assert "battery[0].hasQuality.type" in result.columns
    assert "battery[0].hasQuality.object" in result.columns
    assert result.loc[0, "battery[0].hasQuality.type"] == "Relationship"
    assert (
        result.loc[0, "battery[0].hasQuality.object"]
        == "urn:ngsi-ld:DataQualityAssessment:Vehicle:urn:ngsild:Vehicle:vehicle:MobilityManagement:196671:battery"
    )


def test_data_mapper():
    """
    Data Mapper component tests.
    JSON to NGSI-LD
    """
    print("\nData Mapper component tests: JSON to NGSI-LD")
    df = pd.DataFrame([DATA])
    ngsi_ld_data = data_conversion(df)
    assert isinstance(ngsi_ld_data, list)
    assert len(ngsi_ld_data) == 1
    entity = ngsi_ld_data[0]
    assert entity["id"] == DATA["id"]
    assert entity["type"] == DATA["type"]
    assert entity["category"]["type"] == "Property"
    assert entity["category"]["value"] == "tracked"
    assert entity["vehicleNumber"]["type"] == "Property"
    assert entity["vehicleNumber"]["value"] == "379131"
    assert "battery[0]" in entity
    assert entity["battery[0]"]["type"] == "Property"
    assert entity["battery[0]"]["value"] == 1
    assert entity["battery[0]"]["observedAt"] == "2024-09-25T04:30:06Z"
    assert "location[1]" in entity
    assert entity["location[1]"]["value"]["type"] == "Point"
    assert entity["location[1]"]["value"]["coordinates"] == [43.459994, -3.820141]
    assert "@context" in entity


def test_restore_ngsi_ld_structure():
    """
    Data Mapper component tests.
    Restore NGSI-LD structure.
    """
    print("\nData Mapper component tests: NGSI-LD structure restoration.")
    df = pd.DataFrame([DATA])
    ngsi_ld_data = data_conversion(df)
    restored = restore_ngsi_ld_structure(ngsi_ld_data[0])
    assert "battery" in restored
    assert isinstance(restored["battery"], list)
    assert restored["battery"][0]["type"] == "Property"
    assert restored["battery"][1]["value"] == 0.98
    assert "location" in restored
    assert isinstance(restored["location"], list)
    assert restored["location"][1]["value"]["coordinates"] == [43.459994, -3.820141]


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
    assert merged.iloc[2]["windSpeed"] == "null"