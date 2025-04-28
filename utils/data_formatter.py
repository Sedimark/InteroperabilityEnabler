"""
Data Formatter:
To convert the data expressed in various format (CSV, XLS, XLSX and NGSI-LD)
into the SEDIMARK internal format, i.e., pandas DataFrame.
NGSI-LD was selected as the primary format.

Author: Shahin ABDOUL SOUKOUR - Inria
Maintainer: Shahin ABDOUL SOUKOUR - Inria
"""

import json
import pandas as pd


def data_to_dataframe(file_path):
    """
    Read data from different file types (xls, xlsx, csv, json, jsonld) and
    convert them into a pandas DataFrame.

    Args:
        file_path (str): The path to the data file.

    Return:
        Pandas DataFrame.
    """
    df = None
    try:
        if file_path.endswith(".xls") or file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".json") or file_path.endswith(".jsonld"):
            with open(file_path, "r", encoding="utf-8") as file:
                json_data = json.load(file)

                # Handle JSON/JSON-LD data specifically
                if isinstance(json_data, list):
                    entities = json_data
                else:
                    entities = json_data.get(
                        "@graph", [json_data]
                    )  # Handle as a list of entities

                # Flatten the entities
                flattened_entities = [flatten_entity(entity) for entity in entities]
                df = pd.DataFrame(flattened_entities)
        else:
            raise ValueError(
                "Unsupported file format. Supported formats are xls, xlsx, json, jsonld, and csv."
            )
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return df


def flatten_dict(d, parent_key="", sep=".", preserve_keys=None):
    """
    Recursively flattens a nested dictionary into a flat dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key for recursion, used to create hierarchical keys.
        sep (str): The separator for nested keys (default is '.').
        preserve_keys (list): Keys whose values should not be flattened (default is None).

    Return:
        A flattened dictionary with keys representing the hierarchy.
    """
    if preserve_keys is None:
        preserve_keys = ["coordinates", "@context"]  # Keys to preserve as lists
    items = []
    for k, v in d.items():
        # Create the new key by appending current key to parent_key
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        # Recursively flatten if value is a dictionary
        if isinstance(v, dict):
            if k in preserve_keys:
                # Preserve the dictionary as-is if key is in preserve_keys
                items.append((new_key, v))
            else:
                items.extend(
                    flatten_dict(
                        v, new_key, sep=sep, preserve_keys=preserve_keys
                    ).items()
                )
        elif isinstance(v, list) and k in preserve_keys:
            # Preserve the list as-is
            items.append((new_key, v))
        elif isinstance(v, list):
            # Flatten lists unless the key is in preserve_keys
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    # Flatten each dictionary inside the list
                    items.extend(
                        flatten_dict(
                            item,
                            f"{new_key}[{i}]",
                            sep=sep,
                            preserve_keys=preserve_keys,
                        ).items()
                    )
                else:
                    # Handle primitive values in the list
                    items.append((f"{new_key}[{i}]", item))
        # Handle all other key-value pairs
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_entity(entity):
    """
    Flattens a single NGSI-LD entity by applying flatten_dict.

    Args:
        entity (dict): The NGSI-LD entity to flatten.

    Returns:
        dict: A flattened version of the entity.
    """
    return flatten_dict(entity)
