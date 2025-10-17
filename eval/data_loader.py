"""Utility functions for loading evaluation data."""

import json
from pathlib import Path
from typing import Iterator, List, Union

from pydantic import ValidationError

from eval.schemas.data_models import Conversation, Query


def load_dataset(dataset_path: Path) -> List[Union[Query, Conversation]]:
    """
    Loads a dataset from a .jsonl file and validates each line against Query or Conversation models.

    It automatically detects whether each line is a single query or a conversation.

    Args:
        dataset_path: The path to the dataset .jsonl file.

    Returns:
        A list of validated Query or Conversation objects.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If a line is not valid JSON or does not match any supported schema.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at '{dataset_path}'")

    data_items = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                # Check if the data structure looks like a conversation
                if "turns" in data and "id" in data:
                    item = Conversation.model_validate(data)
                else:
                    item = Query.model_validate(data)
                data_items.append(item)
            except json.JSONDecodeError:
                raise ValueError(f"Error decoding JSON on line {i} in '{dataset_path}'")
            except ValidationError as e:
                raise ValueError(f"Validation error on line {i} in '{dataset_path}':\n{e}")
    return data_items


def stream_dataset(dataset_path: Path) -> Iterator[Union[Query, Conversation]]:
    """
    Streams a dataset from a .jsonl file, yielding one validated Query or Conversation object at a time.

    This is memory-efficient for large datasets and automatically detects the data type.

    Args:
        dataset_path: The path to the dataset .jsonl file.

    Yields:
        A validated Query or Conversation object for each line in the file.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If a line is not valid JSON or does not match any supported schema.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at '{dataset_path}'")

    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                # Check if the data structure looks like a conversation
                if "turns" in data and "id" in data:
                    item = Conversation.model_validate(data)
                else:
                    item = Query.model_validate(data)
                yield item
            except json.JSONDecodeError:
                raise ValueError(f"Error decoding JSON on line {i} in '{dataset_path}'")
            except ValidationError as e:
                raise ValueError(f"Validation error on line {i} in '{dataset_path}':\n{e}")

