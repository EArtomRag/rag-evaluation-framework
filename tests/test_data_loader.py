"""Unit tests for the data_loader module."""

import json
from pathlib import Path
import pytest

from eval.data_loader import load_dataset
from eval.schemas.data_models import Query

# Sample valid data for testing
VALID_QUERIES_DATA = [
    {
        "id": "Q-0001",
        "domanda": "Test question 1?",
        "tipo_domanda": "puntuale",
        "gold_answer": "Answer 1",
        "gold_citations": ["doc1"],
    },
    {
        "id": "Q-0002",
        "domanda": "Test question 2?",
        "tipo_domanda": "inferenziale",
        "gold_citations": ["doc2", "doc3"],
    },
]


@pytest.fixture
def create_test_jsonl(tmp_path: Path) -> Path:
    """Creates a temporary .jsonl file for testing."""
    file_path = tmp_path / "test_queries.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for item in VALID_QUERIES_DATA:
            f.write(json.dumps(item) + "\n")
    return file_path


def test_load_dataset_success(create_test_jsonl: Path):
    """Tests that load_dataset successfully loads and parses a valid .jsonl file."""
    # Act
    queries = load_dataset(create_test_jsonl)

    # Assert
    assert len(queries) == 2
    assert all(isinstance(q, Query) for q in queries)
    assert queries[0].id == "Q-0001"
    assert queries[1].domanda == "Test question 2?"
    assert queries[1].gold_citations == ["doc2", "doc3"]


def test_load_dataset_file_not_found():
    """Tests that load_dataset raises FileNotFoundError for a non-existent file."""
    # Arrange
    non_existent_path = Path("non_existent_file.jsonl")

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        load_dataset(non_existent_path)


def test_load_dataset_invalid_json(tmp_path: Path):
    """Tests that load_dataset raises ValueError for a file with invalid JSON."""
    # Arrange
    file_path = tmp_path / "invalid.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write('{"id": "Q-0001", "domanda": "valid", "tipo_domanda": "puntuale"}\n')
        f.write("{this is not json}\n")

    # Act & Assert
    with pytest.raises(ValueError, match="Error decoding JSON on line 2"):
        load_dataset(file_path)


def test_load_dataset_validation_error(tmp_path: Path):
    """Tests that load_dataset raises ValueError for data that fails Pydantic validation."""
    # Arrange
    file_path = tmp_path / "invalid_schema.jsonl"
    invalid_data = {"id": "Q-0003", "domanda": "Missing type"}  # tipo_domanda is missing
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(invalid_data) + "\n")

    # Act & Assert
    with pytest.raises(ValueError, match="Validation error on line 1"):
        load_dataset(file_path)

