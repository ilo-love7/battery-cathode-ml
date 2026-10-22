"""Pytest configuration and fixtures for battery-cathode-ml tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_formulas():
    """Sample chemical formulas for testing."""
    return [
        "LiFePO4",
        "LiCoO2",
        "LiMn2O4",
        "LiNi0.33Co0.33Mn0.34O2",
        "Li3V2(PO4)3",
    ]


@pytest.fixture
def sample_voltage_data():
    """Sample voltage data DataFrame for testing."""
    data = {
        "formula": ["LiFePO4", "LiCoO2", "LiMn2O4"],
        "average_voltage_V": [3.4, 3.9, 4.0],
        "specific_capacity_mAh_g": [170, 140, 120],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_results_dir(tmp_path):
    """Create a temporary directory for test results."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir


@pytest.fixture
def element_list_fixture():
    """Test element list for feature extraction.

    注意：fixture **不能**命名为 test_* —— pytest 会把任何 test_* 函数
    同时当作测试收集（虚增计数），这里用 element_list_fixture 避免该反模式。
    """
    return ["Li", "Fe", "P", "O", "Co", "Mn", "Ni", "V"]


@pytest.fixture
def mock_training_data(sample_formulas):
    """Mock training data for model testing."""
    n_samples = len(sample_formulas)
    return {
        "formulas": sample_formulas,
        "y_gap": np.random.uniform(0.5, 3.0, n_samples),
        "y_ef": np.random.uniform(-5.0, -1.0, n_samples),
        "y_voltage": np.random.uniform(3.0, 4.5, n_samples),
    }
