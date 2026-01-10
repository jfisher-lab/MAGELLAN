"""
Snapshot test for training bio models.

This test ensures that the binary/nonbinary F1 and MCC metrics remain stable
across different runs of the training script.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest


def test_train_bio_metrics_snapshot_example_1():
    """
    Test that running train_bio.py with the example config produces expected metrics.
    
    This is a snapshot test that verifies the binary f1, binary mcc, nonbinary f1,
    and nonbinary mcc metrics remain consistent.
    """
    if not os.getenv("RUN_SLOW_TESTS"):
        pytest.skip("Set RUN_SLOW_TESTS=1 to run this test")
    # Path to the example config file
    config_path = Path(__file__).parent.parent / "scripts" / "example_configs" / "example_train_bio_config.toml"
    
    # Run the training script as subprocess
    cmd = [
        "uv", "run", "scripts/train_bio.py", 
        "--config", str(config_path)
    ]
    
    # Get timeout from environment variable or default to 300 seconds
    timeout_seconds = int(os.getenv("SLOW_TEST_TIMEOUT", "300"))
    
    result = subprocess.run(
        cmd, 
        cwd=Path(__file__).parent.parent,
        capture_output=True, 
        text=True,
        timeout=timeout_seconds
    )
    
    # Check that the command succeeded
    assert result.returncode == 0, f"Training script failed with error: {result.stderr}"
    
    # Parse metrics from stdout using regex
    output = result.stdout
    
    # Extract metrics using regex patterns
    binary_f1_match = re.search(r"binary f1: ([\d.]+)", output)
    binary_mcc_match = re.search(r"binary mcc: ([\d.]+)", output)
    nonbinary_f1_match = re.search(r"nonbinary f1: ([\d.]+)", output)
    nonbinary_mcc_match = re.search(r"nonbinary mcc: ([\d.]+)", output)
    
    # Ensure all metrics were found
    assert binary_f1_match, "Could not find binary f1 in output"
    assert binary_mcc_match, "Could not find binary mcc in output"
    assert nonbinary_f1_match, "Could not find nonbinary f1 in output"
    assert nonbinary_mcc_match, "Could not find nonbinary mcc in output"
    
    # Extract the actual values
    binary_f1 = float(binary_f1_match.group(1))
    binary_mcc = float(binary_mcc_match.group(1))
    nonbinary_f1 = float(nonbinary_f1_match.group(1))
    nonbinary_mcc = float(nonbinary_mcc_match.group(1))
    
    # Expected values (snapshot from successful run)
    # These are the baseline values that should remain stable
    expected_binary_f1 = 0.8941176470588236
    expected_binary_mcc = 0.6220755959148104
    expected_nonbinary_f1 = 0.7301551458477232
    expected_nonbinary_mcc = 0.5762594305134874
    
    # Assert metrics match expected values with reasonable tolerance
    # Using abs tolerance to account for minor floating point differences
    tolerance = 1e-10
    
    assert abs(binary_f1 - expected_binary_f1) < tolerance, (
        f"Binary F1 score changed: expected {expected_binary_f1}, got {binary_f1}"
    )
    
    assert abs(binary_mcc - expected_binary_mcc) < tolerance, (
        f"Binary MCC changed: expected {expected_binary_mcc}, got {binary_mcc}"
    )
    
    assert abs(nonbinary_f1 - expected_nonbinary_f1) < tolerance, (
        f"Nonbinary F1 score changed: expected {expected_nonbinary_f1}, got {nonbinary_f1}"
    )
    
    assert abs(nonbinary_mcc - expected_nonbinary_mcc) < tolerance, (
        f"Nonbinary MCC changed: expected {expected_nonbinary_mcc}, got {nonbinary_mcc}"
    )
    
    # Verify that metrics are reasonable values (sanity checks)
    assert 0 <= binary_f1 <= 1, f"Binary F1 should be between 0 and 1, got {binary_f1}"
    assert -1 <= binary_mcc <= 1, f"Binary MCC should be between -1 and 1, got {binary_mcc}"
    assert 0 <= nonbinary_f1 <= 1, f"Nonbinary F1 should be between 0 and 1, got {nonbinary_f1}"
    assert -1 <= nonbinary_mcc <= 1, f"Nonbinary MCC should be between -1 and 1, got {nonbinary_mcc}"


def test_train_bio_metrics_snapshot():
    """
    Test that running train_bio.py with the example config produces expected metrics.
    
    This is a snapshot test that verifies the binary f1, binary mcc, nonbinary f1,
    and nonbinary mcc metrics remain consistent.
    """
    if not os.getenv("RUN_SLOW_TESTS"):
        pytest.skip("Set RUN_SLOW_TESTS=1 to run this test")
    # Path to the example config file
    config_path = Path(__file__).parent.parent / "scripts" / "example_configs" / "example_train_bio_config_2.toml"
    
    # Run the training script as subprocess
    cmd = [
        "uv", "run", "scripts/train_bio.py", 
        "--config", str(config_path)
    ]
    
    # Get timeout from environment variable or default to 300 seconds
    timeout_seconds = int(os.getenv("SLOW_TEST_TIMEOUT", "300"))
    
    result = subprocess.run(
        cmd, 
        cwd=Path(__file__).parent.parent,
        capture_output=True, 
        text=True,
        timeout=timeout_seconds
    )
    
    # Check that the command succeeded
    assert result.returncode == 0, f"Training script failed with error: {result.stderr}"
    
    # Parse metrics from stdout using regex
    output = result.stdout
    
    # Extract metrics using regex patterns
    binary_f1_match = re.search(r"binary f1: ([\d.]+)", output)
    binary_mcc_match = re.search(r"binary mcc: ([\d.]+)", output)
    nonbinary_f1_match = re.search(r"nonbinary f1: ([\d.]+)", output)
    nonbinary_mcc_match = re.search(r"nonbinary mcc: ([\d.]+)", output)
    
    # Ensure all metrics were found
    assert binary_f1_match, "Could not find binary f1 in output"
    assert binary_mcc_match, "Could not find binary mcc in output"
    assert nonbinary_f1_match, "Could not find nonbinary f1 in output"
    assert nonbinary_mcc_match, "Could not find nonbinary mcc in output"
    
    # Extract the actual values
    binary_f1 = float(binary_f1_match.group(1))
    binary_mcc = float(binary_mcc_match.group(1))
    nonbinary_f1 = float(nonbinary_f1_match.group(1))
    nonbinary_mcc = float(nonbinary_mcc_match.group(1))
    
    # Expected values (snapshot from successful run)
    # These are the baseline values that should remain stable
    
    expected_binary_f1 = 0.8671679197994987
    expected_binary_mcc = 0.42362038717687917
    expected_nonbinary_f1 = 0.4482927379824205
    expected_nonbinary_mcc = 0.26522111455843395
    
    # Assert metrics match expected values with reasonable tolerance
    # Using abs tolerance to account for minor floating point differences
    tolerance = 1e-10
    
    assert abs(binary_f1 - expected_binary_f1) < tolerance, (
        f"Binary F1 score changed: expected {expected_binary_f1}, got {binary_f1}"
    )
    
    assert abs(binary_mcc - expected_binary_mcc) < tolerance, (
        f"Binary MCC changed: expected {expected_binary_mcc}, got {binary_mcc}"
    )
    
    assert abs(nonbinary_f1 - expected_nonbinary_f1) < tolerance, (
        f"Nonbinary F1 score changed: expected {expected_nonbinary_f1}, got {nonbinary_f1}"
    )
    
    assert abs(nonbinary_mcc - expected_nonbinary_mcc) < tolerance, (
        f"Nonbinary MCC changed: expected {expected_nonbinary_mcc}, got {nonbinary_mcc}"
    )
    
    # Verify that metrics are reasonable values (sanity checks)
    assert 0 <= binary_f1 <= 1, f"Binary F1 should be between 0 and 1, got {binary_f1}"
    assert -1 <= binary_mcc <= 1, f"Binary MCC should be between -1 and 1, got {binary_mcc}"
    assert 0 <= nonbinary_f1 <= 1, f"Nonbinary F1 should be between 0 and 1, got {nonbinary_f1}"
    assert -1 <= nonbinary_mcc <= 1, f"Nonbinary MCC should be between -1 and 1, got {nonbinary_mcc}"