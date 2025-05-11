from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from magellan.plot import analyze_predictions


def test_analyze_binary_predictions_basic():
    """Test basic functionality with simple inputs."""
    # Create test data
    y_true = pd.DataFrame({"exp1": [0, 1, np.nan, 2], "exp2": [2, 0, 1, np.nan]})
    y_pred = pd.DataFrame({"exp1": [0.1, 0.9, 0.0, 1.8], "exp2": [1.9, 0.2, 0.8, 0.0]})

    # Run function
    metrics = analyze_predictions(
        y_true=y_true,
        y_pred=y_pred,
        save_figs=False,
        save_csv=False,
        binary_mode=True,
    )

    # Basic assertions
    assert isinstance(metrics, dict)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert metrics["total_predictions"] == 6  # Number of non-NaN values


def test_analyze_binary_predictions_perfect():
    """Test with perfect predictions."""
    y_true = pd.DataFrame({"exp1": [0, 2, 1], "exp2": [1, 0, 2]})
    y_pred = pd.DataFrame({"exp1": [0.1, 1.9, 0.9], "exp2": [0.9, 0.1, 1.8]})

    metrics = analyze_predictions(
        y_true=y_true,
        y_pred=y_pred,
        save_figs=False,
        save_csv=False,
        binary_mode=True,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mcc"] == 1.0


def test_analyze_binary_predictions_all_wrong():
    """Test with completely wrong predictions."""
    y_true = pd.DataFrame({"exp1": [0, 2], "exp2": [1, 0]})
    y_pred = pd.DataFrame({"exp1": [1.9, 0.1], "exp2": [0.1, 0.9]})

    metrics = analyze_predictions(
        y_true=y_true,
        y_pred=y_pred,
        save_figs=False,
        save_csv=False,
        binary_mode=True,
    )

    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary directory for output files."""
    return tmp_path


def test_file_outputs(temp_output_dir):
    """Test that files are created when save_figs and save_csv are True."""
    y_true = pd.DataFrame({"exp1": [0, 1], "exp2": [2, 0]}, index=["node1", "node2"])
    y_pred = pd.DataFrame(
        {"exp1": [0.1, 0.9], "exp2": [1.8, 0.2]}, index=["node1", "node2"]
    )
    pert_dic = OrderedDict(
        {
            "exp1": {"pert": {"node1": 1}, "exp": {"node2": 0}},
            "exp2": {"pert": {"node2": 1}, "exp": {"node1": 0}},
        }
    )
    analyze_predictions(
        y_true=y_true,
        y_pred=y_pred,
        save_figs=True,
        save_csv=True,
        pert_dic=pert_dic,
        path_data=str(temp_output_dir),
        plot_all_nodes=True,
        binary_mode=True,
    )

    # Check that files were created
    # assert (temp_output_dir / "confusion_matrix_overall.html").exists()
    assert (temp_output_dir / "confusion_matrix_overall.png").exists()
    assert (temp_output_dir / "prediction_metrics.csv").exists()
    # assert (temp_output_dir / "confusion_matrix_node1.html").exists()
    # assert (temp_output_dir / "confusion_matrix_node2.html").exists()


def test_config_integration():
    """Test integration with config dictionary."""
    config = {
        "model_params": {"param1": 1, "param2": "test"},
        "training": {"epochs": 100},
        "paths": {"output": "/tmp"},
    }

    y_true = pd.DataFrame({"exp1": [0, 1]})
    y_pred = pd.DataFrame({"exp1": [0.1, 0.9]})

    metrics = analyze_predictions(
        y_true=y_true,
        y_pred=y_pred,
        save_figs=False,
        save_csv=False,
        config=config,
        binary_mode=True,
    )

    assert "model_params_param1" in metrics
    assert metrics["model_params_param1"] == 1
    assert metrics["training_epochs"] == 100


def test_error_handling():
    """Test error handling."""
    y_true = pd.DataFrame({"exp1": [0, 1]})
    y_pred = pd.DataFrame({"exp1": [0.1, 0.9]})

    # Test missing path_data when save_figs is True
    with pytest.raises(
        ValueError, match="path_data must be provided if save_figs is True"
    ):
        analyze_predictions(
            y_true=y_true,
            y_pred=y_pred,
            save_figs=True,
            save_csv=False,
            binary_mode=True,
        )

    # Test missing path_data when save_csv is True
    with pytest.raises(
        ValueError, match="path_data must be provided if save_csv is True"
    ):
        analyze_predictions(
            y_true=y_true,
            y_pred=y_pred,
            save_figs=False,
            save_csv=True,
            binary_mode=True,
        )


def test_mismatched_dataframes():
    """Test handling of mismatched DataFrames."""
    y_true = pd.DataFrame({"exp1": [0, 1]})
    y_pred = pd.DataFrame({"exp2": [0.1, 0.9]})  # Different column name

    with pytest.raises(ValueError):
        analyze_predictions(
            y_true=y_true,
            y_pred=y_pred,
            save_figs=False,
            save_csv=False,
            binary_mode=True,
        )


def test_all_nan_input():
    """Test handling of all-NaN input."""
    y_true = pd.DataFrame({"exp1": [np.nan, np.nan]})
    y_pred = pd.DataFrame({"exp1": [np.nan, np.nan]})

    with pytest.raises(ValueError):
        analyze_predictions(
            y_true=y_true,
            y_pred=y_pred,
            save_figs=False,
            save_csv=False,
            binary_mode=True,
        )


def test_multiclass_basic():
    """Test basic multiclass functionality."""
    y_true = pd.DataFrame({"exp1": [0, 1, 2, np.nan], "exp2": [2, 0, 1, 1]})
    y_pred = pd.DataFrame({"exp1": [0.1, 0.9, 1.8, 0.0], "exp2": [1.9, 0.2, 0.8, 0.9]})

    metrics = analyze_predictions(
        y_true=y_true, y_pred=y_pred, save_figs=False, save_csv=False, binary_mode=False
    )

    assert isinstance(metrics, dict)
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["total_predictions"] == 7
    assert metrics["roc_auc"] == "NA"  # ROC AUC not applicable for multiclass
    assert metrics["specificity"] == "NA"
    assert metrics["average_precision"] == "NA"


def test_multiclass_perfect():
    """Test perfect multiclass predictions."""
    y_true = pd.DataFrame({"exp1": [0, 1, 2], "exp2": [2, 1, 0]})
    y_pred = pd.DataFrame({"exp1": [0.1, 1.1, 1.9], "exp2": [1.9, 0.9, 0.1]})

    metrics = analyze_predictions(
        y_true=y_true, y_pred=y_pred, save_figs=False, save_csv=False, binary_mode=False
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_multiclass_weighted_metrics():
    """Test weighted metrics with imbalanced classes."""
    y_true = pd.DataFrame(
        {
            "exp1": [0, 0, 0, 1, 2],  # Imbalanced classes
            "exp2": [0, 0, 1, 1, 2],
        }
    )
    y_pred = pd.DataFrame(
        {"exp1": [0.1, 0.2, 0.1, 1.1, 1.9], "exp2": [0.2, 0.1, 0.9, 0.8, 1.8]}
    )

    metrics = analyze_predictions(
        y_true=y_true, y_pred=y_pred, save_figs=False, save_csv=False, binary_mode=False
    )

    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert metrics["specificity"] == "NA"  # Not applicable for multiclass


def test_multiclass_node_plots(temp_output_dir):
    """Test node-specific confusion matrix plots for multiclass."""
    y_true = pd.DataFrame(
        {"exp1": [0, 1, 2], "exp2": [2, 1, 0]}, index=["node1", "node2", "node3"]
    )
    y_pred = pd.DataFrame(
        {"exp1": [0.1, 0.9, 1.8], "exp2": [1.9, 1.1, 0.2]},
        index=["node1", "node2", "node3"],
    )

    analyze_predictions(
        y_true=y_true,
        y_pred=y_pred,
        save_figs=True,
        save_csv=False,
        path_data=str(temp_output_dir),
        plot_all_nodes=True,
        binary_mode=False,
    )

    assert (temp_output_dir / "confusion_matrix_overall.png").exists()
    for node in ["node1", "node3"]:
        assert (temp_output_dir / f"confusion_matrix_{node}.png").exists()
    assert (temp_output_dir / "confusion_matrix_node2.png").exists()


def test_insufficient_classes():
    """Test handling of nodes with insufficient unique classes."""
    y_true = pd.DataFrame(
        {
            "exp1": [0, 0, 0],  # Only one class
            "exp2": [0, 0, 1],
        },
        index=["node1", "node2", "node3"],
    )
    y_pred = pd.DataFrame(
        {"exp1": [0.1, 0.2, 0.1], "exp2": [0.1, 0.2, 0.9]},
        index=["node1", "node2", "node3"],
    )

    metrics = analyze_predictions(
        y_true=y_true, y_pred=y_pred, save_figs=False, save_csv=False, binary_mode=False
    )

    assert isinstance(metrics, dict)
    assert 0 <= metrics["accuracy"] <= 1


def test_mixed_binary_multiclass():
    """Test switching between binary and multiclass modes."""
    y_true = pd.DataFrame({"exp1": [0, 1, 2], "exp2": [1, 0, 1]})
    y_pred = pd.DataFrame({"exp1": [0.1, 0.9, 1.8], "exp2": [0.9, 0.2, 0.8]})

    # Binary mode
    binary_metrics = analyze_predictions(
        y_true=y_true, y_pred=y_pred, save_figs=False, save_csv=False, binary_mode=True
    )

    # Multiclass mode
    multiclass_metrics = analyze_predictions(
        y_true=y_true, y_pred=y_pred, save_figs=False, save_csv=False, binary_mode=False
    )

    assert binary_metrics["roc_auc"] != "NA"
    assert multiclass_metrics["roc_auc"] == "NA"
    assert (
        binary_metrics["total_predictions"] == multiclass_metrics["total_predictions"]
    )


def test_mismatched_nans_binary():
    """Test handling of mismatched NaN patterns in binary mode."""
    y_true = pd.DataFrame({"exp1": [0, 1, np.nan], "exp2": [1, 0, 1]})
    y_pred = pd.DataFrame({"exp1": [0.1, np.nan, 0.8], "exp2": [0.9, 0.2, 0.8]})

    with pytest.raises(ValueError):
        analyze_predictions(
            y_true=y_true,
            y_pred=y_pred,
            save_figs=False,
            save_csv=False,
            binary_mode=True,
        )


def test_mismatched_nans_multiclass():
    """Test handling of mismatched NaN patterns in multiclass mode."""
    y_true = pd.DataFrame({"exp1": [0, 1, 2, np.nan], "exp2": [2, np.nan, 1, 1]})
    y_pred = pd.DataFrame(
        {"exp1": [0.1, 0.9, np.nan, 1.8], "exp2": [1.9, 0.2, 0.8, np.nan]}
    )

    with pytest.raises(ValueError):
        analyze_predictions(
            y_true=y_true,
            y_pred=y_pred,
            save_figs=False,
            save_csv=False,
            binary_mode=False,
        )
