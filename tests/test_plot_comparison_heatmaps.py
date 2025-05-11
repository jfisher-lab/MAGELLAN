import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from magellan.plot import plot_comparison_heatmaps


def test_plot_comparison_heatmaps_basic(monkeypatch):
    """Test basic functionality with simple inputs"""
    # Mock data
    data1 = pd.DataFrame(np.random.randn(3, 3))
    data2 = pd.DataFrame(np.random.randn(3, 3))
    data3 = pd.DataFrame(np.random.randn(3, 3))
    
    to_plot = [(data1, "Plot 1"), (data2, "Plot 2"), (data3, "Plot 3")]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_heatmap.png"
        
        # Mock savefig to avoid actual file creation
        def mock_savefig(*args, **kwargs):
            pass
            
        monkeypatch.setattr("matplotlib.pyplot.savefig", mock_savefig)
        
        plot_comparison_heatmaps(to_plot, output_path)

def test_plot_comparison_heatmaps_custom_params():
    """Test with custom parameters"""
    data1 = pd.DataFrame(np.random.randn(3, 3))
    data2 = pd.DataFrame(np.random.randn(3, 3))
    data3 = pd.DataFrame(np.random.randn(3, 3))
    
    to_plot = [(data1, "Plot 1"), (data2, "Plot 2"), (data3, "Plot 3")]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_heatmap.png"
        
        plot_comparison_heatmaps(
            to_plot,
            output_path,
            cmap="viridis",
            center=0.5,
            vmin=-2,
            vmax=2,
            figsize=(12, 4),
            xticklabels=["A", "B", "C"],
            yticklabels=["X", "Y", "Z"],
            annot=True,
            fmt=".2f"
        )

def test_plot_comparison_heatmaps_invalid_input():
    """Test handling of invalid inputs"""
    with pytest.raises(ValueError):
        # Empty to_plot list
        plot_comparison_heatmaps([], "output.png")
    
    with pytest.raises(ValueError):
        # Wrong number of plots (should be 3)
        data = pd.DataFrame(np.random.randn(3, 3))
        plot_comparison_heatmaps([(data, "Plot 1")], "output.png")

def test_plot_comparison_heatmaps_different_sizes():
    """Test with differently sized dataframes"""
    data1 = pd.DataFrame(np.random.randn(2, 2))
    data2 = pd.DataFrame(np.random.randn(3, 3))
    data3 = pd.DataFrame(np.random.randn(4, 4))
    
    to_plot = [(data1, "Plot 1"), (data2, "Plot 2"), (data3, "Plot 3")]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_heatmap.png"
        plot_comparison_heatmaps(to_plot, output_path)

def test_plot_comparison_heatmaps_file_extension():
    """Test different file extensions"""
    data = pd.DataFrame(np.random.randn(3, 3))
    to_plot = [(data, "Plot 1"), (data, "Plot 2"), (data, "Plot 3")]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test various extensions
        extensions = [".png", ".jpg", ".pdf", ".svg"]
        for ext in extensions:
            output_path = Path(tmpdir) / f"test_heatmap{ext}"
            plot_comparison_heatmaps(to_plot, output_path)