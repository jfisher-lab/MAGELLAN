"""Tests for weight initialization functionality."""

import pytest
import torch

from magellan.synthetic import PruningTestConfig


@pytest.fixture
def fixed_config():
    """Configuration for fixed weight initialization."""
    config = PruningTestConfig()
    config.use_random_weight_init = False
    config.edge_weight_init = 0.5
    return config


@pytest.fixture
def random_config():
    """Configuration for random weight initialization."""
    config = PruningTestConfig()
    config.use_random_weight_init = True
    config.random_weight_init_distribution = "uniform"
    config.random_weight_init_lower = 0.0
    config.random_weight_init_upper = 2.0
    return config


def create_edge_weights(config, n_edges=100):
    """Helper function to create edge weights based on configuration."""
    if config.use_random_weight_init:
        if config.random_weight_init_distribution == "uniform":
            edge_weight = (
                torch.rand(n_edges)
                * (config.random_weight_init_upper - config.random_weight_init_lower)
                + config.random_weight_init_lower
            )
        else:
            raise ValueError(
                f"Unsupported random weight initialization distribution: {config.random_weight_init_distribution}"
            )
    else:
        edge_weight = torch.ones(n_edges) * config.edge_weight_init
    return edge_weight


class TestFixedWeightInitialization:
    """Test suite for fixed weight initialization."""

    def test_fixed_initialization_default(self, fixed_config):
        """Test that fixed initialization creates identical weights."""
        n_edges = 50
        edge_weight = create_edge_weights(fixed_config, n_edges)

        assert edge_weight.shape == (n_edges,)
        assert torch.all(edge_weight == 0.5)

    def test_fixed_initialization_custom_value(self):
        """Test fixed initialization with custom value."""
        config = PruningTestConfig()
        config.use_random_weight_init = False
        config.edge_weight_init = 1.25

        n_edges = 30
        edge_weight = create_edge_weights(config, n_edges)

        assert torch.all(edge_weight == 1.25)

    def test_fixed_initialization_zero(self):
        """Test fixed initialization with zero value."""
        config = PruningTestConfig()
        config.use_random_weight_init = False
        config.edge_weight_init = 0.0

        n_edges = 20
        edge_weight = create_edge_weights(config, n_edges)

        assert torch.all(edge_weight == 0.0)


class TestRandomWeightInitialization:
    """Test suite for random weight initialization."""

    def test_random_initialization_bounds(self, random_config):
        """Test that random weights fall within specified bounds."""
        torch.manual_seed(42)  # For reproducible test
        n_edges = 100
        edge_weight = create_edge_weights(random_config, n_edges)

        assert edge_weight.shape == (n_edges,)
        assert torch.all(edge_weight >= random_config.random_weight_init_lower)
        assert torch.all(edge_weight <= random_config.random_weight_init_upper)

    def test_random_initialization_diversity(self, random_config):
        """Test that random weights are actually different."""
        torch.manual_seed(42)
        n_edges = 100
        edge_weight = create_edge_weights(random_config, n_edges)

        # Check that not all weights are the same
        unique_weights = torch.unique(edge_weight)
        assert len(unique_weights) > 10  # Should have many unique values

        # Check that we don't get the same values twice with different seeds
        torch.manual_seed(123)
        edge_weight2 = create_edge_weights(random_config, n_edges)
        assert not torch.allclose(edge_weight, edge_weight2)

    def test_random_initialization_statistics(self, random_config):
        """Test statistical properties of random initialization."""
        torch.manual_seed(42)
        n_edges = 1000  # Large sample for statistical test
        edge_weight = create_edge_weights(random_config, n_edges)

        # Mean should be approximately midpoint of range
        expected_mean = (
            random_config.random_weight_init_lower
            + random_config.random_weight_init_upper
        ) / 2
        actual_mean = edge_weight.mean().item()
        assert abs(actual_mean - expected_mean) < 0.1  # Allow some variance

        # Standard deviation should be roughly range/sqrt(12) for uniform distribution
        expected_std = (
            random_config.random_weight_init_upper
            - random_config.random_weight_init_lower
        ) / (12**0.5)
        actual_std = edge_weight.std().item()
        assert abs(actual_std - expected_std) < 0.1

    def test_random_initialization_custom_bounds(self):
        """Test random initialization with custom bounds."""
        config = PruningTestConfig()
        config.use_random_weight_init = True
        config.random_weight_init_distribution = "uniform"
        config.random_weight_init_lower = -0.5
        config.random_weight_init_upper = 1.5

        torch.manual_seed(42)
        n_edges = 100
        edge_weight = create_edge_weights(config, n_edges)

        assert torch.all(edge_weight >= -0.5)
        assert torch.all(edge_weight <= 1.5)

    def test_random_initialization_narrow_range(self):
        """Test random initialization with narrow range."""
        config = PruningTestConfig()
        config.use_random_weight_init = True
        config.random_weight_init_distribution = "uniform"
        config.random_weight_init_lower = 0.9
        config.random_weight_init_upper = 1.1

        torch.manual_seed(42)
        n_edges = 50
        edge_weight = create_edge_weights(config, n_edges)

        assert torch.all(edge_weight >= 0.9)
        assert torch.all(edge_weight <= 1.1)
        assert edge_weight.std().item() < 0.1  # Should have low variance


class TestConfigurationValidation:
    """Test suite for configuration validation."""

    def test_invalid_distribution(self):
        """Test error handling for invalid distribution type."""
        config = PruningTestConfig()
        config.use_random_weight_init = True
        config.random_weight_init_distribution = "invalid"
        config.random_weight_init_lower = 0.0
        config.random_weight_init_upper = 2.0

        with pytest.raises(
            ValueError, match="Unsupported random weight initialization distribution"
        ):
            create_edge_weights(config, 10)

    def test_normal_distribution_not_implemented(self):
        """Test that normal distribution raises appropriate error."""
        config = PruningTestConfig()
        config.use_random_weight_init = True
        config.random_weight_init_distribution = "normal"
        config.random_weight_init_lower = 0.0
        config.random_weight_init_upper = 2.0

        with pytest.raises(
            ValueError, match="Unsupported random weight initialization distribution"
        ):
            create_edge_weights(config, 10)


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_single_edge(self, random_config):
        """Test initialization with single edge."""
        torch.manual_seed(42)
        edge_weight = create_edge_weights(random_config, 1)

        assert edge_weight.shape == (1,)
        assert (
            random_config.random_weight_init_lower
            <= edge_weight.item()
            <= random_config.random_weight_init_upper
        )

    def test_zero_edges(self, random_config):
        """Test initialization with zero edges."""
        edge_weight = create_edge_weights(random_config, 0)

        assert edge_weight.shape == (0,)

    def test_large_number_edges(self, random_config):
        """Test initialization with large number of edges."""
        torch.manual_seed(42)
        n_edges = 10000
        edge_weight = create_edge_weights(random_config, n_edges)

        assert edge_weight.shape == (n_edges,)
        assert torch.all(edge_weight >= random_config.random_weight_init_lower)
        assert torch.all(edge_weight <= random_config.random_weight_init_upper)


class TestTrainBioCompatibility:
    """Test that the logic works with train_bio.py parameter structure."""

    def test_train_bio_config_structure(self):
        """Test weight initialization using train_bio.py config structure."""
        # Simulate train_bio.py config parsing
        use_random_weight_init = False
        edge_weight_init = 0.7
        random_weight_init_distribution = "uniform"
        random_weight_init_lower = 0.0
        random_weight_init_upper = 2.0

        n_edges = 50

        # Test fixed initialization (train_bio.py style)
        if use_random_weight_init:
            if random_weight_init_distribution == "uniform":
                edge_weight = (
                    torch.rand(n_edges)
                    * (random_weight_init_upper - random_weight_init_lower)
                    + random_weight_init_lower
                )
            else:
                raise ValueError(
                    f"Unsupported random weight initialization distribution: {random_weight_init_distribution}"
                )
        else:
            edge_weight = torch.ones(n_edges) * edge_weight_init

        assert torch.all(edge_weight == 0.7)

    def test_train_bio_random_config(self):
        """Test random initialization using train_bio.py config structure."""
        use_random_weight_init = True
        edge_weight_init = 0.5  # Should be ignored
        random_weight_init_distribution = "uniform"
        random_weight_init_lower = -1.0
        random_weight_init_upper = 3.0

        torch.manual_seed(42)
        n_edges = 100

        if use_random_weight_init:
            if random_weight_init_distribution == "uniform":
                edge_weight = (
                    torch.rand(n_edges)
                    * (random_weight_init_upper - random_weight_init_lower)
                    + random_weight_init_lower
                )
            else:
                raise ValueError(
                    f"Unsupported random weight initialization distribution: {random_weight_init_distribution}"
                )
        else:
            edge_weight = torch.ones(n_edges) * edge_weight_init

        assert torch.all(edge_weight >= -1.0)
        assert torch.all(edge_weight <= 3.0)
        # Should not all be 0.5 (the fixed init value)
        assert not torch.all(edge_weight == 0.5)
