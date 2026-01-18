
import pytest
import numpy as np
from tensorguard.core.pipeline import RandomSparsifier

class TestRandomSparsification:
    """Test suite for Random Sparsification (Rand-K)."""
    
    def test_sparsity_ratio(self):
        """Verify output sparsity matches requested ratio."""
        sparsifier = RandomSparsifier(sparsity_ratio=0.1)  # Keep 10%
        
        # Create 1000 element array
        grad = np.random.randn(1000)
        grads = {"layer1": grad}
        
        sparse = sparsifier.sparsify(grads)
        sparse_grad = sparse["layer1"]
        
        # Check count of non-zero elements
        non_zeros = np.count_nonzero(sparse_grad)
        
        # Should be exactly 100 (10% of 1000)
        assert non_zeros == 100
        
    def test_zero_ratio_error(self):
        """Verify ValueError on invalid 0.0 ratio."""
        with pytest.raises(ValueError):
            RandomSparsifier(sparsity_ratio=0.0)
            
    def test_over_one_ratio_error(self):
        """Verify ValueError on invalid >1.0 ratio."""
        with pytest.raises(ValueError):
            RandomSparsifier(sparsity_ratio=1.1)

    def test_randomness(self):
        """Verify different calls produce different indices."""
        sparsifier = RandomSparsifier(sparsity_ratio=0.5)
        
        grad = np.arange(100) # Distinct values
        grads = {"layer1": grad}
        
        sparse1 = sparsifier.sparsify(grads)["layer1"]
        sparse2 = sparsifier.sparsify(grads)["layer1"]
        
        # Should preserve values
        assert np.all(np.isin(sparse1[sparse1 != 0], grad))
        
        # Indices should likely differ (probabilistic check)
        # Check if masks coincide exactly
        mask1 = sparse1 != 0
        mask2 = sparse2 != 0
        
        # Extremely unlikely to match exactly for size 100
        assert not np.array_equal(mask1, mask2)
    
    def test_empty_gradient(self):
        """Verify handling of empty gradients."""
        sparsifier = RandomSparsifier(sparsity_ratio=0.5)
        grads = {"layer1": np.array([])}
        sparse = sparsifier.sparsify(grads)
        assert sparse["layer1"].size == 0
