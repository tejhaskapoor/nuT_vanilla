"""Unit tests and benchmarks for nuT models.

Usage:
    pytest test_benchmark_models.py -v -s
    
To run only benchmarks:
    pytest test_benchmark_models.py -v -s -m benchmark
"""

import pytest
import torch
import time
from typing import Dict, List
from unittest.mock import Mock

torch.set_float32_matmul_precision('high')


FEATURES = [
    "sensor_pos_x",
    "sensor_pos_y", 
    "sensor_pos_z",
    "t",
    "charge",
    "string_id",
    "is_signal",
]

IDX_DICT = {feat: idx for idx, feat in enumerate(FEATURES)}

SEQ_LENGTH = 300
BATCH_SIZES = [1, 8, 16, 32]
N_FEATURES = 5

DEFAULT_CONFIG = {
    "idx_dict": IDX_DICT,
    "emb_dims": 256,
    "seq_length": SEQ_LENGTH,
    "emb_type": "nuT",
    "n_features": N_FEATURES,
    "abs_position_encoding": True,
    "refractive_index": 1.33,
    "masks": ["Causality", "Euclidean", "STRING"],
    "mode": "concat",
    "pairwise_dims": 64,
    "num_heads": 8,
    "dropout_attn": 0.0,
    "hidden_dim": 256,
    "dropout_FFNN": 0.0,
    "no_hits_blocks": 4,
    "no_evt_blocks": 2,
}


BENCHMARK_RESULTS = []


def create_mock_data(batch_size: int, seq_length: int = SEQ_LENGTH, n_features: int = N_FEATURES):
    """Create mock data object mimicking torch_geometric Data."""
    data = Mock()
    data.x = torch.randn(batch_size, seq_length, n_features)
    data.batch = torch.zeros(batch_size * seq_length, dtype=torch.long)
    
    for i in range(batch_size):
        start_idx = i * seq_length
        end_idx = min((i + 1) * seq_length, start_idx + seq_length - 10)
        data.batch[start_idx:end_idx] = i
    
    return data


def print_benchmark_summary():
    """Print benchmark results summary."""
    if not BENCHMARK_RESULTS:
        return
        
    print("\n" + "="*100)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*100)
    print(f"{'Model':<30} {'Batch':<8} {'Time (ms)':<12} {'Memory (MB)':<15} {'Throughput (samples/s)'}")
    print("-"*100)
    
    for r in BENCHMARK_RESULTS:
        print(f"{r['model']:<30} {r['batch']:<8} {r['time_ms']:<12.2f} {r['memory_mb']:<15.2f} {r['throughput']:<20.2f}")
    
    print("="*100)


@pytest.fixture
def prometheus_config():
    return DEFAULT_CONFIG.copy()


class TestNuT_PROMETHEUS:
    """Tests for nuT_PROMETHEUS model."""
    
    @pytest.fixture
    def model(self, prometheus_config):
        from nuT.nuT_model import nuT_PROMETHEUS
        model = nuT_PROMETHEUS(**prometheus_config)
        model.eval()
        return model
    
    def test_model_creation(self, model):
        """Test that model can be created."""
        assert model is not None
        assert hasattr(model, 'hits_blocks')
        assert hasattr(model, 'evt_blocks')
    
    def test_forward_pass(self, model):
        """Test forward pass with mock data."""
        batch_size = 4
        data = create_mock_data(batch_size)
        
        with torch.no_grad():
            output = model(data)
        
        assert output.shape == (batch_size, model.nb_outputs)
    
    @pytest.mark.benchmark
    def test_benchmark(self, prometheus_config):
        """Benchmark nuT_PROMETHEUS model."""
        from nuT.nuT_model import nuT_PROMETHEUS
        model = nuT_PROMETHEUS(**prometheus_config)
        model.eval()
        
        for batch_size in BATCH_SIZES:
            data = create_mock_data(batch_size)
            
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = model(data)
            
            forward_time = time.perf_counter() - start_time
            
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                memory_mb = 0.0
            
            BENCHMARK_RESULTS.append({
                "model": "nuT_PROMETHEUS",
                "batch": batch_size,
                "time_ms": forward_time * 1000,
                "memory_mb": memory_mb,
                "throughput": batch_size / forward_time if forward_time > 0 else 0,
            })
            
            assert output.shape == (batch_size, model.nb_outputs)


class TestNuT_optimized:
    """Tests for nuT_optimized model."""
    
    @pytest.fixture
    def model(self, prometheus_config):
        from nuT.nuT_model_optimized import nuT_optimized
        config = prometheus_config.copy()
        config["use_gradient_checkpointing"] = False
        config["cache_static_masks"] = True
        model = nuT_optimized(**config)
        model.eval()
        return model
    
    def test_model_creation(self, model):
        """Test that model can be created."""
        assert model is not None
        assert hasattr(model, 'hits_blocks')
        assert hasattr(model, 'evt_blocks')
    
    def test_forward_pass(self, model):
        """Test forward pass with mock data."""
        batch_size = 4
        data = create_mock_data(batch_size)
        
        with torch.no_grad():
            output = model(data)
        
        assert output.shape == (batch_size, model.nb_outputs)
    
    @pytest.mark.benchmark
    def test_benchmark(self, prometheus_config):
        """Benchmark nuT_optimized model."""
        from nuT.nuT_model_optimized import nuT_optimized
        config = prometheus_config.copy()
        config["use_gradient_checkpointing"] = False
        config["cache_static_masks"] = True
        model = nuT_optimized(**config)
        model.eval()
        
        for batch_size in BATCH_SIZES:
            data = create_mock_data(batch_size)
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = model(data)
            
            forward_time = time.perf_counter() - start_time
            
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                memory_mb = 0.0
            
            BENCHMARK_RESULTS.append({
                "model": "nuT_optimized",
                "batch": batch_size,
                "time_ms": forward_time * 1000,
                "memory_mb": memory_mb,
                "throughput": batch_size / forward_time if forward_time > 0 else 0,
            })
            
            assert output.shape == (batch_size, model.nb_outputs)


class TestNuT_advanced_optimized:
    """Tests for nuT_advanced_optimized model."""
    
    @pytest.fixture
    def model(self, prometheus_config):
        from nuT.nuT_model_advanced import nuT_advanced_optimized
        config = prometheus_config.copy()
        config["use_flash_attention"] = True
        config["use_gradient_checkpointing"] = False
        config["use_mixed_precision"] = False
        config["enable_torch_compile"] = False
        model = nuT_advanced_optimized(**config)
        model.eval()
        return model
    
    def test_model_creation(self, model):
        """Test that model can be created."""
        assert model is not None
        assert hasattr(model, 'hits_blocks')
        assert hasattr(model, 'evt_blocks')
    
    def test_forward_pass(self, model):
        """Test forward pass with mock data."""
        batch_size = 4
        data = create_mock_data(batch_size)
        
        with torch.no_grad():
            output = model(data)
        
        assert output.shape == (batch_size, model.nb_outputs)
    
    @pytest.mark.benchmark
    def test_benchmark(self, prometheus_config):
        """Benchmark nuT_advanced_optimized model."""
        from nuT.nuT_model_advanced import nuT_advanced_optimized
        config = prometheus_config.copy()
        config["use_flash_attention"] = True
        config["use_gradient_checkpointing"] = False
        config["use_mixed_precision"] = False
        config["enable_torch_compile"] = False
        model = nuT_advanced_optimized(**config)
        model.eval()
        
        for batch_size in BATCH_SIZES:
            data = create_mock_data(batch_size)
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = model(data)
            
            forward_time = time.perf_counter() - start_time
            
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                memory_mb = 0.0
            
            BENCHMARK_RESULTS.append({
                "model": "nuT_advanced_optimized",
                "batch": batch_size,
                "time_ms": forward_time * 1000,
                "memory_mb": memory_mb,
                "throughput": batch_size / forward_time if forward_time > 0 else 0,
            })
            
            assert output.shape == (batch_size, model.nb_outputs)


class TestModelComparison:
    """Comparative tests across all models."""
    
    def test_all_models_produce_same_output_shape(self, prometheus_config):
        """Test that all models produce outputs of the same shape."""
        from nuT.nuT_model import nuT_PROMETHEUS
        from nuT.nuT_model_optimized import nuT_optimized
        from nuT.nuT_model_advanced import nuT_advanced_optimized
        
        batch_size = 8
        data = create_mock_data(batch_size)
        
        config_prom = prometheus_config.copy()
        model_prom = nuT_PROMETHEUS(**config_prom)
        model_prom.eval()
        
        config_opt = prometheus_config.copy()
        config_opt["use_gradient_checkpointing"] = False
        config_opt["cache_static_masks"] = True
        model_opt = nuT_optimized(**config_opt)
        model_opt.eval()
        
        config_adv = prometheus_config.copy()
        config_adv["use_flash_attention"] = True
        config_adv["use_gradient_checkpointing"] = False
        config_adv["use_mixed_precision"] = False
        config_adv["enable_torch_compile"] = False
        model_adv = nuT_advanced_optimized(**config_adv)
        model_adv.eval()
        
        with torch.no_grad():
            out_prom = model_prom(data)
            out_opt = model_opt(data)
            out_adv = model_adv(data)
        
        assert out_prom.shape == out_opt.shape == out_adv.shape
        assert out_prom.shape[0] == batch_size
    
    def test_model_parameters_count(self, prometheus_config):
        """Test parameter count across models."""
        from nuT.nuT_model import nuT_PROMETHEUS
        from nuT.nuT_model_optimized import nuT_optimized
        from nuT.nuT_model_advanced import nuT_advanced_optimized
        
        config_prom = prometheus_config.copy()
        model_prom = nuT_PROMETHEUS(**config_prom)
        
        config_opt = prometheus_config.copy()
        config_opt["use_gradient_checkpointing"] = False
        config_opt["cache_static_masks"] = True
        model_opt = nuT_optimized(**config_opt)
        
        config_adv = prometheus_config.copy()
        config_adv["use_flash_attention"] = True
        config_adv["use_gradient_checkpointing"] = False
        config_adv["use_mixed_precision"] = False
        config_adv["enable_torch_compile"] = False
        model_adv = nuT_advanced_optimized(**config_adv)
        
        params_prom = sum(p.numel() for p in model_prom.parameters())
        params_opt = sum(p.numel() for p in model_opt.parameters())
        params_adv = sum(p.numel() for p in model_adv.parameters())
        
        print(f"\nParameter counts:")
        print(f"  nuT_PROMETHEUS:            {params_prom:,}")
        print(f"  nuT_optimized:             {params_opt:,}")
        print(f"  nuT_advanced_optimized:    {params_adv:,}")
        
        assert params_prom > 0
        assert params_opt > 0
        assert params_adv > 0


def pytest_sessionfinish(session, exitstatus):
    """Print benchmark results after all tests complete."""
    print_benchmark_summary()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
