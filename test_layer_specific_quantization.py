#!/usr/bin/env python3
"""
Tests für Layer-spezifische Quantisierung.
"""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from layer_specific_quantization import (
    LayerSensitivityAnalyzer,
    LayerSpecificQuantizer,
    ProfileGuidedOptimizer,
)


class TestLayerSensitivityAnalyzer(unittest.TestCase):
    """Tests für LayerSensitivityAnalyzer."""
    
    def setUp(self):
        """Setup für Tests."""
        self.analyzer = LayerSensitivityAnalyzer(n_layers=32)
        
    def test_init(self):
        """Test Initialisierung."""
        self.assertEqual(self.analyzer.n_layers, 32)
        self.assertEqual(len(self.analyzer.sensitivity_scores), 32)
        self.assertEqual(self.analyzer.optimal_bits, [4] * 32)
        
    def test_record_score(self):
        """Test Score aufzeichnen."""
        self.analyzer.record_score(0, 0.8)
        self.assertEqual(self.analyzer.sensitivity_scores[0], 0.8)
        
    def test_record_score_invalid_index(self):
        """Test ungültiger Index."""
        self.analyzer.record_score(100, 0.5)
        self.assertEqual(self.analyzer.sensitivity_scores[0], 0.0)
        
    def test_compute_optimal_bits(self):
        """Test optimale Bits berechnen."""
        # Setze verschiedene Scores
        for i in range(32):
            self.analyzer.record_score(i, i / 32.0)
            
        bits = self.analyzer.compute_optimal_bits(
            target_avg_bits=3.5,
            min_bits=2,
            max_bits=6,
        )
        
        self.assertEqual(len(bits), 32)
        self.assertTrue(all(2 <= b <= 6 for b in bits))
        
    def test_compute_optimal_bits_uniform(self):
        """Test uniforme Scores."""
        # Alle Scores gleich
        for i in range(32):
            self.analyzer.record_score(i, 0.5)
            
        bits = self.analyzer.compute_optimal_bits(
            target_avg_bits=4.0,
            min_bits=2,
            max_bits=6,
        )
        
        # Alle Bits sollten ähnlich sein
        avg_bits = sum(bits) / len(bits)
        self.assertAlmostEqual(avg_bits, 4.0, delta=0.5)
        
    def test_get_layer_profile(self):
        """Test Layer-Profil abrufen."""
        self.analyzer.record_score(5, 0.7)
        profile = self.analyzer.get_layer_profile(5)
        
        self.assertEqual(profile["layer_idx"], 5)
        self.assertEqual(profile["sensitivity"], 0.7)
        self.assertEqual(profile["optimal_bits"], 4)


class TestLayerSpecificQuantizer(unittest.TestCase):
    """Tests für LayerSpecificQuantizer."""
    
    def setUp(self):
        """Setup für Tests."""
        self.quantizer = LayerSpecificQuantizer(n_layers=32)
        
    def test_init(self):
        """Test Initialisierung."""
        self.assertEqual(self.quantizer.n_layers, 32)
        self.assertEqual(len(self.quantizer.layer_configs), 0)
        
    def test_configure_layer(self):
        """Test Layer konfigurieren."""
        self.quantizer.configure_layer(
            layer_idx=0,
            k_bits=6,
            v_bits=4,
            group_size=128,
        )
        
        config = self.quantizer.get_layer_config(0)
        self.assertEqual(config["k_bits"], 6)
        self.assertEqual(config["v_bits"], 4)
        self.assertEqual(config["group_size"], 128)
        
    def test_get_layer_config_default(self):
        """Test Default-Konfiguration."""
        config = self.quantizer.get_layer_config(0)
        self.assertEqual(config["k_bits"], 4)
        self.assertEqual(config["v_bits"], 2)
        self.assertEqual(config["group_size"], 64)
        
    def test_auto_configure(self):
        """Test Auto-Konfiguration."""
        # Setze Sensitivitäts-Scores
        for i in range(32):
            self.quantizer.analyzer.record_score(i, i / 32.0)
            
        configs = self.quantizer.auto_configure(
            target_avg_bits=3.5,
            min_bits=2,
            max_bits=6,
        )
        
        self.assertEqual(len(configs), 32)
        
    def test_get_summary(self):
        """Test Zusammenfassung."""
        self.quantizer.configure_layer(0, k_bits=6, v_bits=4)
        self.quantizer.configure_layer(1, k_bits=2, v_bits=2)
        
        summary = self.quantizer.get_summary()
        self.assertEqual(summary["n_layers"], 32)
        self.assertEqual(summary["configured"], 2)
        self.assertAlmostEqual(summary["avg_k_bits"], 4.0)
        self.assertAlmostEqual(summary["avg_v_bits"], 3.0)
        
    def test_get_summary_empty(self):
        """Test leere Zusammenfassung."""
        summary = self.quantizer.get_summary()
        self.assertEqual(summary["configured"], 0)
        
    def test_estimate_compression_ratio(self):
        """Test Kompressions-Ratio Schätzung."""
        self.quantizer.configure_layer(0, k_bits=4, v_bits=2)
        ratio = self.quantizer._estimate_compression_ratio()
        self.assertGreater(ratio, 1.0)


class TestProfileGuidedOptimizer(unittest.TestCase):
    """Tests für ProfileGuidedOptimizer."""
    
    def setUp(self):
        """Setup für Tests."""
        self.quantizer = LayerSpecificQuantizer(n_layers=32)
        self.optimizer = ProfileGuidedOptimizer(self.quantizer)
        
    def test_init(self):
        """Test Initialisierung."""
        self.assertEqual(len(self.optimizer.profiling_data), 0)
        
    def test_record_profiling_run(self):
        """Test Profiling-Durchlauf aufzeichnen."""
        configs = {0: {"k_bits": 4, "v_bits": 2}}
        self.optimizer.record_profiling_run(
            layer_configs=configs,
            throughput=100.0,
            quality_score=0.95,
            memory_usage_mb=500.0,
        )
        
        self.assertEqual(len(self.optimizer.profiling_data), 1)
        self.assertEqual(self.optimizer.profiling_data[0]["throughput"], 100.0)
        
    def test_optimize_no_data(self):
        """Test Optimierung ohne Daten."""
        configs = self.optimizer.optimize()
        self.assertEqual(len(configs), 0)
        
    def test_optimize_with_data(self):
        """Test Optimierung mit Daten."""
        # Zeichne mehrere Durchläufe auf
        for i in range(5):
            configs = {
                0: {"k_bits": 4 + i, "v_bits": 2 + i},
                1: {"k_bits": 4 - i, "v_bits": 2},
            }
            self.optimizer.record_profiling_run(
                layer_configs=configs,
                throughput=100.0 + i * 10,
                quality_score=0.95 - i * 0.01,
                memory_usage_mb=500.0 - i * 20,
            )
            
        # Optimiere
        optimized = self.optimizer.optimize(target_quality=0.90)
        
        # Sollte Konfiguration zurückgeben
        self.assertGreater(len(optimized), 0)


if __name__ == "__main__":
    unittest.main()
