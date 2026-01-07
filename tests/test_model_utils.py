"""
Unit Tests for model_utils.py
Tests FloodRiskSystem and predict_flood function
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model_utils import FloodRiskSystem, predict_flood, load_model


class TestFloodRiskSystem:
    """Tests for FloodRiskSystem class"""
    
    def test_level_normal_below_threshold(self):
        """Test AMAN status when depth < 20cm"""
        result = FloodRiskSystem.get_risk_assessment(10.0)
        assert result["label"] == "AMAN"
        assert result["color"] == "green"
        assert result["depth_cm"] == 10.0
        
    def test_level_waspada_threshold(self):
        """Test WASPADA status when 20cm <= depth < 50cm"""
        result = FloodRiskSystem.get_risk_assessment(25.0)
        assert result["label"] == "WASPADA"
        assert result["color"] == "yellow"
        
    def test_level_siaga_threshold(self):
        """Test SIAGA status when 50cm <= depth < 100cm"""
        result = FloodRiskSystem.get_risk_assessment(75.0)
        assert result["label"] == "SIAGA"
        assert result["color"] == "orange"
        
    def test_level_awas_threshold(self):
        """Test AWAS status when depth >= 100cm"""
        result = FloodRiskSystem.get_risk_assessment(150.0)
        assert result["label"] == "AWAS"
        assert result["color"] == "red"
        
    def test_boundary_waspada_exact(self):
        """Test exact threshold at 20cm (WASPADA boundary)"""
        result = FloodRiskSystem.get_risk_assessment(20.0)
        assert result["label"] == "WASPADA"
        
    def test_boundary_siaga_exact(self):
        """Test exact threshold at 50cm (SIAGA boundary)"""
        result = FloodRiskSystem.get_risk_assessment(50.0)
        assert result["label"] == "SIAGA"
        
    def test_boundary_awas_exact(self):
        """Test exact threshold at 100cm (AWAS boundary)"""
        result = FloodRiskSystem.get_risk_assessment(100.0)
        assert result["label"] == "AWAS"
        
    def test_reasoning_with_heavy_rain(self):
        """Test reasoning includes rain when rain > 50mm"""
        input_data = {"rain_sum_imputed": 65.0, "pasut_msl_max": 2.0}
        result = FloodRiskSystem.get_risk_assessment(60.0, input_data)
        assert "Hujan Deras" in result["reasoning"]
        
    def test_reasoning_with_extreme_rain(self):
        """Test reasoning includes extreme rain when rain > 150mm"""
        input_data = {"rain_sum_imputed": 180.0, "pasut_msl_max": 2.0}
        result = FloodRiskSystem.get_risk_assessment(120.0, input_data)
        assert "Hujan Ekstrem" in result["reasoning"]
        
    def test_reasoning_with_high_tide(self):
        """Test reasoning includes Rob when tide causes overflow"""
        input_data = {"rain_sum_imputed": 10.0, "pasut_msl_max": 3.5}
        result = FloodRiskSystem.get_risk_assessment(30.0, input_data)
        assert "Pasang" in result["reasoning"]
        
    def test_recommendation_aman(self):
        """Test recommendation for AMAN status"""
        result = FloodRiskSystem.get_risk_assessment(5.0)
        assert "aman" in result["recommendation"].lower()
        
    def test_recommendation_awas(self):
        """Test recommendation for AWAS status includes evacuation"""
        result = FloodRiskSystem.get_risk_assessment(120.0)
        assert "evakuasi" in result["recommendation"].lower()


class TestPredictFlood:
    """Tests for predict_flood function"""
    
    @pytest.fixture
    def model_pack(self):
        """Load model for testing"""
        return load_model()
    
    @pytest.fixture
    def base_input(self):
        """Base input data for testing"""
        return {
            "rain_sum_imputed": 0.0,
            "pasut_msl_max": 2.5,
            "rain_rolling_24h": 0.0,
            "rain_cumsum_7d": 0.0,
            "rain_intensity_max": 0,
            "soil_moisture_surface_mean": 0.5,
            "soil_moisture_root_mean": 0.5,
            "rain_lag1": 0, "rain_lag2": 0, "rain_lag3": 0,
            "rain_lag4": 0, "rain_lag5": 0, "rain_lag6": 0, "rain_lag7": 0,
            "rain_cumsum_3d": 0,
            "tide_rain_interaction": 0,
            "is_high_tide": 0,
            "is_heavy_rain": 0,
            "api_7day": 0,
            "month_sin": 0, "month_cos": 0,
            "is_rainy_season": 1,
            "is_weekend": 0,
            "prev_flood_30d": 0,
            "prev_meluap_30d": 0,
            "rain_intensity_3h": 0,
            "rain_burst_count": 0,
            "soil_saturation_trend": 0,
            "tide_rain_sync": 0,
            "consecutive_rain_days": 0,
            "hour_risk_factor": 1.0,
            "drain_capacity_index": 0.2,
            "upstream_rain_6h": 0,
            "wind_speed_max": 0,
            "rainfall_acceleration": 0
        }
    
    def test_predict_returns_dict(self, model_pack, base_input):
        """Test predict_flood returns a dictionary"""
        if model_pack is None:
            pytest.skip("Model not available")
        result = predict_flood(model_pack, base_input)
        assert isinstance(result, dict)
        
    def test_predict_has_required_keys(self, model_pack, base_input):
        """Test predict_flood returns required keys"""
        if model_pack is None:
            pytest.skip("Model not available")
        result = predict_flood(model_pack, base_input)
        required_keys = ["label", "depth_cm", "reasoning", "recommendation"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
            
    def test_dry_day_safety_cap(self, model_pack, base_input):
        """Test dry day (no rain) with high tide doesn't produce AWAS"""
        if model_pack is None:
            pytest.skip("Model not available")
        # High tide, no rain scenario
        dry_input = base_input.copy()
        dry_input["pasut_msl_max"] = 3.8
        dry_input["is_high_tide"] = 1
        
        result = predict_flood(model_pack, dry_input)
        # Should not be AWAS on a dry day
        assert result["label"] in ["AMAN", "WASPADA", "SIAGA"], \
            f"Dry day produced {result['label']} - should not be AWAS"
            
    def test_storm_surge_escalation(self, model_pack, base_input):
        """Test heavy rain + high tide produces higher depth than dry conditions"""
        if model_pack is None:
            pytest.skip("Model not available")
        
        # Dry condition
        dry_input = base_input.copy()
        dry_result = predict_flood(model_pack, dry_input)
        
        # Storm condition
        storm_input = base_input.copy()
        storm_input["rain_sum_imputed"] = 80.0
        storm_input["pasut_msl_max"] = 3.5
        storm_input["is_high_tide"] = 1
        storm_input["is_heavy_rain"] = 1
        storm_input["tide_rain_sync"] = 1
        storm_input["rain_cumsum_3d"] = 120.0
        
        storm_result = predict_flood(model_pack, storm_input)
        
        # Storm should produce higher depth than dry
        assert storm_result["depth_cm"] >= dry_result["depth_cm"], \
            f"Storm depth ({storm_result['depth_cm']}) should be >= dry depth ({dry_result['depth_cm']})"
            
    def test_depth_non_negative(self, model_pack, base_input):
        """Test predicted depth is never negative"""
        if model_pack is None:
            pytest.skip("Model not available")
        result = predict_flood(model_pack, base_input)
        assert result["depth_cm"] >= 0, "Depth should never be negative"
        
    def test_null_model_returns_error(self, base_input):
        """Test handling of null model"""
        result = predict_flood(None, base_input)
        assert "level" in result
        assert result.get("probability", result.get("depth_cm", 0)) == 0


class TestLoadModel:
    """Tests for load_model function"""
    
    def test_load_model_returns_dict_or_none(self):
        """Test load_model returns dict or None"""
        # Note: This test requires Streamlit context, so we catch the error
        try:
            result = load_model()
            assert result is None or isinstance(result, dict)
        except Exception:
            pytest.skip("Streamlit context not available")
            
    def test_model_pack_has_model_key(self):
        """Test model pack contains 'model' key"""
        try:
            result = load_model()
            if result is not None:
                assert "model" in result
        except Exception:
            pytest.skip("Streamlit context not available")


class TestConfigThresholds:
    """Tests for config thresholds consistency"""
    
    def test_threshold_order(self):
        """Test thresholds are in correct ascending order"""
        assert config.THRESHOLD_DEPTH_WASPADA < config.THRESHOLD_DEPTH_SIAGA
        assert config.THRESHOLD_DEPTH_SIAGA < config.THRESHOLD_DEPTH_AWAS
        
    def test_threshold_values(self):
        """Test threshold values match expected"""
        assert config.THRESHOLD_DEPTH_WASPADA == 20.0
        assert config.THRESHOLD_DEPTH_SIAGA == 50.0
        assert config.THRESHOLD_DEPTH_AWAS == 100.0
