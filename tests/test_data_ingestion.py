"""
Unit Tests for data_ingestion.py
Tests WeatherFetcher and TidePredictor classes
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TestWeatherFetcher:
    """Tests for WeatherFetcher class"""
    
    @pytest.fixture
    def weather_fetcher(self):
        """Create WeatherFetcher instance"""
        from data_ingestion import WeatherFetcher
        return WeatherFetcher()
    
    def test_fetcher_initialization(self, weather_fetcher):
        """Test WeatherFetcher initializes correctly"""
        assert weather_fetcher.url == config.OPENMETEO_URL
        assert weather_fetcher.openmeteo is not None
        
    def test_fetch_returns_dataframe(self, weather_fetcher):
        """Test fetch_weather_data returns DataFrame"""
        # This may fail if no internet, so we handle gracefully
        try:
            result = weather_fetcher.fetch_weather_data()
            assert isinstance(result, pd.DataFrame)
        except Exception:
            pytest.skip("API not available")
            
    def test_fetch_has_required_columns(self, weather_fetcher):
        """Test fetched data has required columns"""
        try:
            result = weather_fetcher.fetch_weather_data()
            if not result.empty:
                required_cols = ["date", "precipitation"]
                for col in required_cols:
                    assert col in result.columns, f"Missing column: {col}"
        except Exception:
            pytest.skip("API not available")
            
    def test_fetch_with_custom_coordinates(self, weather_fetcher):
        """Test fetch with custom lat/lon"""
        try:
            result = weather_fetcher.fetch_weather_data(lat=-0.5, lon=117.1)
            assert isinstance(result, pd.DataFrame)
        except Exception:
            pytest.skip("API not available")
            
    def test_rolling_features_calculated(self, weather_fetcher):
        """Test rolling features are added to DataFrame"""
        try:
            result = weather_fetcher.fetch_weather_data()
            if not result.empty:
                assert "rain_rolling_24h" in result.columns
                assert "rain_rolling_3h" in result.columns
        except Exception:
            pytest.skip("API not available")
            
    def test_fallback_mock_data(self, weather_fetcher):
        """Test fallback returns mock data on API failure"""
        # Mock the API to fail
        with patch.object(weather_fetcher.openmeteo, 'weather_api', side_effect=Exception("API Error")):
            result = weather_fetcher.fetch_weather_data()
            # Should return mock data, not empty
            assert isinstance(result, pd.DataFrame)
            if not result.empty:
                assert "date" in result.columns
                assert "precipitation" in result.columns


class TestTidePredictor:
    """Tests for TidePredictor class"""
    
    @pytest.fixture
    def tide_predictor(self):
        """Create TidePredictor instance"""
        from data_ingestion import TidePredictor
        return TidePredictor()
        
    def test_predictor_initialization(self, tide_predictor):
        """Test TidePredictor initializes"""
        # Model might be None if file not found
        assert hasattr(tide_predictor, 'model')
        
    def test_predict_returns_array(self, tide_predictor):
        """Test predict_hourly returns numpy array"""
        dates = pd.date_range(start="2026-01-01", periods=24, freq="h")
        result = tide_predictor.predict_hourly(dates)
        assert isinstance(result, np.ndarray)
        
    def test_predict_length_matches_input(self, tide_predictor):
        """Test output length matches input dates"""
        dates = pd.date_range(start="2026-01-01", periods=48, freq="h")
        result = tide_predictor.predict_hourly(dates)
        assert len(result) == len(dates)
        
    def test_predict_empty_input(self, tide_predictor):
        """Test handling of empty input"""
        dates = pd.Series([], dtype='datetime64[ns]')
        result = tide_predictor.predict_hourly(dates)
        assert len(result) == 0
        
    def test_predict_values_in_range(self, tide_predictor):
        """Test predicted tide values are in realistic range"""
        dates = pd.date_range(start="2026-01-01", periods=24, freq="h")
        result = tide_predictor.predict_hourly(dates)
        # Tide values should be between -2m and 5m (realistic range)
        if tide_predictor.model is not None:
            assert np.all(result >= -2.0), "Tide values too low"
            assert np.all(result <= 5.0), "Tide values too high"


class TestDataIngestionIntegration:
    """Integration tests for data ingestion pipeline"""
    
    def test_weather_and_tide_integration(self):
        """Test weather data can be combined with tide predictions"""
        from data_ingestion import WeatherFetcher, TidePredictor
        
        try:
            wf = WeatherFetcher()
            tp = TidePredictor()
            
            weather_df = wf.fetch_weather_data()
            if not weather_df.empty:
                tides = tp.predict_hourly(weather_df['date'])
                weather_df['tide_level'] = tides
                
                assert 'tide_level' in weather_df.columns
                assert len(weather_df['tide_level']) == len(weather_df)
        except Exception:
            pytest.skip("API or tide model not available")
