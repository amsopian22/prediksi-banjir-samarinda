
import random
import logging
from datetime import datetime, timedelta

# Setup Logging
logger = logging.getLogger(__name__)

class FloodValidator:
    """
    Simulates checking Sentinel-1 SAR imagery for flood validation.
    In a real implementation, this would connect to Sentinel Hub / Copernicus API.
    """
    
    def __init__(self):
        self.last_check = None
    
    def check_satellite_availability(self, lat, lon):
        """
        Checks real Sentinel-1 availability using Google Earth Engine.
        """
        try:
            import ee
            from google.oauth2.service_account import Credentials
            import json
            import os
            
            # 1. Load Credentials
            key_path = "new-key-sentinel/banjir-samarinda-sentinel-4bd9890a5f7a.json"
            if not os.path.exists(key_path):
                logger.warning("Sentinel Key not found. Falling back to Mock.")
                return False, 0
                
            credentials = Credentials.from_service_account_file(key_path)
            scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/earthengine'])
            
            # 2. Initialize Earth Engine with retry
            try:
                ee.Initialize(credentials=scoped_credentials)
            except Exception:
                ee.Initialize(credentials=scoped_credentials) # Simple retry
                
            # 3. Query Sentinel-1 Collection
            point = ee.Geometry.Point([lon, lat])
            
            # Look for recent images (last 24 hours)
            now = datetime.utcnow()
            start_date = (now - timedelta(hours=24)).strftime('%Y-%m-%d')
            end_date = now.strftime('%Y-%m-%d')
            
            collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                          .filterBounds(point)
                          .filterDate(start_date, end_date)
                          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                          .filter(ee.Filter.eq('instrumentMode', 'IW')))
                          
            count = collection.size().getInfo()
            
            if count > 0:
                # Calculate time difference of the latest image
                latest_image = collection.first()
                timestamp = latest_image.get('system:time_start').getInfo() # ms
                img_time = datetime.fromtimestamp(timestamp / 1000.0)
                diff_hours = (datetime.utcnow() - img_time).total_seconds() / 3600
                return True, diff_hours
            else:
                return False, 0
                
        except Exception as e:
            logger.error(f"GEE Error: {e}")
            # Fallback to Mock if GEE fails (e.g. auth issue) to prevent crash
            return False, 0

    def validate_flood_signature(self, lat, lon):
        """
        Detects flood water using thresholding on S1 Image (Real Analysis).
        """
        try:
           import ee
           # Only works if initialized
           point = ee.Geometry.Point([lon, lat])
           collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                          .filterBounds(point)
                          .sort('system:time_start', False) # Newest first
                          .limit(1))
                          
           image = collection.first()
           
           # Get VH value at point (Water usually < -20 dB)
           # Reducing region to get pixel value
           info = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10).getInfo()
           vh_val = info.get('VH', 0)
           
           # Threshold: Water is dark in Radar (VH < -18)
           is_flooded = vh_val < -18.0 
           
           return {
                "detected": is_flooded,
                "confidence": 0.95,
                "image_url": None, # URL generation requires extra steps
                "satellite": "Sentinel-1 (Real-time)"
           }
        except:
             return {"detected": False, "confidence": 0}
    def check_rain_radar(self):
        """
        Checks real-time availability of RainViewer Radar.
        Returns: (Available: bool, timestamp: int)
        """
        import requests
        try:
            # Real API Call to check availability
            url = "https://api.rainviewer.com/public/weather-maps.json"
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                if "radar" in data and "past" in data["radar"]:
                    latest_ts = data["radar"]["past"][-1]["time"]
                    
                    # Check age
                    now_ts = datetime.utcnow().timestamp()
                    diff_min = (now_ts - latest_ts) / 60
                    
                    if diff_min < 45: # Relaxed to 45 mins
                        return True, latest_ts
        except Exception as e:
            logger.error(f"RainViewer Check Failed: {e}")
            
        return False, None

    def get_radar_intensity(self, lat, lon, ts):
        """
        Fetches the specific radar tile for the lat/lon and timestamp,
        and checks the pixel value to determine rain presence.
        """
        import math
        import requests
        from PIL import Image
        from io import BytesIO
        
        try:
            # 1. Calculate Tile Coords (Zoom Level 8 provides good local resolution ~600m/pixel)
            zoom = 8 
            lat_rad = math.radians(lat)
            n = 2.0 ** zoom
            x_val = (lon + 180.0) / 360.0 * n
            y_val = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
            
            xtile = int(x_val)
            ytile = int(y_val)
            
            # Calculate pixel offset within 256x256 tile
            x_pixel = int((x_val - xtile) * 256)
            y_pixel = int((y_val - ytile) * 256)
            
            # 2. Fetch Tile
            # URL format: https://tilecache.rainviewer.com/v2/radar/{ts}/{size}/{z}/{x}/{y}/{color}/{smooth}_{snow}.png
            # Color 2 (Universal Blue) is good for standard rain.
            url = f"https://tilecache.rainviewer.com/v2/radar/{ts}/256/{zoom}/{xtile}/{ytile}/2/1_1.png"
            
            response = requests.get(url, timeout=4)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                
                # 3. Check Pixel (RGBA)
                # Ensure coordinates are within bounds
                x_pixel = max(0, min(255, x_pixel))
                y_pixel = max(0, min(255, y_pixel))
                
                pixel = img.getpixel((x_pixel, y_pixel)) 
                
                # Check opacity (alpha channel). 0 means transparent (no rain).
                if len(pixel) == 4 and pixel[3] > 0:
                    # Also checking if it's not just noise (very light rain)
                    # For now >0 is detection
                    return True
                    
        except ImportError:
            logger.error("PIL (Pillow) library missing. Cannot process radar images.")
            return False
        except Exception as e:
            logger.error(f"Radar Tile fetch error: {e}")
            
        return False

    def get_hybrid_status(self, lat, lon):
        """
        PRIORITY 1: Sentinel-1 (SAR) - High Confidence, penetrates clouds.
        PRIORITY 2: RainViewer (Radar) - Medium Confidence, indicates heavy rain.
        """
        # 1. Check Sentinel-1 (Real)
        sat_avail, hours = self.check_satellite_availability(lat, lon)
        if sat_avail:
            det_res = self.validate_flood_signature(lat, lon)
            if det_res['detected']:
                return {
                    "source": "SATELIT",
                    "label": "TERKONFIRMASI SATELIT",
                    "detail": f"{det_res['satellite']} ({hours:.1f} jam lalu)",
                    "color_hex": "#2ecc71" # Green
                }
        
        # 2. Check Radar (Real API + Image Processing)
        radar_avail, radar_ts = self.check_rain_radar()
        if radar_avail:
            # Perform actual image analysis
            is_raining = self.get_radar_intensity(lat, lon, radar_ts)
            
            if is_raining:
                 date_str = datetime.fromtimestamp(radar_ts).strftime('%H:%M')
                 return {
                    "source": "RADAR",
                    "label": "TERKONFIRMASI RADAR",
                    "detail": f"Hujan Terdeteksi ({date_str})",
                    "color_hex": "#3498db" # Blue
                }
                
        return None
