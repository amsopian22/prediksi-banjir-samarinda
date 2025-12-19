
import ee
from google.oauth2.service_account import Credentials
import datetime
import os

def test_sentinel_auth():
    print("Testing GEE Auth...")
    key_path = "new-key-sentinel/banjir-samarinda-sentinel-4bd9890a5f7a.json"
    
    if not os.path.exists(key_path):
        print(f"Key file not found at {key_path}")
        return

    try:
        credentials = Credentials.from_service_account_file(key_path)
        scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/earthengine'])
        ee.Initialize(credentials=scoped_credentials)
        print("GEE Initialized Successfully!")
        
        # Test Query (30 Days window to ensure we hit an image)
        lon = 117.1536
        lat = -0.5022
        point = ee.Geometry.Point([lon, lat])
        
        now = datetime.datetime.utcnow()
        start = (now - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        end = now.strftime('%Y-%m-%d')
        
        print(f"Querying Sentinel-1 from {start} to {end}...")
        collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                      .filterBounds(point)
                      .filterDate(start, end)
                      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                      .filter(ee.Filter.eq('instrumentMode', 'IW')))
                      
        count = collection.size().getInfo()
        print(f"Found {count} images in the last 30 days.")
        
        if count > 0:
            img = collection.first()
            date = img.get('system:time_start').getInfo()
            print(f"Latest Image Timestamp: {date}")
            
    except Exception as e:
        print(f"GEE Error: {e}")

if __name__ == "__main__":
    test_sentinel_auth()
