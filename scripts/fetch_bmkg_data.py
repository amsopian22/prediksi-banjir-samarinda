"""
BMKG Weather Data Fetcher for all Samarinda Kelurahan.
Fetches weather data from BMKG API and stores in DuckDB.
Designed for GitHub Actions scheduled runs.
"""
import os
import sys
import logging
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_ingestion import BMKGFetcher
from database_manager import get_db

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_all_kelurahan():
    """Fetch BMKG weather data for all kelurahan in Samarinda."""
    fetcher = BMKGFetcher()
    db = get_db()
    
    total = len(config.SAMARINDA_KELURAHAN)
    success_count = 0
    error_count = 0
    
    logger.info(f"🌦️ Starting BMKG data fetch for {total} kelurahan...")
    
    for kelurahan_name, info in config.SAMARINDA_KELURAHAN.items():
        adm4_code = info["code"]
        kecamatan = info["kecamatan"]
        
        try:
            logger.info(f"  Fetching: {kelurahan_name} ({adm4_code})...")
            
            # Fetch data from BMKG
            df = fetcher.fetch_weather_data(adm4_code=adm4_code)
            
            if not df.empty:
                # Store in DuckDB
                db.log_bmkg_weather(df, adm4_code, kelurahan_name, kecamatan)
                success_count += 1
                logger.info(f"    ✅ Saved {len(df)} records for {kelurahan_name}")
            else:
                logger.warning(f"    ⚠️ No data returned for {kelurahan_name}")
                error_count += 1
            
            # Rate limiting - be nice to BMKG API
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"    ❌ Error fetching {kelurahan_name}: {e}")
            error_count += 1
    
    # Summary
    logger.info(f"\n📊 BMKG Fetch Summary:")
    logger.info(f"   Total Kelurahan: {total}")
    logger.info(f"   ✅ Success: {success_count}")
    logger.info(f"   ❌ Errors: {error_count}")
    
    # Get DB stats
    stats = db.get_stats()
    logger.info(f"   📦 Total BMKG records in DB: {stats.get('bmkg_records', 0)}")
    
    return 0 if error_count < total / 2 else 1  # Allow up to 50% failures


def main():
    """Main entry point."""
    logger.info("=" * 50)
    logger.info("BMKG Weather Data Fetcher")
    logger.info("=" * 50)
    
    return fetch_all_kelurahan()


if __name__ == "__main__":
    exit(main())
