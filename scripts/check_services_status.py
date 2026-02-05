
import requests
from datetime import datetime

def check_rainviewer():
    print("Checking RainViewer API...")
    try:
        url = "https://api.rainviewer.com/public/weather-maps.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            host = data.get("host")
            if "radar" in data and "past" in data["radar"] and len(data["radar"]["past"]) > 0:
                latest = data["radar"]["past"][-1]
                ts = latest['time']
                dt = datetime.fromtimestamp(ts)
                print(f"✅ RainViewer Active. Latest Radar: {dt} (Timestamp: {ts})")
                print(f"   Host: {host}")
                return True
            else:
                print("❌ RainViewer responded but no radar data found.")
        else:
            print(f"❌ RainViewer API Error: Status {response.status_code}")
    except Exception as e:
        print(f"❌ RainViewer Connection Error: {e}")
    return False

if __name__ == "__main__":
    check_rainviewer()
