"""
Telegram Alert Integration Module
Sends real-time flood alerts to stakeholders via Telegram.
"""

import os
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime
import config

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends flood alerts via Telegram Bot.
    
    Usage:
        1. Create bot via @BotFather on Telegram
        2. Set TELEGRAM_BOT_TOKEN environment variable
        3. Add bot to group chat and get chat ID
        4. Set TELEGRAM_CHAT_IDS environment variable (comma-separated)
    """
    
    API_URL = "https://api.telegram.org/bot{token}/sendMessage"
    
    # Alert level emoji mapping
    LEVEL_EMOJI = {
        "AMAN": "✅",
        "WASPADA": "⚠️",
        "SIAGA": "🟠",
        "AWAS": "🔴"
    }
    
    def __init__(self, token: str = None, chat_ids: List[str] = None):
        """
        Initialize Telegram notifier.
        
        Args:
            token: Bot token (or uses TELEGRAM_BOT_TOKEN env var or Streamlit secrets)
            chat_ids: List of chat IDs to send to (or uses TELEGRAM_CHAT_IDS env var)
        """
        # Try multiple sources for token: param > env > streamlit secrets
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        
        if not self.token:
            try:
                import streamlit as st
                if "TELEGRAM_BOT_TOKEN" in st.secrets:
                    self.token = st.secrets["TELEGRAM_BOT_TOKEN"]
            except Exception:
                pass
        
        # Try multiple sources for chat IDs
        chat_ids_str = os.getenv("TELEGRAM_CHAT_IDS", "")
        if not chat_ids_str:
            try:
                import streamlit as st
                if "TELEGRAM_CHAT_IDS" in st.secrets:
                    chat_ids_str = st.secrets["TELEGRAM_CHAT_IDS"]
            except Exception:
                pass
                
        self.chat_ids = chat_ids or [
            cid.strip() for cid in chat_ids_str.split(",") if cid.strip()
        ]
        self.enabled = bool(self.token and self.chat_ids)
        
        if not self.enabled:
            logger.warning("Telegram notifications disabled - missing token or chat IDs")
    
    def format_alert_message(
        self, 
        location: str, 
        status: str, 
        depth_cm: float,
        reasoning: str = "",
        recommendation: str = ""
    ) -> str:
        """
        Format alert message for Telegram.
        
        Args:
            location: Location name
            status: Risk status (AMAN, WASPADA, SIAGA, AWAS)
            depth_cm: Predicted water depth in centimeters
            reasoning: Explanation of the alert
            recommendation: Recommended action
            
        Returns:
            Formatted message string
        """
        emoji = self.LEVEL_EMOJI.get(status, "ℹ️")
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M WITA")
        
        message = f"""
{emoji} *PERINGATAN BANJIR - {status}*
━━━━━━━━━━━━━━━━━━━━

📍 *Lokasi:* {location}
🌊 *Kedalaman:* {depth_cm:.0f} cm
⏰ *Waktu:* {timestamp}

📋 *Penyebab:*
{reasoning or 'Data cuaca dan pasang surut'}

🎯 *Rekomendasi:*
{recommendation or 'Pantau kondisi secara berkala'}

━━━━━━━━━━━━━━━━━━━━
_Sumber: Sistem Peringatan Dini Banjir Samarinda_
        """.strip()
        
        return message
    
    def send_alert(
        self,
        location: str,
        status: str,
        depth_cm: float,
        reasoning: str = "",
        recommendation: str = ""
    ) -> Dict[str, bool]:
        """
        Send alert to all configured chat IDs.
        
        Args:
            location: Location name
            status: Risk status
            depth_cm: Predicted water depth
            reasoning: Explanation
            recommendation: Action to take
            
        Returns:
            Dict with chat_id -> success status
        """
        if not self.enabled:
            logger.warning("Telegram notifications not enabled")
            return {}
            
        message = self.format_alert_message(
            location, status, depth_cm, reasoning, recommendation
        )
        
        results = {}
        for chat_id in self.chat_ids:
            success = self._send_message(chat_id, message)
            results[chat_id] = success
            
        return results
    
    def _send_message(self, chat_id: str, message: str) -> bool:
        """
        Send a message to a specific chat.
        
        Args:
            chat_id: Telegram chat ID
            message: Message text (supports Markdown)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            url = self.API_URL.format(token=self.token)
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get("ok"):
                logger.info(f"Alert sent to chat {chat_id}")
                return True
            else:
                logger.error(f"Telegram API error: {data}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_test_message(self) -> Dict[str, bool]:
        """Send a test message to verify configuration."""
        test_message = """
✅ *Sistem Peringatan Banjir Samarinda*

Ini adalah pesan uji coba. Jika Anda menerima pesan ini, 
notifikasi Telegram telah berhasil dikonfigurasi.

_Waktu: {time}_
        """.format(time=datetime.now().strftime("%d/%m/%Y %H:%M WITA")).strip()
        
        results = {}
        for chat_id in self.chat_ids:
            success = self._send_message(chat_id, test_message)
            results[chat_id] = success
            
        return results


# Singleton instance for use across the app
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get or create the global Telegram notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


def send_flood_alert(
    location: str,
    status: str,
    depth_cm: float,
    reasoning: str = "",
    recommendation: str = "",
    min_level: str = "SIAGA"
) -> bool:
    """
    Convenience function to send flood alert if status meets threshold.
    
    Args:
        location: Location name
        status: Current risk status
        depth_cm: Predicted water depth
        reasoning: Explanation
        recommendation: Action to take
        min_level: Minimum level to trigger alert (default: SIAGA)
        
    Returns:
        True if alert was sent (or attempted), False if below threshold
    """
    level_order = ["AMAN", "WASPADA", "SIAGA", "AWAS"]
    
    try:
        current_idx = level_order.index(status)
        min_idx = level_order.index(min_level)
    except ValueError:
        logger.error(f"Invalid status: {status} or min_level: {min_level}")
        return False
    
    if current_idx < min_idx:
        logger.debug(f"Status {status} below threshold {min_level}, not sending alert")
        return False
    
    notifier = get_notifier()
    results = notifier.send_alert(
        location, status, depth_cm, reasoning, recommendation
    )
    
    return any(results.values())


if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Telegram Notifier...")
    print(f"Token configured: {bool(os.getenv('TELEGRAM_BOT_TOKEN'))}")
    print(f"Chat IDs configured: {bool(os.getenv('TELEGRAM_CHAT_IDS'))}")
    
    notifier = TelegramNotifier()
    if notifier.enabled:
        results = notifier.send_test_message()
        print(f"Test results: {results}")
    else:
        print("Notifier not enabled - set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS")
