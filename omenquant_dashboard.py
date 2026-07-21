"""
OmenQuant Trading Dashboard
============================
Streamlit dashboard + Telegram bildirim sistemi.

Kurulum:
    pip install streamlit yfinance plotly requests apscheduler

Çalıştırma:
    streamlit run omenquant_dashboard.py

Telegram Bot Kurulumu:
    1. @BotFather'a /newbot yaz, bot oluştur
    2. Bot token'ı al
    3. Bota mesaj at, sonra tarayıcıda aç:
       https://api.telegram.org/bot<TOKEN>/getUpdates
    4. Chat ID'yi bul
    5. Dashboard'da ayarlara gir, token ve chat_id'yi yapıştır

Ömer Faruk Şafakoğlu - OmenQuant Trading System
Yıldız Teknik Üniversitesi, İstatistik Bölümü, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import threading
import time

try:
    import yfinance as yf
except ImportError:
    st.error("pip install yfinance")
    st.stop()

try:
    import requests
    REQUESTS_AVAILABLE = True
except:
    REQUESTS_AVAILABLE = False


# =============================================================================
# CONFIG
# =============================================================================

SETTINGS_FILE = "omenquant_settings.json"
POSITIONS_FILE = "omenquant_positions.json"
ALERTS_LOG = "omenquant_alerts.json"

WATCHLIST = {
    # Backtestte kârlı
    "TUPRS.IS":  {"name": "Tüpraş",             "sector": "Enerji",    "backtest": "+28.3%"},
    "THYAO.IS":  {"name": "Türk Hava Yolları",   "sector": "Havacılık", "backtest": "+26.1%"},
    "TOASO.IS":  {"name": "Tofaş",               "sector": "Otomotiv",  "backtest": "+25.4%"},
    "ASELS.IS":  {"name": "Aselsan",             "sector": "Savunma",   "backtest": "+17.3%"},
    "LOGO.IS":   {"name": "Logo Yazılım",        "sector": "Teknoloji", "backtest": "+14.8%"},
    "ISCTR.IS":  {"name": "İş Bankası C",        "sector": "Banka",     "backtest": "+9.8%"},
    "AKBNK.IS":  {"name": "Akbank",              "sector": "Banka",     "backtest": "+8.6%"},
    "EREGL.IS":  {"name": "Ereğli Demir Çelik",  "sector": "Metal",     "backtest": "+7.4%"},
    # BIST30
    "GARAN.IS":  {"name": "Garanti BBVA",        "sector": "Banka",     "backtest": "yeni"},
    "YKBNK.IS":  {"name": "Yapı Kredi",          "sector": "Banka",     "backtest": "yeni"},
    "HALKB.IS":  {"name": "Halkbank",            "sector": "Banka",     "backtest": "yeni"},
    "VAKBN.IS":  {"name": "Vakıfbank",           "sector": "Banka",     "backtest": "yeni"},
    "KCHOL.IS":  {"name": "Koç Holding",         "sector": "Holding",   "backtest": "yeni"},
    "SAHOL.IS":  {"name": "Sabancı Holding",     "sector": "Holding",   "backtest": "yeni"},
    "SISE.IS":   {"name": "Şişecam",             "sector": "Cam",       "backtest": "yeni"},
    "TCELL.IS":  {"name": "Turkcell",            "sector": "Telekom",   "backtest": "yeni"},
    "BIMAS.IS":  {"name": "BİM",                 "sector": "Perakende", "backtest": "yeni"},
    "PGSUS.IS":  {"name": "Pegasus",             "sector": "Havacılık", "backtest": "yeni"},
    "EKGYO.IS":  {"name": "Emlak Konut GYO",     "sector": "GYO",       "backtest": "yeni"},
    "ENKAI.IS":  {"name": "Enka İnşaat",         "sector": "İnşaat",    "backtest": "yeni"},
    "FROTO.IS":  {"name": "Ford Otosan",         "sector": "Otomotiv",  "backtest": "yeni"},
    "KOZAL.IS":  {"name": "Koza Altın",          "sector": "Madencilik","backtest": "yeni"},
    "KOZAA.IS":  {"name": "Koza Anadolu Metal",  "sector": "Madencilik","backtest": "yeni"},
    "PETKM.IS":  {"name": "Petkim",              "sector": "Kimya",     "backtest": "yeni"},
    "TAVHL.IS":  {"name": "TAV Havalimanları",   "sector": "Havacılık", "backtest": "yeni"},
    "TTKOM.IS":  {"name": "Türk Telekom",        "sector": "Telekom",   "backtest": "yeni"},
    "ARCLK.IS":  {"name": "Arçelik",             "sector": "Beyaz Eşya","backtest": "yeni"},
    "MGROS.IS":  {"name": "Migros",              "sector": "Perakende", "backtest": "yeni"},
    "SASA.IS":   {"name": "SASA Polyester",      "sector": "Kimya",     "backtest": "yeni"},
    "SOKM.IS":   {"name": "Şok Marketler",       "sector": "Perakende", "backtest": "yeni"},
}


# =============================================================================
# TELEGRAM BOT
# =============================================================================

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"telegram_token": "", "telegram_chat_id": "", "auto_scan_minutes": 60}

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

def send_telegram(message: str, settings: dict = None):
    """Telegram mesajı gönder"""
    if settings is None:
        settings = load_settings()
    token = settings.get("telegram_token", "")
    chat_id = settings.get("telegram_chat_id", "")
    
    if not token or not chat_id or not REQUESTS_AVAILABLE:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        # Önce HTML dene, başarısız olursa düz text gönder
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        resp = requests.post(url, json=payload, timeout=10)
        result = resp.json()
        
        if result.get('ok'):
            return True
        
        # HTML parse hatası varsa düz text olarak tekrar dene
        payload = {"chat_id": chat_id, "text": message.replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>', '')}
        resp = requests.post(url, json=payload, timeout=10)
        return resp.json().get('ok', False)
    except:
        return False

def test_telegram(settings):
    """Telegram bağlantısını test et"""
    return send_telegram("✅ OmenQuant bağlantı testi başarılı!", settings)


# =============================================================================
# ARKA PLAN TARAYICI (30 dk'da bir)
# =============================================================================

SCAN_STATE_FILE = "omenquant_scan_state.json"

def _load_scan_state():
    if os.path.exists(SCAN_STATE_FILE):
        try:
            with open(SCAN_STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"last_scan": None, "sent_alerts": {}}

def _save_scan_state(state):
    with open(SCAN_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def _alert_key(ticker, alert_type):
    """Aynı gün aynı bildirimi tekrar göndermemek için key"""
    return f"{ticker}_{alert_type}_{datetime.now().strftime('%Y-%m-%d')}"

def run_background_scan():
    """
    Arka plan tarayıcı — 30 dk'da bir çalışır.
    
    Kontrol eder:
    1. Açık pozisyonlar → stop/hedef/trailing tetiklendi mi?
    2. Watchlist → güçlü alım fırsatı var mı? (skor ≥ 8)
    """
    settings = load_settings()
    if not settings.get('telegram_token') or not settings.get('telegram_chat_id'):
        return  # Telegram ayarlanmamış
    
    scan_state = _load_scan_state()
    sent = scan_state.get("sent_alerts", {})
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    alerts_sent = 0
    
    # ── 1) AÇIK POZİSYONLARI KONTROL ET ─────────────────────
    positions = load_positions()
    for ticker, pos in list(positions.items()):
        try:
            data = yf.download(ticker, start=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                               end=end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.dropna()
            if len(data) < 2:
                continue
            
            current = float(data['Close'].iloc[-1])
            high = float(data['High'].iloc[-1])
            low = float(data['Low'].iloc[-1])
            name = ticker.replace('.IS', '')
            entry = pos['entry_price']
            pnl_pct = (current - entry) / entry * 100
            
            # Sinyal verisini de al
            full_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(full_data.columns, pd.MultiIndex):
                full_data.columns = full_data.columns.get_level_values(0)
            full_data = full_data.dropna()
            sig = get_signal(full_data) if len(full_data) >= 52 else {}
            
            exit_info = check_exit(ticker, pos, current, high, low, sig)
            
            # STOP LOSS tetiklendi
            if exit_info['exit'] and 'STOP' in exit_info['reason']:
                key = _alert_key(ticker, "STOP")
                if key not in sent:
                    send_telegram(
                        f"🚨 <b>STOP LOSS TETİKLENDİ!</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"📊 <b>{name}</b>  |  {now_str}\n\n"
                        f"💰 Fiyat: <b>₺{current:.2f}</b>\n"
                        f"🛑 Stop: <b>₺{pos['stop_loss']:.2f}</b>\n"
                        f"📉 K/Z: <b>{pnl_pct:+.2f}%</b>\n\n"
                        f"⚠️ <b>Pozisyonu kapat!</b>\n"
                        f"<i>⚡ OmenQuant</i>",
                        settings
                    )
                    sent[key] = now_str
                    alerts_sent += 1
            
            # TAKE PROFIT tetiklendi
            elif exit_info['exit'] and 'TAKE PROFIT' in exit_info['reason']:
                key = _alert_key(ticker, "TP")
                if key not in sent:
                    send_telegram(
                        f"🎉 <b>HEDEF FİYAT ULAŞILDI!</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"📊 <b>{name}</b>  |  {now_str}\n\n"
                        f"💰 Fiyat: <b>₺{current:.2f}</b>\n"
                        f"🎯 Hedef: <b>₺{pos['take_profit']:.2f}</b>\n"
                        f"📈 K/Z: <b>{pnl_pct:+.2f}%</b>\n\n"
                        f"✅ <b>Kar al!</b>\n"
                        f"<i>⚡ OmenQuant</i>",
                        settings
                    )
                    sent[key] = now_str
                    alerts_sent += 1
            
            # TRAILING STOP veya diğer çıkış sinyalleri
            elif exit_info['exit']:
                key = _alert_key(ticker, "EXIT")
                if key not in sent:
                    send_telegram(
                        f"⚠️ <b>ÇIKIŞ SİNYALİ!</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"📊 <b>{name}</b>  |  {now_str}\n\n"
                        f"💰 Fiyat: <b>₺{current:.2f}</b>\n"
                        f"📉 K/Z: <b>{pnl_pct:+.2f}%</b>\n"
                        f"📋 Sebep: {exit_info['reason']}\n\n"
                        f"⚠️ <b>Pozisyonu değerlendir!</b>\n"
                        f"<i>⚡ OmenQuant</i>",
                        settings
                    )
                    sent[key] = now_str
                    alerts_sent += 1
            
            time.sleep(0.5)  # Yahoo rate limit
        except:
            continue
    
    # ── 2) GÜÇLÜ ALIM FIRSATI TARA ──────────────────────────
    for ticker in WATCHLIST:
        try:
            # Zaten açık pozisyon varsa atla
            if ticker in positions:
                continue
            
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.dropna()
            if len(data) < 52:
                continue
            
            sig = get_signal(data)
            name = ticker.replace('.IS', '')
            info = WATCHLIST.get(ticker, {})
            
            # Skor ≥ 8 = GÜÇLÜ AL fırsatı
            if sig['score'] >= 8 and sig['signal'] == 'GÜÇLÜ AL':
                key = _alert_key(ticker, "FIRSAT")
                if key not in sent:
                    send_telegram(
                        f"💎 <b>GÜÇLÜ ALIM FIRSATI!</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"📊 <b>{name}</b> — {info.get('name', '')}\n"
                        f"🏷️ Sektör: {info.get('sector', '')}\n"
                        f"⏰ {now_str}\n\n"
                        f"💰 Fiyat: <b>₺{sig['price']:.2f}</b>\n"
                        f"⭐ Skor: <b>{sig['score']}/10</b>\n"
                        f"📊 RSI: {sig['rsi']:.0f}  |  Trend: {sig['trend_score']}/6\n"
                        f"🚀 Mom 5g: {sig['momentum_5d']:+.1f}%\n\n"
                        f"🛡️ SL: ₺{sig['sl_price']:.2f}\n"
                        f"🎯 TP: ₺{sig['tp_price']:.2f}\n"
                        f"⚖️ R:R = 1:{sig['risk_reward']}\n\n"
                        f"<i>⚡ OmenQuant — Güçlü alım sinyali!</i>",
                        settings
                    )
                    sent[key] = now_str
                    alerts_sent += 1
            
            time.sleep(0.5)
        except:
            continue
    
    # State kaydet
    scan_state["last_scan"] = now_str
    scan_state["sent_alerts"] = sent
    _save_scan_state(scan_state)
    
    return alerts_sent

def _scanner_loop(interval_minutes=30):
    """Arka plan thread'i — belirtilen aralıkta tarama yapar"""
    while True:
        try:
            run_background_scan()
        except:
            pass
        time.sleep(interval_minutes * 60)

def start_scanner_if_needed():
    """Scanner thread'ini sadece 1 kere başlat"""
    if 'scanner_started' not in st.session_state:
        settings = load_settings()
        interval = settings.get('auto_scan_minutes', 30)
        if settings.get('telegram_token') and settings.get('telegram_chat_id'):
            t = threading.Thread(target=_scanner_loop, args=(interval,), daemon=True)
            t.start()
            st.session_state.scanner_started = True
            st.session_state.scanner_interval = interval


# =============================================================================
# POZİSYON YÖNETİMİ
# =============================================================================

def load_positions():
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_positions(positions):
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(positions, f, indent=2, ensure_ascii=False)

def add_position(ticker, entry_price, shares, atr):
    positions = load_positions()
    sl = round(entry_price - 2.5 * atr, 2)
    tp = round(entry_price + 4.0 * atr, 2)
    trail = round(entry_price - 2.0 * atr, 2)
    
    positions[ticker] = {
        'entry_price': round(entry_price, 2),
        'entry_date': datetime.now().strftime('%Y-%m-%d'),
        'shares': shares,
        'stop_loss': sl,
        'take_profit': tp,
        'trailing_stop': trail,
        'peak_price': round(entry_price, 2),
        'atr_at_entry': round(atr, 4),
    }
    save_positions(positions)
    return positions[ticker]

def remove_position(ticker):
    positions = load_positions()
    if ticker in positions:
        pos = positions.pop(ticker)
        save_positions(positions)
        return pos
    return None

def check_exit(ticker, pos, current_price, high, low, signal_data):
    """Çıkış kontrol"""
    entry = pos['entry_price']
    sl = pos['stop_loss']
    tp = pos['take_profit']
    trail = pos['trailing_stop']
    peak = pos['peak_price']
    atr = pos['atr_at_entry']
    days = (datetime.now() - datetime.strptime(pos['entry_date'], '%Y-%m-%d')).days
    
    pnl_pct = (current_price - entry) / entry * 100
    
    # Peak güncelle
    if current_price > peak:
        pos['peak_price'] = round(current_price, 2)
        peak = current_price
        new_trail = round(peak - 2.0 * atr, 2)
        if new_trail > trail:
            pos['trailing_stop'] = new_trail
            trail = new_trail
        positions = load_positions()
        positions[ticker] = pos
        save_positions(positions)
    
    result = {
        'exit': False, 'reason': '', 'pnl_pct': round(pnl_pct, 2),
        'days': days, 'peak': peak, 'sl': sl, 'trail': trail, 'tp': tp
    }
    
    if low <= sl:
        result.update({'exit': True, 'reason': f'🛑 STOP LOSS (₺{sl:.2f})'})
    elif high >= tp:
        result.update({'exit': True, 'reason': f'🎯 TAKE PROFIT (₺{tp:.2f})'})
    elif days >= 3 and low <= trail:
        result.update({'exit': True, 'reason': f'📉 TRAILING STOP (₺{trail:.2f})'})
    elif days >= 5 and not signal_data.get('in_uptrend', True):
        result.update({'exit': True, 'reason': '🔄 TREND DÖNÜŞÜ (SMA20<SMA50)'})
    elif days >= 5 and signal_data.get('score', 5) <= 2:
        result.update({'exit': True, 'reason': f'💀 MOMENTUM KAYBI (skor {signal_data.get("score", 0)}/10)'})
    elif days >= 7 and signal_data.get('macd_hist', 0) < 0 and signal_data.get('rsi', 50) < 40:
        result.update({'exit': True, 'reason': f'📊 MACD negatif + RSI {signal_data.get("rsi", 0):.0f}'})
    
    return result


# =============================================================================
# TEKNİK ANALİZ (live_signals.py'den)
# =============================================================================

def compute_indicators(df):
    df = df.copy()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['sma_200'] = df['Close'].rolling(200).mean()
    df['ema_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    
    hl = df['High'] - df['Low']
    hc = np.abs(df['High'] - df['Close'].shift())
    lc = np.abs(df['Low'] - df['Close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['Close'] * 100
    
    df['vol_sma20'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / (df['vol_sma20'] + 1)
    df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    df['trend_score'] = (
        (df['Close'] > df['ema_5']).astype(int) +
        (df['Close'] > df['ema_10']).astype(int) +
        (df['Close'] > df['ema_20']).astype(int) +
        (df['Close'] > df['sma_50']).astype(int) +
        (df['ema_5'] > df['ema_20']).astype(int) +
        (df['ema_20'] > df['sma_50']).astype(int)
    )
    return df

def get_signal(df):
    if len(df) < 52:
        return {'signal': 'BEKLE', 'score': 0, 'reason': 'Yetersiz veri'}
    
    df = compute_indicators(df)
    t = df.iloc[-1]
    y = df.iloc[-2]
    
    price, sma20, sma50 = t['Close'], t['sma_20'], t['sma_50']
    sma200 = t['sma_200']
    rsi, macd_h = t['rsi'], t['macd_hist']
    macd_prev = y['macd_hist']
    trend = int(t['trend_score'])
    atr, atr_pct = t['atr'], t['atr_pct']
    mom5, mom20 = t['momentum_5'], t['momentum_20']
    vol_ratio = t['vol_ratio']
    
    in_uptrend = sma20 > sma50
    long_trend = price > sma200 if not pd.isna(sma200) else False
    macd_improving = macd_h > macd_prev
    sma20_dist = (price - sma20) / sma20 * 100 if sma20 > 0 else 0
    
    score = 0
    reasons = []
    warnings = []
    
    if in_uptrend and long_trend: score += 3; reasons.append("✅ Güçlü trend (SMA20>50, >SMA200)")
    elif in_uptrend: score += 2; reasons.append("📈 Yükseliş trendi (SMA20>50)")
    elif price > sma50: score += 1; reasons.append("📊 Fiyat SMA50 üstünde")
    else: reasons.append("❌ Düşüş trendi")
    
    if mom5 > 0.02 and mom20 > 0.05: score += 2; reasons.append(f"🚀 Güçlü momentum ({mom5*100:+.1f}%)")
    elif mom5 > 0: score += 1; reasons.append(f"📈 Pozitif momentum ({mom5*100:+.1f}%)")
    else: reasons.append(f"📉 Negatif momentum ({mom5*100:+.1f}%)")
    
    if macd_h > 0 and macd_improving: score += 2; reasons.append("✅ MACD güçleniyor")
    elif macd_h > 0: score += 1; reasons.append("📊 MACD pozitif")
    elif macd_improving: score += 1; reasons.append("📈 MACD toparlanıyor")
    else: reasons.append("❌ MACD zayıf")
    
    rsi_overbought = rsi > 70
    if 35 < rsi < 65: score += 2; reasons.append(f"✅ RSI sağlıklı ({rsi:.0f})")
    elif 30 < rsi < 75: score += 1
    if rsi_overbought: warnings.append(f"⚠️ RSI aşırı alım ({rsi:.0f})")
    
    if -2 < sma20_dist < 3 and in_uptrend: score += 1; reasons.append("🎯 İdeal pullback girişi")
    elif sma20_dist > 8: warnings.append(f"⚠️ SMA20'den %{sma20_dist:.1f} uzakta")
    
    if score >= 8: signal = 'GÜÇLÜ AL'
    elif score >= 6 and not rsi_overbought: signal = 'AL'
    elif score >= 6: signal = 'İZLE'
    elif score >= 4: signal = 'İZLE'
    elif score >= 2 and in_uptrend: signal = 'BEKLE'
    else: signal = 'UZAK DUR'
    
    sl = round(price - 2.5 * atr, 2) if not pd.isna(atr) else 0
    tp = round(price + 4.0 * atr, 2) if not pd.isna(atr) else 0
    rr = round((tp - price) / (price - sl), 1) if sl < price and not pd.isna(atr) else 0
    
    return {
        'signal': signal, 'score': score, 'reasons': reasons, 'warnings': warnings,
        'price': round(price, 2), 'sma20': round(sma20, 2), 'sma50': round(sma50, 2),
        'sma200': round(sma200, 2) if not pd.isna(sma200) else 0,
        'rsi': round(rsi, 1), 'macd_hist': round(macd_h, 4),
        'trend_score': trend, 'atr': round(atr, 4) if not pd.isna(atr) else 0,
        'atr_pct': round(atr_pct, 2) if not pd.isna(atr_pct) else 0,
        'momentum_5d': round(mom5 * 100, 2) if not pd.isna(mom5) else 0,
        'momentum_20d': round(mom20 * 100, 2) if not pd.isna(mom20) else 0,
        'vol_ratio': round(vol_ratio, 2) if not pd.isna(vol_ratio) else 0,
        'sl_price': sl, 'tp_price': tp, 'risk_reward': rr,
        'sma20_distance': round(sma20_dist, 2),
        'in_uptrend': in_uptrend,
    }


# =============================================================================
# VERİ ÇEKME + TARAMA
# =============================================================================

@st.cache_data(ttl=300)  # 5 dk cache
def scan_all_stocks():
    """Tüm hisseleri tara"""
    results = []
    start = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    
    tickers = list(WATCHLIST.keys())
    for i, ticker in enumerate(tickers):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.dropna()
            if len(data) < 52:
                continue
            
            sig = get_signal(data)
            sig['ticker'] = ticker
            sig['name'] = WATCHLIST[ticker]['name']
            sig['sector'] = WATCHLIST[ticker]['sector']
            sig['backtest'] = WATCHLIST[ticker]['backtest']
            # data'yı cache'e koyma — çok büyük
            results.append(sig)
        except:
            pass
    
    return results

@st.cache_data(ttl=300)
def get_stock_data(ticker, days=180):
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna()


# =============================================================================
# GRAFİK
# =============================================================================

def plot_stock_chart(data, ticker, signal_data=None, position=None):
    """Hisse grafiği — candlestick + SMA + RSI + MACD"""
    df = compute_indicators(data.copy())
    df = df.tail(90)  # Son 90 gün
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03)
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Fiyat', increasing_line_color='#38ef7d', decreasing_line_color='#f45c43'
    ), row=1, col=1)
    
    # SMA
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA20',
                             line=dict(color='#ffd700', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name='SMA50',
                             line=dict(color='#667eea', width=1.5)), row=1, col=1)
    
    # Bollinger
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='BB üst',
                             line=dict(color='gray', width=0.5, dash='dot'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='BB alt',
                             line=dict(color='gray', width=0.5, dash='dot'),
                             fill='tonexty', fillcolor='rgba(100,100,100,0.05)', showlegend=False), row=1, col=1)
    
    # Pozisyon çizgileri
    if position:
        for level, color, name in [
            (position['entry_price'], '#ffd700', 'Giriş'),
            (position['stop_loss'], '#f45c43', 'Stop Loss'),
            (position['take_profit'], '#38ef7d', 'Take Profit'),
            (position['trailing_stop'], '#ff6b6b', 'Trailing'),
        ]:
            fig.add_hline(y=level, line_dash="dash", line_color=color,
                         annotation_text=name, row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                             line=dict(color='#667eea', width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    colors = ['#38ef7d' if v >= 0 else '#f45c43' for v in df['macd_hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='MACD Hist',
                         marker_color=colors), row=3, col=1)
    
    fig.update_layout(
        title=f"{ticker.replace('.IS', '')} — {WATCHLIST.get(ticker, {}).get('name', '')}",
        template='plotly_dark',
        paper_bgcolor='#0a0a1a',
        plot_bgcolor='#0a0a1a',
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", y=1.02),
    )
    fig.update_yaxes(title_text="Fiyat (₺)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig


# =============================================================================
# STREAMLIT DASHBOARD
# =============================================================================

st.set_page_config(
    page_title="OmenQuant Trading",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .stApp { background-color: #0a0a1a; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 15px; border-radius: 12px; border: 1px solid #333;
        text-align: center; margin-bottom: 10px;
    }
    .signal-al { color: #38ef7d; font-weight: bold; font-size: 1.2em; }
    .signal-izle { color: #ffd700; font-weight: bold; font-size: 1.2em; }
    .signal-bekle { color: #888; font-weight: bold; font-size: 1.2em; }
    .signal-uzak { color: #f45c43; font-weight: bold; font-size: 1.2em; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bull-market.png", width=60)
    st.title("⚡ OmenQuant")
    st.caption("BIST Trading Sistemi")
    
    page = st.radio("Sayfa", ["📊 Sinyal Tarama", "💼 Pozisyonlarım", "📈 Grafik", "⚙️ Ayarlar"])
    
    st.divider()
    
    # Son tarama
    if 'last_scan' in st.session_state:
        st.caption(f"Son tarama: {st.session_state.last_scan}")
    
    if st.button("🔄 Yenile", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Arka plan tarayıcıyı başlat
    start_scanner_if_needed()
    if st.session_state.get('scanner_started'):
        st.caption(f"🤖 Tarayıcı aktif ({st.session_state.get('scanner_interval', 30)} dk)")


# =============================================================================
# SAYFA: SİNYAL TARAMA
# =============================================================================

if page == "📊 Sinyal Tarama":
    st.header("📊 Günlük Sinyal Taraması")
    
    with st.spinner(f"🔍 {len(WATCHLIST)} hisse taranıyor..."):
        results = scan_all_stocks()
    st.session_state.last_scan = datetime.now().strftime('%H:%M')
    st.session_state.results = results
    
    # KPI
    guclu_al = [r for r in results if r['signal'] == 'GÜÇLÜ AL']
    al = [r for r in results if r['signal'] == 'AL']
    izle = [r for r in results if r['signal'] == 'İZLE']
    bekle = [r for r in results if r['signal'] == 'BEKLE']
    uzak = [r for r in results if r['signal'] == 'UZAK DUR']
    avg_score = np.mean([r['score'] for r in results]) if results else 0
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🟢 Güçlü AL", len(guclu_al))
    c2.metric("🟢 AL", len(al))
    c3.metric("🟡 İZLE", len(izle))
    c4.metric("⚪ BEKLE", len(bekle))
    c5.metric("🔴 Uzak Dur", len(uzak))
    c6.metric("📊 Ort. Skor", f"{avg_score:.1f}/10")
    
    st.divider()
    
    # Pozisyon çıkış kontrolleri
    positions = load_positions()
    if positions:
        exit_alerts = []
        for ticker, pos in positions.items():
            matching = [r for r in results if r['ticker'] == ticker]
            if matching:
                sig = matching[0]
                try:
                    data = get_stock_data(ticker, 30)
                    if len(data) > 0:
                        ec = check_exit(ticker, pos, sig['price'], data['High'].iloc[-1], data['Low'].iloc[-1], sig)
                        if ec['exit']:
                            exit_alerts.append({'ticker': ticker, **ec, 'name': WATCHLIST.get(ticker, {}).get('name', '')})
                except:
                    pass
        
        if exit_alerts:
            st.error(f"🚨 ÇIKIŞ UYARISI — {len(exit_alerts)} pozisyon kapat!")
            for ea in exit_alerts:
                name = ea['ticker'].replace('.IS', '')
                st.warning(f"⛔ **{name}** — {ea['reason']} | K/Z: {ea['pnl_pct']:+.2f}% | {ea['days']} gün")
                
                # Telegram gönder
                settings = load_settings()
                if settings.get('telegram_token'):
                    msg = f"🚨 ÇIKIŞ UYARISI\n{name}: {ea['reason']}\nK/Z: {ea['pnl_pct']:+.2f}%"
                    send_telegram(msg, settings)
            st.divider()
    
    # Sinyal tablosu
    if results:
        df_results = pd.DataFrame([{
            'Hisse': r['ticker'].replace('.IS', ''),
            'İsim': r['name'],
            'Sinyal': r['signal'],
            'Skor': f"{r['score']}/10",
            'Fiyat': f"₺{r['price']:.2f}",
            'RSI': f"{r['rsi']:.0f}",
            'MACD': f"{r['macd_hist']:+.4f}",
            'Trend': f"{r['trend_score']}/6",
            'Mom 5g': f"{r['momentum_5d']:+.1f}%",
            'SMA20 Uzk': f"%{r['sma20_distance']:.1f}",
            'SL': f"₺{r['sl_price']:.2f}",
            'TP': f"₺{r['tp_price']:.2f}",
            'R:R': f"1:{r['risk_reward']}",
        } for r in sorted(results, key=lambda x: -x['score'])])
        
        # Renklendirme
        def color_signal(val):
            if 'AL' in str(val): return 'color: #38ef7d; font-weight: bold'
            elif val == 'İZLE': return 'color: #ffd700'
            elif val == 'BEKLE': return 'color: #888'
            elif val == 'UZAK DUR': return 'color: #f45c43'
            return ''
        
        styled = df_results.style.applymap(color_signal, subset=['Sinyal'])
        st.dataframe(styled, use_container_width=True, height=600)
    
    # AL sinyalleri detay
    for group, title, emoji in [(guclu_al, "GÜÇLÜ AL", "🟢🟢"), (al, "AL", "🟢")]:
        if group:
            st.subheader(f"{emoji} {title} Detayları")
            for r in sorted(group, key=lambda x: -x['score']):
                name = r['ticker'].replace('.IS', '')
                with st.expander(f"{name} — {r['name']} | Skor: {r['score']}/10"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Neden AL?**")
                        for reason in r.get('reasons', []):
                            st.write(f"  {reason}")
                        if r.get('warnings'):
                            st.write("**Uyarılar:**")
                            for w in r['warnings']:
                                st.write(f"  {w}")
                    with col2:
                        st.metric("Stop Loss", f"₺{r['sl_price']:.2f}")
                        st.metric("Take Profit", f"₺{r['tp_price']:.2f}")
                        st.metric("Risk:Reward", f"1:{r['risk_reward']}")
                    
                    # Grafik
                    try:
                        chart_data = get_stock_data(r['ticker'], 180)
                        if len(chart_data) > 0:
                            fig = plot_stock_chart(chart_data, r['ticker'])
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass


# =============================================================================
# SAYFA: POZİSYONLARIM
# =============================================================================

elif page == "💼 Pozisyonlarım":
    st.header("💼 Pozisyon Yönetimi")
    
    positions = load_positions()
    settings = load_settings()
    
    # Yeni pozisyon ekle
    st.subheader("➕ Pozisyon Ekle")
    
    ticker_options = [t.replace('.IS', '') for t in WATCHLIST.keys()]
    
    col1, col2 = st.columns([1, 3])
    with col1:
        new_ticker = st.selectbox("Hisse", ticker_options)
    with col2:
        price_mode = st.radio("Fiyat", ["📡 Piyasa fiyatını al", "✏️ Manuel gir"], horizontal=True)
    
    # Piyasa fiyatını çek
    full_ticker = new_ticker + ".IS"
    market_price = 0.0
    try:
        mdata = get_stock_data(full_ticker, 30)
        if len(mdata) > 0:
            market_price = round(float(mdata['Close'].iloc[-1]), 2)
    except:
        pass
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if price_mode == "📡 Piyasa fiyatını al":
            new_price = st.number_input("Giriş Fiyatı (₺)", value=market_price, step=0.01, disabled=True)
            st.caption(f"Son kapanış: ₺{market_price:.2f}")
        else:
            new_price = st.number_input("Giriş Fiyatı (₺)", min_value=0.01, value=market_price if market_price > 0 else 100.0, step=0.01)
    with col2:
        new_shares = st.number_input("Adet", min_value=1, value=100, step=1)
    with col3:
        commission_rate = st.number_input("Komisyon (%)", min_value=0.0, value=settings.get('commission_rate', 0.2), step=0.01, format="%.2f")
    with col4:
        # Maliyet hesabı
        gross_cost = new_price * new_shares
        commission_tl = gross_cost * commission_rate / 100
        total_cost = gross_cost + commission_tl
        
        st.metric("Toplam Maliyet", f"₺{total_cost:,.2f}")
        st.caption(f"İşlem: ₺{gross_cost:,.2f} + Kom: ₺{commission_tl:,.2f}")
    
    if st.button("✅ Pozisyon Ekle", use_container_width=True):
        try:
            data = get_stock_data(full_ticker, 90)
            df_ind = compute_indicators(data)
            atr = df_ind['atr'].iloc[-1]
            pos = add_position(full_ticker, new_price, new_shares, atr)
            
            # Komisyon bilgisini pozisyona kaydet
            positions_all = load_positions()
            positions_all[full_ticker]['commission_rate'] = commission_rate
            positions_all[full_ticker]['commission_tl'] = round(commission_tl, 2)
            positions_all[full_ticker]['total_cost'] = round(total_cost, 2)
            save_positions(positions_all)
            
            # Komisyon oranını ayarlara kaydet
            settings['commission_rate'] = commission_rate
            save_settings(settings)
            
            # Telegram bildirim — rerun'dan ÖNCE, ayrı try bloğunda
            tg_sent = False
            tg_debug = ""
            try:
                tg_settings = load_settings()
                tg_token = tg_settings.get('telegram_token', '').strip()
                tg_chat = tg_settings.get('telegram_chat_id', '').strip()
                
                tg_debug = f"Token: {'VAR' if tg_token else 'YOK'} ({len(tg_token)} chr) | Chat: {'VAR' if tg_chat else 'YOK'} ({tg_chat})"
                
                if tg_token and tg_chat:
                    msg = (f"📌 POZİSYON AÇILDI\n"
                           f"{'━'*28}\n"
                           f"📊 {new_ticker}\n\n"
                           f"💰 Fiyat: ₺{new_price:.2f} x {new_shares} adet\n"
                           f"💵 Maliyet: ₺{total_cost:,.2f} (Kom: ₺{commission_tl:.2f})\n"
                           f"🛡 SL: ₺{pos['stop_loss']:.2f}\n"
                           f"🎯 TP: ₺{pos['take_profit']:.2f}\n\n"
                           f"⚡ OmenQuant")
                    tg_sent = send_telegram(msg, tg_settings)
                    tg_debug += f" | Gönderim: {'OK' if tg_sent else 'BAŞARISIZ'}"
            except Exception as tg_err:
                tg_debug = f"HATA: {tg_err}"
            
            if tg_sent:
                st.success(f"✅ {new_ticker} eklendi + Telegram bildirim gönderildi!")
            else:
                st.success(f"✅ {new_ticker} eklendi! SL: ₺{pos['stop_loss']:.2f} | TP: ₺{pos['take_profit']:.2f}")
                st.error(f"🔍 Telegram debug: {tg_debug}")
            
            # time.sleep(1)  # debug için rerun kapalı
            # st.rerun()
        except Exception as e:
            st.error(f"Hata: {e}")
    
    st.divider()
    
    # Mevcut pozisyonlar
    if positions:
        st.subheader(f"📊 Açık Pozisyonlar ({len(positions)})")
        
        total_pnl = 0
        total_commission = 0
        
        for ticker, pos in positions.items():
            name = ticker.replace('.IS', '')
            info = WATCHLIST.get(ticker, {})
            
            try:
                data = get_stock_data(ticker, 30)
                current_price = float(data['Close'].iloc[-1])
            except:
                current_price = pos['entry_price']
            
            entry = pos['entry_price']
            shares = pos['shares']
            comm_rate = pos.get('commission_rate', 0.2)
            entry_comm = pos.get('commission_tl', entry * shares * comm_rate / 100)
            exit_comm = current_price * shares * comm_rate / 100
            
            gross_pnl = (current_price - entry) * shares
            net_pnl = gross_pnl - entry_comm - exit_comm
            pnl_pct = (current_price - entry) / entry * 100
            net_pnl_pct = net_pnl / (entry * shares) * 100
            
            total_pnl += net_pnl
            total_commission += entry_comm + exit_comm
            days = (datetime.now() - datetime.strptime(pos['entry_date'], '%Y-%m-%d')).days
            
            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
            
            with col1:
                icon = "🟢" if net_pnl > 0 else "🔴"
                st.write(f"**{icon} {name}** — {info.get('name', '')}")
                st.caption(f"{pos['entry_date']} | {days} gün | Kom: %{comm_rate}")
            with col2:
                st.metric("Giriş → Şuan", f"₺{current_price:.2f}", f"{pnl_pct:+.2f}%")
            with col3:
                st.write(f"Brüt K/Z: **₺{gross_pnl:+,.0f}**")
                st.write(f"Komisyon: ₺{entry_comm + exit_comm:,.0f}")
                st.write(f"Net K/Z: **₺{net_pnl:+,.0f}** ({net_pnl_pct:+.2f}%)")
            with col4:
                st.write(f"SL: ₺{pos['stop_loss']:.2f}")
                st.write(f"Trail: ₺{pos['trailing_stop']:.2f}")
                st.write(f"TP: ₺{pos['take_profit']:.2f}")
            with col5:
                if st.button("❌ Kapat", key=f"close_{ticker}"):
                    remove_position(ticker)
                    tg_settings = load_settings()
                    if tg_settings.get('telegram_token') and tg_settings.get('telegram_chat_id'):
                        send_telegram(
                            f"🔴 <b>POZİSYON KAPATILDI</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"📊 <b>{name}</b>\n\n"
                            f"📈 Net K/Z: <b>{net_pnl_pct:+.2f}%</b> (₺{net_pnl:+,.0f})\n"
                            f"💸 Komisyon: ₺{entry_comm + exit_comm:,.0f}\n\n"
                            f"<i>⚡ OmenQuant</i>",
                            tg_settings
                        )
                    st.rerun()
            
            st.divider()
        
        # Portföy özet
        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Net K/Z", f"₺{total_pnl:+,.0f}")
        c2.metric("💸 Toplam Komisyon", f"₺{total_commission:,.0f}")
        c3.metric("📦 Pozisyon Sayısı", len(positions))
    else:
        st.info("📭 Açık pozisyon yok. Sinyal Tarama sayfasından AL sinyali olan hisselere giriş yap.")


# =============================================================================
# SAYFA: GRAFİK
# =============================================================================

elif page == "📈 Grafik":
    st.header("📈 Hisse Grafiği")
    
    ticker_options = [t.replace('.IS', '') for t in WATCHLIST.keys()]
    selected = st.selectbox("Hisse Seç", ticker_options)
    full_ticker = selected + ".IS"
    
    data = get_stock_data(full_ticker, 365)
    if len(data) > 0:
        sig = get_signal(data)
        
        # KPI
        c1, c2, c3, c4, c5 = st.columns(5)
        signal_color = "🟢" if "AL" in sig['signal'] else "🟡" if sig['signal'] == "İZLE" else "🔴" if sig['signal'] == "UZAK DUR" else "⚪"
        c1.metric("Sinyal", f"{signal_color} {sig['signal']}")
        c2.metric("Skor", f"{sig['score']}/10")
        c3.metric("RSI", f"{sig['rsi']:.0f}")
        c4.metric("Trend", f"{sig['trend_score']}/6")
        c5.metric("Mom 5g", f"{sig['momentum_5d']:+.1f}%")
        
        # Pozisyon varsa göster
        positions = load_positions()
        pos = positions.get(full_ticker)
        
        fig = plot_stock_chart(data, full_ticker, sig, pos)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detaylar
        with st.expander("📋 Sinyal Detayları"):
            for r in sig.get('reasons', []):
                st.write(r)
            for w in sig.get('warnings', []):
                st.warning(w)
            st.write(f"SL: ₺{sig['sl_price']:.2f} | TP: ₺{sig['tp_price']:.2f} | R:R = 1:{sig['risk_reward']}")


# =============================================================================
# SAYFA: AYARLAR
# =============================================================================

elif page == "⚙️ Ayarlar":
    st.header("⚙️ Ayarlar")
    
    settings = load_settings()
    
    st.subheader("📱 Telegram Bildirimleri")
    st.info("""
    **Telegram Bot Kurulumu:**
    1. Telegram'da @BotFather'a `/newbot` yaz, bot oluştur
    2. Verilen **Bot Token**'ı aşağıya yapıştır
    3. Botuna bir mesaj at (herhangi bir şey)
    4. Tarayıcıda aç: `https://api.telegram.org/bot<TOKEN>/getUpdates`
    5. JSON'da `"chat":{"id":123456}` — bu senin **Chat ID**'n
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        token = st.text_input("Bot Token", value=settings.get("telegram_token", ""), type="password")
    with col2:
        chat_id = st.text_input("Chat ID", value=settings.get("telegram_chat_id", ""))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Kaydet"):
            settings['telegram_token'] = token
            settings['telegram_chat_id'] = chat_id
            save_settings(settings)
            st.success("✅ Ayarlar kaydedildi!")
    with col2:
        if st.button("🧪 Test Et"):
            settings['telegram_token'] = token
            settings['telegram_chat_id'] = chat_id
            if test_telegram(settings):
                st.success("✅ Telegram mesajı gönderildi!")
            else:
                st.error("❌ Gönderim başarısız. Token ve Chat ID'yi kontrol et.")
    
    st.divider()
    
    st.subheader("⏰ Otomatik Tarama")
    scan_interval = st.number_input("Tarama Aralığı (dakika)", min_value=5, max_value=120, 
                                     value=settings.get('auto_scan_minutes', 30), step=5)
    settings['auto_scan_minutes'] = scan_interval
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Tarayıcıyı Başlat / Yeniden Başlat"):
            settings['auto_scan_minutes'] = scan_interval
            save_settings(settings)
            if 'scanner_started' in st.session_state:
                del st.session_state['scanner_started']
            start_scanner_if_needed()
            st.success(f"✅ Tarayıcı {scan_interval} dk aralıkla çalışacak")
    with col2:
        if st.button("🔍 Şimdi Tara"):
            with st.spinner("Taranıyor..."):
                count = run_background_scan()
            st.success(f"✅ Tarama tamamlandı! {count or 0} bildirim gönderildi")
    
    scan_state = _load_scan_state()
    if scan_state.get("last_scan"):
        st.caption(f"Son tarama: {scan_state['last_scan']}")
    
    st.info(f"""
    **Tarayıcı şunları kontrol eder ({scan_interval} dk'da bir):**
    - 📌 Açık pozisyonlar → Stop/Hedef/Trailing tetiklendi mi?
    - 💎 Watchlist → Güçlü alım fırsatı var mı? (Skor ≥ 8)
    - ⚠️ Çıkış sinyalleri → Trend dönüşü, momentum kaybı
    """)
    
    st.divider()
    
    st.subheader("📊 Sistem Bilgisi")
    st.write(f"Watchlist: {len(WATCHLIST)} hisse")
    st.write(f"Açık pozisyon: {len(load_positions())}")
    st.write(f"Strateji: TREND_FOLLOW (SMA20/50 + MACD)")
    st.write(f"Backtest: +%191, Sharpe 2.52")
