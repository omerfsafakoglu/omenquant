"""
OmenQuant Canlı Sinyal Sistemi
================================
Her gün çalıştır → hangi hisseyi AL / SAT / TUT göreceksin.

Kullanım:
    python live_signals.py

Strateji: TREND_FOLLOW (SMA20/SMA50 crossover + MACD)
Portföy Performans: +%191, Sharpe 2.52, ₺16K/ay (backtest)

Ömer Faruk Şafakoğlu - OmenQuant Trading System
Yıldız Teknik Üniversitesi, İstatistik Bölümü, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    print("❌ pip install yfinance")
    sys.exit(1)


# =============================================================================
# WATCHLIST — Backtestte kârlı çıkan hisseler
# =============================================================================

WATCHLIST = {
    # === MEVCUT (Backtestte kârlı) ===
    "TUPRS.IS":  {"name": "Tüpraş",             "sector": "Enerji",    "backtest": "+28.3%"},
    "THYAO.IS":  {"name": "Türk Hava Yolları",   "sector": "Havacılık", "backtest": "+26.1%"},
    "TOASO.IS":  {"name": "Tofaş",               "sector": "Otomotiv",  "backtest": "+25.4%"},
    "ASELS.IS":  {"name": "Aselsan",             "sector": "Savunma",   "backtest": "+17.3%"},
    "LOGO.IS":   {"name": "Logo Yazılım",        "sector": "Teknoloji", "backtest": "+14.8%"},
    "ISCTR.IS":  {"name": "İş Bankası C",        "sector": "Banka",     "backtest": "+9.8%"},
    "AKBNK.IS":  {"name": "Akbank",              "sector": "Banka",     "backtest": "+8.6%"},
    "EREGL.IS":  {"name": "Ereğli Demir Çelik",  "sector": "Metal",     "backtest": "+7.4%"},
    
    # === BIST30 EKLENENLER ===
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
# POZİSYON TAKİP SİSTEMİ
# =============================================================================

POSITIONS_FILE = "omenquant_positions.json"

class PositionTracker:
    """
    Açık pozisyonları dosyada tutar. Her gün çıkış koşullarını kontrol eder.
    
    Dosya yapısı (omenquant_positions.json):
    {
        "THYAO.IS": {
            "entry_price": 320.0,
            "entry_date": "2026-02-17",
            "shares": 100,
            "stop_loss": 295.0,
            "take_profit": 370.0,
            "trailing_stop": 305.0,
            "peak_price": 348.0,
            "atr_at_entry": 10.5
        }
    }
    """
    
    def __init__(self):
        self.positions = self._load()
    
    def _load(self) -> dict:
        """Pozisyonları dosyadan yükle"""
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save(self):
        """Pozisyonları dosyaya kaydet"""
        with open(POSITIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.positions, f, indent=2, ensure_ascii=False)
    
    def add_position(self, ticker: str, entry_price: float, shares: int,
                     stop_loss: float, take_profit: float, trailing_stop: float,
                     atr: float):
        """Yeni pozisyon ekle"""
        self.positions[ticker] = {
            'entry_price': round(entry_price, 2),
            'entry_date': datetime.now().strftime('%Y-%m-%d'),
            'shares': shares,
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'trailing_stop': round(trailing_stop, 2),
            'peak_price': round(entry_price, 2),
            'atr_at_entry': round(atr, 4),
        }
        self._save()
        print(f"\n  ✅ POZİSYON EKLENDİ: {ticker.replace('.IS','')} | ₺{entry_price:.2f} x {shares} adet")
        print(f"     SL: ₺{stop_loss:.2f} | TP: ₺{take_profit:.2f} | Trail: ₺{trailing_stop:.2f}")
    
    def remove_position(self, ticker: str, reason: str = ""):
        """Pozisyon kapat"""
        if ticker in self.positions:
            pos = self.positions.pop(ticker)
            self._save()
            print(f"\n  🔴 POZİSYON KAPATILDI: {ticker.replace('.IS','')} | Sebep: {reason}")
            return pos
        return None
    
    def check_exits(self, ticker: str, current_price: float, high: float,
                    low: float, signal_data: dict) -> dict:
        """
        Çıkış koşullarını kontrol et.
        
        Çıkış sebepleri:
        1. STOP_LOSS: Fiyat SL'nin altına düştü
        2. TAKE_PROFIT: Fiyat TP'ye ulaştı  
        3. TRAILING_STOP: Fiyat zirveden trailing kadar düştü
        4. TREND_REVERSAL: SMA20 < SMA50 death cross
        5. MOMENTUM_LOSS: Skor 2'nin altına düştü
        6. MACD_FLIP: MACD negatife + RSI 45 altı
        """
        if ticker not in self.positions:
            return {'exit': False}
        
        pos = self.positions[ticker]
        entry = pos['entry_price']
        sl = pos['stop_loss']
        tp = pos['take_profit']
        trail = pos['trailing_stop']
        peak = pos['peak_price']
        atr = pos['atr_at_entry']
        entry_date = pos['entry_date']
        
        # Gün sayısı
        days_held = (datetime.now() - datetime.strptime(entry_date, '%Y-%m-%d')).days
        
        # Zirve güncelle
        if current_price > peak:
            pos['peak_price'] = round(current_price, 2)
            peak = current_price
            # Trailing stop da güncellenir
            new_trail = round(peak - 2.0 * atr, 2)
            if new_trail > trail:
                pos['trailing_stop'] = new_trail
                trail = new_trail
            self._save()
        
        # Kâr/zarar
        pnl_pct = (current_price - entry) / entry * 100
        pnl_tl = (current_price - entry) * pos['shares']
        
        result = {
            'exit': False,
            'reason': '',
            'entry_price': entry,
            'current_price': current_price,
            'pnl_pct': round(pnl_pct, 2),
            'pnl_tl': round(pnl_tl, 2),
            'days_held': days_held,
            'peak_price': peak,
            'stop_loss': sl,
            'trailing_stop': trail,
            'take_profit': tp,
        }
        
        # === ÇIKIŞ KONTROLLERI ===
        
        # 1. STOP LOSS
        if low <= sl:
            result['exit'] = True
            result['reason'] = f"🛑 STOP LOSS — Fiyat ₺{sl:.2f} altına düştü"
            return result
        
        # 2. TAKE PROFIT
        if high >= tp:
            result['exit'] = True
            result['reason'] = f"🎯 TAKE PROFIT — Fiyat ₺{tp:.2f}'ye ulaştı"
            return result
        
        # 3. TRAILING STOP (min 3 gün sonra aktif)
        if days_held >= 3 and low <= trail:
            result['exit'] = True
            result['reason'] = f"📉 TRAILING STOP — Zirveden (₺{peak:.2f}) düşüş, trail ₺{trail:.2f}"
            return result
        
        # 4. TREND REVERSAL: SMA20 < SMA50 (min 5 gün sonra)
        if days_held >= 5 and not signal_data.get('in_uptrend', True):
            result['exit'] = True
            result['reason'] = "🔄 TREND DÖNÜŞÜ — SMA20 < SMA50 death cross"
            return result
        
        # 5. MOMENTUM KAYBI: Skor 2'nin altı (min 5 gün sonra)
        score = signal_data.get('score', 5)
        if days_held >= 5 and score <= 2:
            result['exit'] = True
            result['reason'] = f"💀 MOMENTUM KAYBI — Skor {score}/10'a düştü"
            return result
        
        # 6. MACD FLİP + RSI düşük (min 7 gün sonra)
        macd = signal_data.get('macd_hist', 0)
        rsi = signal_data.get('rsi', 50)
        if days_held >= 7 and macd < 0 and rsi < 40:
            result['exit'] = True
            result['reason'] = f"📊 MACD NEGATİF + RSI {rsi:.0f} — trend zayıflıyor"
            return result
        
        return result
    
    def get_portfolio_summary(self) -> str:
        """Portföy özeti"""
        if not self.positions:
            return "  📭 Açık pozisyon yok. AL sinyali olan hisselere giriş yapabilirsin."
        
        lines = []
        lines.append(f"  📊 {len(self.positions)} açık pozisyon:")
        for ticker, pos in self.positions.items():
            name = ticker.replace('.IS', '')
            entry = pos['entry_price']
            days = (datetime.now() - datetime.strptime(pos['entry_date'], '%Y-%m-%d')).days
            lines.append(f"     {name}: ₺{entry:.2f} x {pos['shares']} adet | {days} gün | SL:₺{pos['stop_loss']:.2f} Trail:₺{pos['trailing_stop']:.2f} TP:₺{pos['take_profit']:.2f}")
        return '\n'.join(lines)


# =============================================================================
# TEKNİK ANALİZ
# =============================================================================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Trend_follow için gerekli indikatörleri hesapla"""
    df = df.copy()
    
    # SMA
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['sma_200'] = df['Close'].rolling(200).mean()
    
    # EMA
    df['ema_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['Close'] * 100
    
    # Volume
    df['vol_sma20'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / (df['vol_sma20'] + 1)
    
    # Momentum
    df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    # Trend skoru (0-6)
    df['trend_score'] = (
        (df['Close'] > df['ema_5']).astype(int) +
        (df['Close'] > df['ema_10']).astype(int) +
        (df['Close'] > df['ema_20']).astype(int) +
        (df['Close'] > df['sma_50']).astype(int) +
        (df['ema_5'] > df['ema_20']).astype(int) +
        (df['ema_20'] > df['sma_50']).astype(int)
    )
    
    return df


def get_signal(df: pd.DataFrame) -> dict:
    """
    Canlı sinyal — SIFIR POZİSYON perspektifi.
    
    "Şu an bu hisseye girmeli miyim?" sorusuna cevap verir.
    Crossover beklemez — mevcut trend gücüne göre karar verir.
    
    AL: Trend aktif + momentum güçlü + RSI uygun
    İZLE: Trend var ama giriş riskli (RSI aşırı alım, momentum zayıf)
    BEKLE: Trend yok veya kararsız
    UZAK DUR: Düşüş trendi aktif
    """
    if len(df) < 52:
        return {'signal': 'BEKLE', 'reason': 'Yetersiz veri', 'strength': 0}
    
    df = compute_indicators(df)
    
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    five_days_ago = df.iloc[-6] if len(df) > 6 else df.iloc[0]
    
    price = today['Close']
    sma20 = today['sma_20']
    sma50 = today['sma_50']
    sma200 = today['sma_200']
    rsi = today['rsi']
    macd_h = today['macd_hist']
    macd_h_prev = yesterday['macd_hist']
    trend = int(today['trend_score'])
    atr = today['atr']
    atr_pct = today['atr_pct']
    momentum_5 = today['momentum_5']
    momentum_20 = today['momentum_20']
    vol_ratio = today['vol_ratio']
    bb_upper = today['bb_upper']
    bb_lower = today['bb_lower']
    
    # Temel durumlar
    in_uptrend = sma20 > sma50           # Kısa vadeli trend yukarı
    long_trend = price > sma200 if not pd.isna(sma200) else False  # Uzun vadeli trend
    price_above_sma50 = price > sma50
    macd_positive = macd_h > 0
    macd_improving = macd_h > macd_h_prev  # MACD ivme kazanıyor
    rsi_healthy = 35 < rsi < 65           # Ne aşırı alım ne aşırı satım
    rsi_overbought = rsi > 70
    rsi_oversold = rsi < 30
    momentum_positive = momentum_5 > 0
    volume_strong = vol_ratio > 1.0       # Ortalama üstü hacim
    
    # SL/TP seviyeler (giriş yapılırsa)
    sl_price = price - 2.5 * atr
    tp_price = price + 4.0 * atr
    trail_price = price - 2.0 * atr
    risk_reward = (tp_price - price) / (price - sl_price) if price > sl_price else 0
    
    # Pullback tespiti: fiyat SMA20'ye yaklaşmış mı (trend içi geri çekilme)
    sma20_distance = (price - sma20) / sma20 * 100 if sma20 > 0 else 0
    near_sma20 = -2 < sma20_distance < 3  # SMA20'ye yakın = iyi giriş noktası
    
    # ===== PUANLAMA SİSTEMİ (0-10) =====
    score = 0
    reasons = []
    warnings = []
    
    # 1. Trend yapısı (max 3 puan)
    if in_uptrend and long_trend:
        score += 3
        reasons.append("✅ Güçlü trend (SMA20>50, fiyat>SMA200)")
    elif in_uptrend:
        score += 2
        reasons.append("📈 Yükseliş trendi aktif (SMA20>50)")
    elif price_above_sma50:
        score += 1
        reasons.append("📊 Fiyat SMA50 üstünde ama trend zayıf")
    else:
        reasons.append("❌ Düşüş trendi (SMA20<50)")
    
    # 2. Momentum (max 2 puan)
    if momentum_5 > 0.02 and momentum_20 > 0.05:
        score += 2
        reasons.append(f"🚀 Güçlü momentum (5g:{momentum_5*100:+.1f}%, 20g:{momentum_20*100:+.1f}%)")
    elif momentum_5 > 0:
        score += 1
        reasons.append(f"📈 Pozitif momentum (5g:{momentum_5*100:+.1f}%)")
    else:
        reasons.append(f"📉 Negatif momentum (5g:{momentum_5*100:+.1f}%)")
    
    # 3. MACD (max 2 puan)
    if macd_positive and macd_improving:
        score += 2
        reasons.append("✅ MACD pozitif ve güçleniyor")
    elif macd_positive:
        score += 1
        reasons.append("📊 MACD pozitif ama ivme kaybediyor")
    elif macd_improving:
        score += 1
        reasons.append("📈 MACD negatif ama toparlanıyor")
    else:
        reasons.append("❌ MACD negatif ve zayıflıyor")
    
    # 4. RSI (max 2 puan)
    if rsi_healthy:
        score += 2
        reasons.append(f"✅ RSI sağlıklı bölgede ({rsi:.0f})")
    elif 30 < rsi < 75:
        score += 1
        if rsi_overbought:
            warnings.append(f"⚠️ RSI aşırı alım bölgesinde ({rsi:.0f}) — giriş riskli")
    else:
        if rsi_overbought:
            warnings.append(f"🔴 RSI çok yüksek ({rsi:.0f}) — düzeltme riski")
        elif rsi_oversold:
            warnings.append(f"🟡 RSI aşırı satım ({rsi:.0f}) — dip olabilir ama trend yok")
    
    # 5. Giriş zamanlaması (max 1 puan)
    if near_sma20 and in_uptrend:
        score += 1
        reasons.append("🎯 Fiyat SMA20'ye yakın — ideal pullback girişi")
    elif sma20_distance > 8:
        warnings.append(f"⚠️ Fiyat SMA20'den %{sma20_distance:.1f} uzakta — geç kalınmış olabilir")
    
    # ===== SİNYAL BELİRLE =====
    if score >= 8:
        signal = 'GÜÇLÜ AL'
        color = '🟢🟢'
    elif score >= 6 and not rsi_overbought:
        signal = 'AL'
        color = '🟢'
    elif score >= 6 and rsi_overbought:
        signal = 'İZLE'
        color = '🟡'
        warnings.append("Trend güçlü ama RSI yüksek — pullback bekle")
    elif score >= 4:
        signal = 'İZLE'
        color = '🟡'
    elif score >= 2 and in_uptrend:
        signal = 'BEKLE'
        color = '⚪'
    else:
        signal = 'UZAK DUR'
        color = '🔴'
    
    return {
        'signal': signal,
        'color': color,
        'score': score,
        'reasons': reasons,
        'warnings': warnings,
        'reason': reasons[0] if reasons else '',
        'strength': score,
        'price': round(price, 2),
        'sma20': round(sma20, 2),
        'sma50': round(sma50, 2),
        'sma200': round(sma200, 2) if not pd.isna(sma200) else 0,
        'rsi': round(rsi, 1),
        'macd_hist': round(macd_h, 4),
        'trend_score': trend,
        'atr_pct': round(atr_pct, 2),
        'momentum_5d': round(momentum_5 * 100, 2),
        'momentum_20d': round(momentum_20 * 100, 2),
        'vol_ratio': round(vol_ratio, 2),
        'sl_price': round(sl_price, 2),
        'tp_price': round(tp_price, 2),
        'trail_price': round(trail_price, 2),
        'risk_reward': round(risk_reward, 2),
        'sma20_distance': round(sma20_distance, 2),
        'bb_upper': round(bb_upper, 2),
        'bb_lower': round(bb_lower, 2),
        'in_uptrend': in_uptrend,
    }


# =============================================================================
# ANA SİSTEM
# =============================================================================

def run_daily_signals(watchlist: dict = None, lookback_days: int = 365):
    """
    Günlük sinyal taraması — tüm watchlist'i tara, sinyal üret.
    """
    if watchlist is None:
        watchlist = WATCHLIST
    
    today = datetime.now()
    start = today - timedelta(days=lookback_days)
    
    print(f"\n{'='*70}")
    print(f"  ⚡ OmenQuant Canlı Sinyal Sistemi")
    print(f"  📅 {today.strftime('%Y-%m-%d %H:%M')}")
    print(f"  📊 Strateji: TREND_FOLLOW (SMA20/50 + MACD)")
    print(f"  🎯 {len(watchlist)} hisse taranıyor...")
    print(f"{'='*70}")
    
    # Pozisyon tracker
    tracker = PositionTracker()
    print(f"\n{tracker.get_portfolio_summary()}\n")
    
    results = []
    exit_alerts = []  # Çıkış uyarıları
    
    for ticker, info in watchlist.items():
        try:
            data = yf.download(ticker, start=start.strftime('%Y-%m-%d'), 
                             end=today.strftime('%Y-%m-%d'), progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.dropna()
            
            if len(data) < 52:
                print(f"  ⚠️ {ticker.replace('.IS','')}: Yetersiz veri ({len(data)} gün)")
                continue
            
            sig = get_signal(data)
            sig['ticker'] = ticker
            sig['name'] = info['name']
            sig['sector'] = info['sector']
            sig['backtest'] = info['backtest']
            results.append(sig)
            
            # Açık pozisyon varsa çıkış kontrolü
            if ticker in tracker.positions:
                today_high = data['High'].iloc[-1]
                today_low = data['Low'].iloc[-1]
                exit_check = tracker.check_exits(ticker, sig['price'], today_high, today_low, sig)
                exit_check['ticker'] = ticker
                exit_check['name'] = info['name']
                if exit_check['exit']:
                    exit_alerts.append(exit_check)
                else:
                    # Pozisyon durumu güncelle
                    sig['position'] = exit_check
            
        except Exception as e:
            print(f"  ❌ {ticker.replace('.IS','')}: {e}")
    
    # ===== ÇIKIŞ UYARILARI (EN ÖNCELİKLİ) =====
    if exit_alerts:
        print(f"\n  🚨{'='*64}")
        print(f"  🚨  ÇIKIŞ UYARISI — POZİSYON KAPAT ({len(exit_alerts)} hisse)")
        print(f"  🚨{'='*64}")
        for ea in exit_alerts:
            name = ea['ticker'].replace('.IS', '')
            print(f"\n  {'─'*60}")
            print(f"  ⛔ {name} — {ea['name']}")
            print(f"  {ea['reason']}")
            print(f"  💰 Giriş: ₺{ea['entry_price']:.2f} → Şu an: ₺{ea['current_price']:.2f}")
            print(f"  📈 Kâr/Zarar: {ea['pnl_pct']:+.2f}% (₺{ea['pnl_tl']:+,.0f})")
            print(f"  ⏱️  Pozisyon süresi: {ea['days_held']} gün")
            print(f"  📊 Zirve: ₺{ea['peak_price']:.2f}")
    
    # ===== AÇIK POZİSYON DURUMU =====
    open_positions = {t: r for r in results for t in [r['ticker']] if t in tracker.positions and not any(e['ticker'] == t for e in exit_alerts)}
    if open_positions:
        print(f"\n  🔵{'='*64}")
        print(f"  🔵  AÇIK POZİSYONLAR — Devam Et ({len(open_positions)} hisse)")
        print(f"  🔵{'='*64}")
        for ticker, r in open_positions.items():
            pos = tracker.positions[ticker]
            name = ticker.replace('.IS', '')
            entry = pos['entry_price']
            pnl = (r['price'] - entry) / entry * 100
            days = (datetime.now() - datetime.strptime(pos['entry_date'], '%Y-%m-%d')).days
            pnl_color = '🟢' if pnl > 0 else '🔴'
            print(f"  {pnl_color} {name:<8} Giriş:₺{entry:.2f} Şuan:₺{r['price']:.2f} {pnl:+.2f}% | {days}g | SL:₺{pos['stop_loss']:.2f} Trail:₺{pos['trailing_stop']:.2f} TP:₺{pos['take_profit']:.2f}")
    
    # Sinyalleri kategorize et
    guclu_al = [r for r in results if r['signal'] == 'GÜÇLÜ AL']
    al_signals = [r for r in results if r['signal'] == 'AL']
    izle_signals = [r for r in results if r['signal'] == 'İZLE']
    bekle_signals = [r for r in results if r['signal'] == 'BEKLE']
    uzak_dur = [r for r in results if r['signal'] == 'UZAK DUR']
    
    # ===== GÜÇLÜ AL =====
    if guclu_al:
        print(f"\n  🟢🟢{'='*62}")
        print(f"  🟢🟢 GÜÇLÜ AL — Hemen Girilebilir ({len(guclu_al)} hisse)")
        print(f"  🟢🟢{'='*62}")
        for r in sorted(guclu_al, key=lambda x: -x['score']):
            _print_signal(r)
    
    # ===== AL =====
    if al_signals:
        print(f"\n  🟢{'='*64}")
        print(f"  🟢  AL — İyi Giriş Fırsatı ({len(al_signals)} hisse)")
        print(f"  🟢{'='*64}")
        for r in sorted(al_signals, key=lambda x: -x['score']):
            _print_signal(r)
    
    # ===== İZLE =====
    if izle_signals:
        print(f"\n  🟡{'='*64}")
        print(f"  🟡  İZLE — Pullback Bekle ({len(izle_signals)} hisse)")
        print(f"  🟡{'='*64}")
        for r in sorted(izle_signals, key=lambda x: -x['score']):
            _print_signal_compact(r)
    
    # ===== BEKLE =====
    if bekle_signals:
        print(f"\n  ⚪{'='*64}")
        print(f"  ⚪  BEKLE — Trend Zayıf ({len(bekle_signals)} hisse)")
        print(f"  ⚪{'='*64}")
        for r in sorted(bekle_signals, key=lambda x: -x['score']):
            _print_signal_compact(r)
    
    # ===== UZAK DUR =====
    if uzak_dur:
        print(f"\n  🔴{'='*64}")
        print(f"  🔴  UZAK DUR — Düşüş Trendi ({len(uzak_dur)} hisse)")
        print(f"  🔴{'='*64}")
        for r in sorted(uzak_dur, key=lambda x: -x['score']):
            _print_signal_compact(r)
    
    # ===== PANO =====
    print(f"\n\n{'='*70}")
    print(f"  📋 PORTFÖY PANOSU")
    print(f"{'='*70}")
    print(f"\n  {'Hisse':<8} {'Sinyal':<12} {'Skor':>4} {'Fiyat':>10} {'SMA20':>10} {'SMA50':>10} {'RSI':>6} {'MACD':>8} {'Trend':>6} {'Mom5d':>8}")
    print(f"  {'─'*88}")
    
    for r in sorted(results, key=lambda x: -x['score']):
        name = r['ticker'].replace('.IS', '')
        sig = r['signal']
        
        if sig == 'GÜÇLÜ AL': icon = '🟢🟢'
        elif sig == 'AL': icon = '🟢'
        elif sig == 'İZLE': icon = '🟡'
        elif sig == 'BEKLE': icon = '⚪'
        else: icon = '🔴'
        
        print(f"  {icon} {name:<6} {sig:<10} {r['score']:>3}/10 {r['price']:>10.2f} {r['sma20']:>10.2f} {r['sma50']:>10.2f} {r['rsi']:>5.1f} {r['macd_hist']:>+8.4f} {r['trend_score']:>4}/6 {r['momentum_5d']:>+7.2f}%")
    
    # ===== RİSK YÖNETİMİ =====
    active = guclu_al + al_signals
    if active:
        print(f"\n\n{'='*70}")
        print(f"  🛡️ RİSK YÖNETİMİ")
        print(f"{'='*70}")
        print(f"\n  {'Hisse':<8} {'Fiyat':>10} {'Stop Loss':>12} {'Trailing':>12} {'Take Profit':>12} {'ATR%':>6}")
        print(f"  {'─'*64}")
        for r in active:
            name = r['ticker'].replace('.IS', '')
            print(f"  {name:<8} {r['price']:>10.2f} {r['sl_price']:>12.2f} {r['trail_price']:>12.2f} {r['tp_price']:>12.2f} {r['atr_pct']:>5.2f}%")
        
        print(f"\n  ⚠️ Stop Loss = Giriş - 2.5×ATR | Trailing = Zirve - 2.0×ATR | TP = Giriş + 4.0×ATR")
    
    # ===== ÖZET =====
    print(f"\n\n{'='*70}")
    print(f"  📊 GÜNLÜK ÖZET — {today.strftime('%Y-%m-%d')}")
    print(f"{'='*70}")
    print(f"  🟢🟢 GÜÇLÜ AL: {len(guclu_al)} hisse")
    print(f"  🟢 AL:       {len(al_signals)} hisse")
    print(f"  🟡 İZLE:     {len(izle_signals)} hisse")
    print(f"  ⚪ BEKLE:    {len(bekle_signals)} hisse")
    print(f"  🔴 UZAK DUR: {len(uzak_dur)} hisse")
    
    # Piyasa genel durumu
    avg_trend = np.mean([r['trend_score'] for r in results]) if results else 0
    avg_rsi = np.mean([r['rsi'] for r in results]) if results else 50
    bullish = sum(1 for r in results if r['in_uptrend'])
    
    print(f"\n  📈 Piyasa Durumu:")
    print(f"     Ort. Trend Skoru: {avg_trend:.1f}/6 {'(güçlü)' if avg_trend >= 4 else '(zayıf)' if avg_trend < 2 else '(orta)'}")
    print(f"     Ort. RSI: {avg_rsi:.1f} {'(aşırı alım)' if avg_rsi > 70 else '(aşırı satım)' if avg_rsi < 30 else '(normal)'}")
    print(f"     Yükseliş Trendinde: {bullish}/{len(results)} hisse")
    
    print(f"\n  💡 Öneriler:")
    if avg_trend >= 4 and len(al_signals) > 0:
        print(f"     ✅ Piyasa güçlü, AL sinyallerini değerlendir")
    elif avg_trend < 2:
        print(f"     ⚠️ Piyasa zayıf, yeni pozisyon açma, mevcut pozisyonları koru")
    else:
        print(f"     📊 Piyasa kararsız, seçici ol, sadece en güçlü sinyallere gir")
    
    if len(active) >= 5:
        print(f"     ⚠️ {len(active)} aktif pozisyon — max 5 öneriliyor, en zayıfı kapat")
    
    print(f"\n{'='*70}")
    print(f"  ⏰ Bir sonraki tarama: Yarın piyasa kapanışından sonra")
    print(f"  📌 Bu sinyaller yatırım tavsiyesi DEĞİLDİR")
    print(f"{'='*70}\n")
    
    # Sonuçları HTML'e de kaydet
    _save_html_dashboard(results, today)
    
    return results


def _print_signal(r):
    """Detaylı sinyal çıktısı"""
    name = r['ticker'].replace('.IS', '')
    print(f"\n  {'─'*60}")
    print(f"  📌 {name} — {r['name']} ({r['sector']}) | Skor: {r['score']}/10")
    for reason in r.get('reasons', []):
        print(f"     {reason}")
    for warning in r.get('warnings', []):
        print(f"     {warning}")
    print(f"  💰 Fiyat: ₺{r['price']:.2f} | SMA20'ye uzaklık: %{r.get('sma20_distance', 0):.1f}")
    print(f"  📊 SMA20: ₺{r['sma20']:.2f} | SMA50: ₺{r['sma50']:.2f} | SMA200: ₺{r.get('sma200', 0):.2f}")
    print(f"  📈 RSI: {r['rsi']:.1f} | MACD Hist: {r['macd_hist']:+.4f}")
    print(f"  🔥 Trend: {r['trend_score']}/6 | Momentum 5g: {r['momentum_5d']:+.2f}% | 20g: {r['momentum_20d']:+.2f}%")
    print(f"  🛡️ Stop Loss: ₺{r['sl_price']:.2f} | TP: ₺{r['tp_price']:.2f} | R:R = 1:{r.get('risk_reward', 0):.1f}")
    print(f"  📊 Backtest: {r['backtest']}")


def _print_signal_compact(r):
    """Kompakt sinyal çıktısı"""
    name = r['ticker'].replace('.IS', '')
    print(f"  {name:<8} ₺{r['price']:<10.2f} RSI:{r['rsi']:>5.1f} Trend:{r['trend_score']}/6 Mom:{r['momentum_5d']:+.2f}% | {r['reason']}")


def _save_html_dashboard(results, today):
    """Sinyalleri HTML dashboard'a kaydet"""
    
    al = [r for r in results if r['signal'] in ('GÜÇLÜ AL', 'AL')]
    izle = [r for r in results if r['signal'] == 'İZLE']
    bekle = [r for r in results if r['signal'] == 'BEKLE']
    uzak = [r for r in results if r['signal'] == 'UZAK DUR']
    
    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>OmenQuant Günlük Sinyaller — {today.strftime('%Y-%m-%d')}</title>
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0a0a1a; color:#e0e0e0; line-height:1.6; padding:20px; }}
    .container {{ max-width:1000px; margin:0 auto; }}
    .header {{ text-align:center; padding:30px; background:linear-gradient(135deg,#1a1a2e,#16213e); border-radius:16px; margin-bottom:20px; border:1px solid #333; }}
    .header h1 {{ font-size:2em; background:linear-gradient(45deg,#38ef7d,#11998e); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
    .signal-group {{ margin-bottom:20px; }}
    .signal-header {{ padding:12px 20px; border-radius:12px 12px 0 0; font-weight:bold; font-size:1.1em; }}
    .al-header {{ background:linear-gradient(135deg,#1a3a1a,#2a4a2a); color:#38ef7d; border:1px solid #38ef7d33; }}
    .sat-header {{ background:linear-gradient(135deg,#3a1a1a,#4a2a2a); color:#f45c43; border:1px solid #f45c4333; }}
    .tut-header {{ background:linear-gradient(135deg,#1a1a3a,#2a2a4a); color:#667eea; border:1px solid #667eea33; }}
    .bekle-header {{ background:#1e1e2e; color:#888; border:1px solid #333; }}
    .signal-card {{ background:#1a1a2e; padding:15px 20px; border:1px solid #333; border-top:none; }}
    .signal-card:last-child {{ border-radius:0 0 12px 12px; }}
    .signal-row {{ display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid #222; }}
    .signal-row:last-child {{ border:none; }}
    .ticker {{ font-size:1.2em; font-weight:bold; }}
    .price {{ font-size:1.1em; }}
    .detail {{ font-size:0.85em; color:#888; }}
    .badge {{ padding:2px 8px; border-radius:4px; font-size:0.8em; font-weight:bold; }}
    .badge-al {{ background:#38ef7d22; color:#38ef7d; }}
    .badge-sat {{ background:#f45c4322; color:#f45c43; }}
    .badge-tut {{ background:#667eea22; color:#667eea; }}
    table {{ width:100%; border-collapse:collapse; margin-top:15px; }}
    th {{ background:#2a2a4a; padding:10px; text-align:left; font-size:0.85em; }}
    td {{ padding:8px 10px; border-bottom:1px solid #222; font-size:0.85em; }}
    .risk-table {{ margin-top:20px; background:#1a1a2e; border-radius:12px; padding:20px; border:1px solid #333; }}
    .footer {{ text-align:center; padding:20px; color:#555; font-size:0.8em; margin-top:20px; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>⚡ OmenQuant Günlük Sinyaller</h1>
    <p>{today.strftime('%Y-%m-%d %H:%M')} | TREND_FOLLOW | {len(results)} hisse</p>
</div>
"""
    
    def render_group(signals, group_name, css_class, emoji):
        if not signals:
            return ""
        s = f'<div class="signal-group">'
        s += f'<div class="signal-header {css_class}">{emoji} {group_name} ({len(signals)} hisse)</div>'
        s += '<div class="signal-card">'
        for r in signals:
            name = r['ticker'].replace('.IS', '')
            badge_cls = 'badge-al' if r['signal'] == 'AL' else 'badge-sat' if r['signal'] == 'SAT' else 'badge-tut'
            s += f'''<div class="signal-row">
                <div>
                    <span class="ticker">{name}</span>
                    <span class="detail"> — {r['name']} ({r['sector']})</span>
                </div>
                <div style="text-align:right;">
                    <span class="price">₺{r['price']:.2f}</span>
                    <span class="badge {badge_cls}">{r['signal']}</span><br>
                    <span class="detail">RSI:{r['rsi']:.0f} | Trend:{r['trend_score']}/6 | Mom:{r['momentum_5d']:+.1f}%</span>
                </div>
            </div>'''
            if r['signal'] in ('AL', 'SAT'):
                s += f'<div class="detail" style="padding:5px 0 10px 0;">{r["reason"]}<br>'
                s += f'🛡️ SL: ₺{r["sl_price"]:.2f} | TP: ₺{r["tp_price"]:.2f} | Trail: ₺{r["trail_price"]:.2f}</div>'
        s += '</div></div>'
        return s
    
    html += render_group(al, 'AL — Giriş Fırsatı', 'al-header', '🟢')
    html += render_group(izle, 'İZLE — Pullback Bekle', 'tut-header', '🟡')
    html += render_group(bekle, 'BEKLE — Trend Zayıf', 'bekle-header', '⚪')
    html += render_group(uzak, 'UZAK DUR — Düşüş Trendi', 'sat-header', '🔴')
    
    # Özet tablo
    html += """
<div class="risk-table">
    <h3 style="color:#667eea; margin-bottom:10px;">📋 Tüm Hisseler</h3>
    <table>
        <tr><th>Hisse</th><th>Sinyal</th><th>Fiyat</th><th>SMA20</th><th>SMA50</th><th>RSI</th><th>MACD</th><th>Trend</th><th>Mom 5g</th></tr>
"""
    for r in results:
        name = r['ticker'].replace('.IS', '')
        sig = r['signal']
        sig_color = '#38ef7d' if sig == 'AL' else '#f45c43' if sig == 'SAT' else '#667eea' if sig == 'TUT' else '#888'
        html += f'<tr><td><b>{name}</b></td><td style="color:{sig_color}"><b>{sig}</b></td>'
        html += f'<td>₺{r["price"]:.2f}</td><td>₺{r["sma20"]:.2f}</td><td>₺{r["sma50"]:.2f}</td>'
        html += f'<td>{r["rsi"]:.0f}</td><td>{r["macd_hist"]:+.4f}</td><td>{r["trend_score"]}/6</td><td>{r["momentum_5d"]:+.1f}%</td></tr>'
    
    html += """</table></div>"""
    
    html += f"""
<div class="footer">
    OmenQuant Trading System | Ömer Faruk Şafakoğlu | YTÜ İstatistik 2025<br>
    Bu sinyaller yatırım tavsiyesi değildir. | {today.strftime('%Y-%m-%d %H:%M')}
</div>
</div></body></html>"""
    
    filepath = f"omenquant_signals_{today.strftime('%Y%m%d')}.html"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n📄 HTML Dashboard: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'add':
        # Pozisyon ekleme: python live_signals.py add THYAO 320.50 100
        tracker = PositionTracker()
        if len(sys.argv) >= 5:
            ticker = sys.argv[2].upper()
            if not ticker.endswith('.IS'):
                ticker += '.IS'
            price = float(sys.argv[3])
            shares = int(sys.argv[4])
            
            # ATR hesapla
            data = yf.download(ticker, period='6mo', progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            
            sl = round(price - 2.5 * atr, 2)
            tp = round(price + 4.0 * atr, 2)
            trail = round(price - 2.0 * atr, 2)
            
            tracker.add_position(ticker, price, shares, sl, tp, trail, atr)
        else:
            print("Kullanım: python live_signals.py add THYAO 320.50 100")
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'remove':
        # Pozisyon silme: python live_signals.py remove THYAO
        tracker = PositionTracker()
        if len(sys.argv) >= 3:
            ticker = sys.argv[2].upper()
            if not ticker.endswith('.IS'):
                ticker += '.IS'
            tracker.remove_position(ticker, "Manuel kapatma")
        else:
            print("Kullanım: python live_signals.py remove THYAO")
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'positions':
        # Pozisyon listele
        tracker = PositionTracker()
        print(tracker.get_portfolio_summary())
    
    else:
        # Normal tarama
        run_daily_signals()
