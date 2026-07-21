"""
OmenQuant v3 - Professional Backtesting Engine
================================================
Kapsamlı backtest sistemi: Strateji motoru, walk-forward optimizasyon,
Monte Carlo simülasyonu, çoklu varlık desteği, detaylı raporlama.

Akademik Referanslar:
- Eyüboğlu (2018): Enflasyon-sektör ilişkileri
- Aslan (2024): Altın-BIST korelasyonları
- Kırman (2016): Dolar-altın dinamikleri

Ömer Faruk Şafakoğlu - OmenQuant Trading System
Yıldız Teknik Üniversitesi, İstatistik Bölümü, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import json
import warnings
import copy
import math
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class Side(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0

class ExitReason(Enum):
    SIGNAL = "SIGNAL"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_STOP = "TIME_STOP"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    END_OF_DATA = "END_OF_DATA"

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradeRecord:
    """Tek bir işlem kaydı"""
    trade_id: int
    ticker: str
    side: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: float = 0.0
    entry_value: float = 0.0
    exit_value: float = 0.0
    commission_paid: float = 0.0
    slippage_cost: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0
    holding_days: int = 0
    exit_reason: str = ""
    max_favorable: float = 0.0  # MFE - Maximum Favorable Excursion
    max_adverse: float = 0.0    # MAE - Maximum Adverse Excursion
    risk_reward_actual: float = 0.0
    

@dataclass
class BacktestConfig:
    """Backtest konfigürasyonu"""
    # Sermaye
    initial_capital: float = 100_000.0
    currency: str = "TRY"
    
    # İşlem maliyetleri
    commission_rate: float = 0.002       # %0.2 komisyon (BIST standart)
    slippage_rate: float = 0.0005        # %0.05 slippage (daha gerçekçi, likit hisseler)
    
    # Pozisyon boyutlandırma
    max_position_pct: float = 0.95       # Portföyün max %95'i tek pozisyona (swing trade)
    risk_per_trade_pct: float = 0.02     # İşlem başına max %2 risk
    max_open_positions: int = 1          # Swing trade: tek pozisyon
    
    # Risk yönetimi - ATR-bazlı dinamik stop'lar
    stop_loss_pct: float = 0.07          # %7 stop loss
    take_profit_pct: float = 0.00        # 0 = take profit yok, trailing yönetsin
    trailing_stop_pct: float = 0.05      # %5 trailing stop
    use_trailing_stop: bool = True
    use_atr_stops: bool = True           # ATR bazlı dinamik stop/trail
    atr_stop_multiplier: float = 2.5     # Stop = entry - 2.5*ATR
    atr_trail_multiplier: float = 2.0    # Trail = peak - 2.0*ATR
    time_stop_days: int = 0              # 0 = time stop kapalı
    use_time_stop: bool = False
    max_portfolio_drawdown: float = 0.25 # %25 portföy drawdown limiti
    
    # Risksiz faiz oranı (Türkiye ~%45-50 yıllık)
    risk_free_rate: float = 0.45
    
    # Re-entry: SAT sinyalinden sonra tekrar giriş
    allow_reentry: bool = True
    reentry_cooldown_days: int = 1       # SAT'tan sonra 1 gün bekle (hızlı re-entry)
    
    # Benchmark
    benchmark_ticker: str = "XU100.IS"   # BIST100
    
    # Walk-forward
    wf_train_days: int = 252             # 1 yıl eğitim
    wf_test_days: int = 63              # 3 ay test
    wf_step_days: int = 63              # 3 ay adım
    
    # Monte Carlo
    mc_simulations: int = 1000
    mc_confidence: float = 0.95


# =============================================================================
# TEKNİK ANALİZ - SİNYAL MOTORU
# =============================================================================

class TechnicalEngine:
    """
    Teknik analiz sinyal üreteci.
    RSI 30/70, MA crossover, MACD, Bollinger Bands kullanır.
    OmenQuant'ın mevcut sinyal mantığıyla uyumlu.
    """
    
    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Tüm teknik indikatörleri hesapla"""
        df = df.copy()
        
        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        df['rsi'] = TechnicalEngine._calc_rsi(df['Close'], period=14)
        
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
        df['bb_mid'] = bb_mid
        df['bb_pct'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['Close']
        
        # Volume
        df['vol_sma20'] = df['Volume'].rolling(20).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_sma20']
        
        # Momentum
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Volatilite
        df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Stochastic RSI
        rsi = df['rsi']
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        df['stoch_rsi'] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
        
        # ===== VOLATİLİTE REJİMİ =====
        # ATR'nin 60 günlük ortalamasına göre rejim belirle
        df['atr_sma60'] = df['atr'].rolling(60).mean()
        df['atr_ratio'] = df['atr'] / df['atr_sma60']
        # Rejim: LOW (<0.8), NORMAL (0.8-1.3), HIGH (>1.3)
        df['vol_regime'] = 'NORMAL'
        df.loc[df['atr_ratio'] < 0.8, 'vol_regime'] = 'LOW'
        df.loc[df['atr_ratio'] > 1.3, 'vol_regime'] = 'HIGH'
        
        # ===== TREND GÜCÜ =====
        # ADX benzeri trend güç ölçümü (basitleştirilmiş)
        df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / df['Close'] * 100
        
        # Fiyatın MA'larla ilişkisi (çoklu timeframe trend skoru)
        df['ma_alignment'] = (
            (df['Close'] > df['ema_5']).astype(int) +
            (df['Close'] > df['ema_10']).astype(int) +
            (df['Close'] > df['ema_20']).astype(int) +
            (df['Close'] > df['sma_50']).astype(int) +
            (df['ema_5'] > df['ema_20']).astype(int) +
            (df['ema_20'] > df['sma_50']).astype(int)
        )  # 0-6 arası skor, 6=full bullish alignment
        
        return df
    
    @staticmethod
    def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def generate_signals(df: pd.DataFrame, strategy: str = "combined") -> pd.Series:
        """
        Strateji bazlı sinyal üret.
        
        Stratejiler:
            - 'rsi': RSI 30/70 mean reversion
            - 'ma_crossover': MA5/MA20 crossover
            - 'macd': MACD crossover
            - 'bollinger': Bollinger Band breakout
            - 'combined': Tüm sinyallerin ağırlıklı birleşimi (OmenQuant default)
            - 'momentum': Trend following momentum
        """
        df = TechnicalEngine.compute_indicators(df)
        
        if strategy == "rsi":
            return TechnicalEngine._strategy_rsi(df)
        elif strategy == "ma_crossover":
            return TechnicalEngine._strategy_ma_crossover(df)
        elif strategy == "macd":
            return TechnicalEngine._strategy_macd(df)
        elif strategy == "bollinger":
            return TechnicalEngine._strategy_bollinger(df)
        elif strategy == "momentum":
            return TechnicalEngine._strategy_momentum(df)
        elif strategy == "trend_follow":
            return TechnicalEngine._strategy_trend_follow(df)
        elif strategy == "adaptive":
            return TechnicalEngine._strategy_adaptive(df)
        elif strategy == "vol_filtered":
            return TechnicalEngine._strategy_vol_filtered(df)
        elif strategy == "ml_enhanced":
            return TechnicalEngine._strategy_ml_enhanced(df)
        elif strategy == "combined":
            return TechnicalEngine._strategy_combined(df)
        else:
            raise ValueError(f"Bilinmeyen strateji: {strategy}")
    
    @staticmethod
    def _strategy_rsi(df: pd.DataFrame) -> pd.Series:
        """RSI Mean Reversion: RSI<30 AL, RSI>70 SAT"""
        signals = pd.Series(0, index=df.index)
        signals[df['rsi'] < 30] = 1
        signals[df['rsi'] > 70] = -1
        return signals
    
    @staticmethod
    def _strategy_ma_crossover(df: pd.DataFrame) -> pd.Series:
        """MA5/MA20 Golden/Death Cross"""
        signals = pd.Series(0, index=df.index)
        # Golden cross: kısa MA uzun MA'yı yukarı kesiyor
        golden = (df['sma_5'] > df['sma_20']) & (df['sma_5'].shift(1) <= df['sma_20'].shift(1))
        death = (df['sma_5'] < df['sma_20']) & (df['sma_5'].shift(1) >= df['sma_20'].shift(1))
        signals[golden] = 1
        signals[death] = -1
        return signals
    
    @staticmethod
    def _strategy_macd(df: pd.DataFrame) -> pd.Series:
        """MACD Crossover"""
        signals = pd.Series(0, index=df.index)
        bull = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        bear = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        signals[bull] = 1
        signals[bear] = -1
        return signals
    
    @staticmethod
    def _strategy_bollinger(df: pd.DataFrame) -> pd.Series:
        """Bollinger Band Mean Reversion"""
        signals = pd.Series(0, index=df.index)
        signals[df['Close'] < df['bb_lower']] = 1
        signals[df['Close'] > df['bb_upper']] = -1
        return signals
    
    @staticmethod
    def _strategy_momentum(df: pd.DataFrame) -> pd.Series:
        """Trend-following momentum"""
        signals = pd.Series(0, index=df.index)
        # Fiyat SMA50 üstünde + RSI 40-70 arası + MACD pozitif = AL
        bull = ((df['Close'] > df['sma_50']) & 
                (df['rsi'] > 40) & (df['rsi'] < 70) & 
                (df['macd_hist'] > 0) &
                (df['momentum_10'] > 0))
        bear = ((df['Close'] < df['sma_50']) & 
                (df['rsi'] < 60) & (df['rsi'] > 30) &
                (df['macd_hist'] < 0) &
                (df['momentum_10'] < 0))
        signals[bull] = 1
        signals[bear] = -1
        return signals
    
    @staticmethod
    def _strategy_combined(df: pd.DataFrame) -> pd.Series:
        """
        OmenQuant Combined Strategy v5 - Balanced Active Swing
        
        v3: %23 getiri, +%11.67 alpha, 3 trade (çok az)
        v4: %5 getiri, -%6 alpha, 14 trade (çok fazla, kalitesiz)
        
        v5: v3'ün sabırlı çıkışları + daha fazla giriş fırsatı
        Hedef: 5-10 trade/yıl, 15-30 gün pozisyon süresi
        
        Giriş: EMA20 bazlı (geniş filtre) + 5 farklı tetikleyici
        Çıkış: SMA50 bazlı (sabırlı) — trend devam ettiği sürece pozisyonda kal
        """
        signals = pd.Series(0, index=df.index)
        
        # EMA5 yoksa hesapla
        if 'ema_5' not in df.columns:
            df = df.copy()
            df['ema_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        if 'momentum_10' not in df.columns:
            df = df.copy()
            df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        required = ['sma_20', 'sma_50', 'ema_5', 'ema_10', 'ema_20', 'rsi', 
                     'macd_hist', 'macd', 'macd_signal', 'bb_pct']
        for col in required:
            if col not in df.columns:
                return signals
        
        state = 'FLAT'
        bars_in_position = 0
        
        for i in range(2, len(df)):
            price = df['Close'].iloc[i]
            prev_price = df['Close'].iloc[i-1]
            
            ema5 = df['ema_5'].iloc[i]
            ema10 = df['ema_10'].iloc[i]
            ema20 = df['ema_20'].iloc[i]
            sma20 = df['sma_20'].iloc[i]
            sma50 = df['sma_50'].iloc[i]
            
            ema5_prev = df['ema_5'].iloc[i-1]
            ema10_prev = df['ema_10'].iloc[i-1]
            ema20_prev = df['ema_20'].iloc[i-1]
            sma50_prev = df['sma_50'].iloc[i-1]
            
            rsi = df['rsi'].iloc[i]
            rsi_prev = df['rsi'].iloc[i-1]
            
            macd_h = df['macd_hist'].iloc[i]
            macd_h_prev = df['macd_hist'].iloc[i-1]
            macd = df['macd'].iloc[i]
            macd_sig = df['macd_signal'].iloc[i]
            macd_prev = df['macd'].iloc[i-1]
            macd_sig_prev = df['macd_signal'].iloc[i-1]
            
            bb_pct = df['bb_pct'].iloc[i]
            bb_pct_prev = df['bb_pct'].iloc[i-1]
            
            if pd.isna(sma50) or pd.isna(ema20) or pd.isna(rsi):
                continue
            
            # ===== TREND FİLTRE =====
            # Geniş filtre: fiyat EMA20 VEYA SMA50 üstünde (daha fazla fırsat)
            bullish_context = price > ema20 or price > sma50
            
            if state == 'FLAT':
                enter = False
                
                # 1) EMA5/EMA20 golden cross
                if not pd.isna(ema5_prev):
                    if ema5 > ema20 and ema5_prev <= ema20_prev and bullish_context:
                        enter = True
                
                # 2) EMA10/EMA20 golden cross (biraz daha yavaş ama güvenilir)
                if ema10 > ema20 and ema10_prev <= ema20_prev and bullish_context:
                    enter = True
                
                # 3) MACD bull cross
                if not pd.isna(macd_prev):
                    if macd > macd_sig and macd_prev <= macd_sig_prev and bullish_context:
                        enter = True
                
                # 4) RSI oversold bounce: 40 altından yukarı kırılma
                if not pd.isna(rsi_prev) and rsi > 43 and rsi_prev < 40 and price > sma50:
                    enter = True
                
                # 5) Fiyat SMA50 üstüne çıkıyor (trend başlangıcı)
                if price > sma50 and prev_price <= sma50_prev and macd_h > macd_h_prev:
                    enter = True
                
                # 6) Bollinger bounce (alt banddan dönüş, trendde)
                if not pd.isna(bb_pct_prev):
                    if bb_pct > 0.25 and bb_pct_prev < 0.15 and price > sma50:
                        enter = True
                
                if enter:
                    signals.iloc[i] = 1
                    state = 'LONG'
                    bars_in_position = 0
            
            elif state == 'LONG':
                bars_in_position += 1
                exit_signal = False
                
                # ===== ÇIKIŞ: SABIR (v3 tarzı SMA50 bazlı) =====
                
                # 1) Ana çıkış: Fiyat SMA50 altına + MACD negatif
                #    Bu kombinasyon gerçek trend kırılmasını gösterir
                if price < sma50 and macd_h < 0 and bars_in_position >= 5:
                    exit_signal = True
                
                # 2) SMA20/SMA50 death cross (ağır trend kırılması)
                if sma20 < sma50 and df['sma_20'].iloc[i-1] >= sma50_prev:
                    exit_signal = True
                
                # 3) Momentum çöküşü: RSI 70'ten hızlı düşüş + MACD bozuluyor
                if not pd.isna(rsi_prev) and rsi_prev > 70 and rsi < 60 and macd_h < macd_h_prev:
                    if bars_in_position >= 5:
                        exit_signal = True
                
                # 4) Fiyat hem EMA20 hem SMA50 altında (ciddi zayıflık)
                if price < ema20 and price < sma50 and rsi < 45:
                    exit_signal = True
                
                if exit_signal:
                    signals.iloc[i] = -1
                    state = 'FLAT'
        
        return signals
    
    @staticmethod
    def _strategy_adaptive(df: pd.DataFrame) -> pd.Series:
        """
        Adaptif Strateji Seçimi (Regime Detector)
        
        Son 60 günün performansına göre en iyi çalışan stratejiyi otomatik seç.
        Walk-forward verisi gösterdi ki her dönemde farklı strateji kazanıyor.
        """
        signals = pd.Series(0, index=df.index)
        
        sub_strategies = {
            'rsi': TechnicalEngine._strategy_rsi(df),
            'ma_crossover': TechnicalEngine._strategy_ma_crossover(df),
            'macd': TechnicalEngine._strategy_macd(df),
            'bollinger': TechnicalEngine._strategy_bollinger(df),
            'momentum': TechnicalEngine._strategy_momentum(df),
        }
        
        lookback = 60
        eval_period = 20
        active_strategy = None
        last_eval = 0
        
        for i in range(lookback + 1, len(df)):
            if i - last_eval >= eval_period or active_strategy is None:
                last_eval = i
                best_score = -999
                best_strat = 'momentum'
                
                for name, sig in sub_strategies.items():
                    window_sig = sig.iloc[max(0, i-lookback):i]
                    window_ret = df['returns'].iloc[max(0, i-lookback):i]
                    long_returns = window_ret[window_sig.shift(1) == 1].sum()
                    short_returns = -window_ret[window_sig.shift(1) == -1].sum()
                    total_perf = long_returns + short_returns
                    aligned = window_ret[window_sig.shift(1) == 1]
                    win_rate = (aligned > 0).mean() if len(aligned) > 0 else 0
                    score = total_perf + win_rate * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_strat = name
                
                active_strategy = best_strat
            
            signals.iloc[i] = sub_strategies[active_strategy].iloc[i]
        
        return signals
    
    @staticmethod
    def _strategy_vol_filtered(df: pd.DataFrame) -> pd.Series:
        """
        Volatilite Filtreli Combined Strateji
        
        ATR rejimi bazlı:
        - LOW vol: Agresif giriş
        - NORMAL vol: Normal combined
        - HIGH vol: Savunmacı (sadece güçlü sinyallerde gir)
        """
        signals = pd.Series(0, index=df.index)
        
        if 'ema_5' not in df.columns:
            df = df.copy()
            df['ema_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        
        required = ['sma_20', 'sma_50', 'ema_5', 'ema_10', 'ema_20', 'rsi', 
                     'macd_hist', 'macd', 'macd_signal', 'bb_pct', 'vol_regime', 'ma_alignment']
        for col in required:
            if col not in df.columns:
                return signals
        
        state = 'FLAT'
        bars_in_position = 0
        
        for i in range(2, len(df)):
            price = df['Close'].iloc[i]
            prev_price = df['Close'].iloc[i-1]
            vol_regime = df['vol_regime'].iloc[i]
            ma_align = df['ma_alignment'].iloc[i]
            
            ema5 = df['ema_5'].iloc[i]
            ema10 = df['ema_10'].iloc[i]
            ema20 = df['ema_20'].iloc[i]
            sma50 = df['sma_50'].iloc[i]
            sma50_prev = df['sma_50'].iloc[i-1]
            ema5_prev = df['ema_5'].iloc[i-1]
            ema10_prev = df['ema_10'].iloc[i-1]
            ema20_prev = df['ema_20'].iloc[i-1]
            
            rsi = df['rsi'].iloc[i]
            rsi_prev = df['rsi'].iloc[i-1]
            macd_h = df['macd_hist'].iloc[i]
            macd_h_prev = df['macd_hist'].iloc[i-1]
            macd = df['macd'].iloc[i]
            macd_sig = df['macd_signal'].iloc[i]
            macd_prev = df['macd'].iloc[i-1]
            macd_sig_prev = df['macd_signal'].iloc[i-1]
            bb_pct = df['bb_pct'].iloc[i]
            bb_pct_prev = df['bb_pct'].iloc[i-1] if i > 0 else 0.5
            
            if pd.isna(sma50) or pd.isna(ema20) or pd.isna(rsi) or pd.isna(vol_regime):
                continue
            
            if state == 'FLAT':
                enter = False
                
                if vol_regime == 'HIGH':
                    # Sadece çok güçlü sinyallerde gir
                    if ma_align >= 5 and rsi > 50 and macd_h > 0 and price > sma50:
                        if not pd.isna(macd_prev) and macd > macd_sig and macd_prev <= macd_sig_prev:
                            enter = True
                        if ema10 > ema20 and ema10_prev <= ema20_prev:
                            enter = True
                
                elif vol_regime == 'LOW':
                    # Agresif giriş
                    if price > ema20 and macd_h > macd_h_prev and rsi > 45:
                        enter = True
                    if not pd.isna(ema5_prev) and ema5 > ema20 and ema5_prev <= ema20_prev:
                        enter = True
                    if not pd.isna(bb_pct_prev) and bb_pct > 0.3 and bb_pct_prev < 0.2 and price > sma50:
                        enter = True
                
                else:
                    # NORMAL: v5 combined mantığı
                    bullish_context = price > ema20 or price > sma50
                    if not pd.isna(ema5_prev) and ema5 > ema20 and ema5_prev <= ema20_prev and bullish_context:
                        enter = True
                    if ema10 > ema20 and ema10_prev <= ema20_prev and bullish_context:
                        enter = True
                    if not pd.isna(macd_prev) and macd > macd_sig and macd_prev <= macd_sig_prev and bullish_context:
                        enter = True
                    if not pd.isna(rsi_prev) and rsi > 43 and rsi_prev < 40 and price > sma50:
                        enter = True
                    if price > sma50 and prev_price <= sma50_prev and macd_h > macd_h_prev:
                        enter = True
                
                if enter:
                    signals.iloc[i] = 1
                    state = 'LONG'
                    bars_in_position = 0
            
            elif state == 'LONG':
                bars_in_position += 1
                exit_signal = False
                
                if vol_regime == 'HIGH':
                    if price < ema20 and macd_h < 0 and bars_in_position >= 3:
                        exit_signal = True
                    if price < sma50 and rsi < 45:
                        exit_signal = True
                else:
                    sma20 = df['sma_20'].iloc[i]
                    sma20_prev = df['sma_20'].iloc[i-1]
                    if price < sma50 and macd_h < 0 and bars_in_position >= 5:
                        exit_signal = True
                    if sma20 < sma50 and sma20_prev >= sma50_prev:
                        exit_signal = True
                    if not pd.isna(rsi_prev) and rsi_prev > 70 and rsi < 60 and macd_h < macd_h_prev:
                        if bars_in_position >= 5:
                            exit_signal = True
                    if price < ema20 and price < sma50 and rsi < 45:
                        exit_signal = True
                
                if exit_signal:
                    signals.iloc[i] = -1
                    state = 'FLAT'
        
        return signals
    
    @staticmethod
    def _strategy_ml_enhanced(df: pd.DataFrame) -> pd.Series:
        """
        ML-Enhanced Strateji (LSTM Pattern Simülasyonu)
        
        LSTM'in yakalaması gereken pattern'ları teknik göstergelerle simüle eder:
        - Momentum rejimi, Trend alignment, Mean reversion, MACD momentum
        Volatiliteye göre eşik ayarı yapar.
        
        NOT: Tezde gerçek LSTM ile değiştirilecek placeholder.
        """
        signals = pd.Series(0, index=df.index)
        
        if 'ema_5' not in df.columns:
            df = df.copy()
            df['ema_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        
        required = ['sma_50', 'ema_20', 'rsi', 'macd_hist', 'bb_pct', 
                     'vol_regime', 'ma_alignment', 'stoch_rsi', 'returns']
        for col in required:
            if col not in df.columns:
                return signals
        
        state = 'FLAT'
        bars_in_position = 0
        
        for i in range(5, len(df)):
            price = df['Close'].iloc[i]
            rsi = df['rsi'].iloc[i]
            macd_h = df['macd_hist'].iloc[i]
            bb_pct = df['bb_pct'].iloc[i]
            vol_regime = df['vol_regime'].iloc[i]
            ma_align = df['ma_alignment'].iloc[i]
            sma50 = df['sma_50'].iloc[i]
            ema20 = df['ema_20'].iloc[i]
            
            if pd.isna(sma50) or pd.isna(rsi) or pd.isna(ma_align):
                continue
            
            # ML skor hesapla
            score = 0.0
            
            # 1) Momentum rejimi (%30)
            recent_ret = df['returns'].iloc[max(0,i-5):i]
            if len(recent_ret) > 0:
                pos_ratio = (recent_ret > 0).mean()
                score += ((pos_ratio - 0.5) * 2) * 0.30
            
            # 2) Trend alignment (%30)
            score += ((ma_align / 3) - 1) * 0.30
            
            # 3) Mean reversion (%20)
            if rsi < 35 and bb_pct < 0.2:
                score += 0.8 * 0.20
            elif rsi > 70 and bb_pct > 0.8:
                score += -0.8 * 0.20
            elif rsi < 45:
                score += 0.3 * 0.20
            elif rsi > 60:
                score += -0.3 * 0.20
            
            # 4) MACD momentum (%20)
            macd_norm = macd_h / (abs(price) * 0.01 + 1e-10)
            score += max(min(macd_norm, 1.0), -1.0) * 0.20
            
            # Volatilite eşik ayarı
            if vol_regime == 'HIGH':
                buy_thresh, sell_thresh = 0.55, -0.30
            elif vol_regime == 'LOW':
                buy_thresh, sell_thresh = 0.30, -0.25
            else:
                buy_thresh, sell_thresh = 0.40, -0.30
            
            if state == 'FLAT':
                if score > buy_thresh and (price > ema20 or price > sma50):
                    signals.iloc[i] = 1
                    state = 'LONG'
                    bars_in_position = 0
            
            elif state == 'LONG':
                bars_in_position += 1
                exit_signal = False
                
                if score < sell_thresh and bars_in_position >= 5:
                    exit_signal = True
                if price < sma50 and macd_h < 0 and bars_in_position >= 5:
                    exit_signal = True
                if price < ema20 and price < sma50 and rsi < 40:
                    exit_signal = True
                
                if exit_signal:
                    signals.iloc[i] = -1
                    state = 'FLAT'
        
        return signals

    @staticmethod
    def _strategy_trend_follow(df: pd.DataFrame) -> pd.Series:
        """
        Pure Trend Following - Basit ama etkili.
        SMA20 > SMA50 iken sürekli LONG kal, altındayken FLAT.
        BIST bull market'larda en iyi çalışan strateji.
        """
        signals = pd.Series(0, index=df.index)
        
        # Trende gir
        enter_long = ((df['sma_20'] > df['sma_50']) & 
                      (df['sma_20'].shift(1) <= df['sma_50'].shift(1)))
        # Ya da fiyat SMA50 üstüne çıkınca + MACD pozitif
        enter_long2 = ((df['Close'] > df['sma_50']) & 
                       (df['Close'].shift(1) <= df['sma_50'].shift(1)) &
                       (df['macd_hist'] > 0))
        
        signals[enter_long | enter_long2] = 1
        
        # Trendden çık
        exit_long = ((df['sma_20'] < df['sma_50']) & 
                     (df['sma_20'].shift(1) >= df['sma_50'].shift(1)))
        exit_long2 = ((df['Close'] < df['sma_50']) & 
                      (df['Close'].shift(1) >= df['sma_50'].shift(1)) &
                      (df['macd_hist'] < 0))
        
        signals[exit_long | exit_long2] = -1
        
        return signals


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Profesyonel backtesting motoru.
    
    Özellikler:
        - Gerçekçi komisyon + slippage
        - Stop loss / take profit / trailing stop / time stop
        - Risk-bazlı pozisyon boyutlandırma (Kelly / Fixed Fractional)
        - Çoklu pozisyon desteği
        - MFE / MAE analizi
        - Detaylı trade log
        - Benchmark karşılaştırma
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.trades: List[TradeRecord] = []
        self.portfolio_history: List[Dict] = []
        self.open_positions: Dict[str, TradeRecord] = {}
        self.trade_counter = 0
        self.cash = self.config.initial_capital
        self.peak_value = self.config.initial_capital
        self._halted = False
        
    def reset(self):
        """Motoru sıfırla"""
        self.trades = []
        self.portfolio_history = []
        self.open_positions = {}
        self.trade_counter = 0
        self.cash = self.config.initial_capital
        self.peak_value = self.config.initial_capital
        self._halted = False
    
    # -------------------------------------------------------------------------
    # Pozisyon Boyutlandırma
    # -------------------------------------------------------------------------
    
    def _calc_position_size(self, price: float, stop_price: float = None, atr: float = None) -> float:
        """
        Pozisyon boyutu hesapla.
        
        Swing trade modu (max_open_positions=1): Portföyün büyük kısmıyla gir.
        Multi-pozisyon modu: Risk-bazlı (Fixed Fractional) hesapla.
        """
        portfolio_value = self._portfolio_value_at_price({})
        
        if self.config.max_open_positions == 1:
            # Swing trade: portföyün max_position_pct kadarıyla gir
            max_investment = portfolio_value * self.config.max_position_pct
            cost_per_share = price * (1 + self.config.commission_rate + self.config.slippage_rate)
            shares = max_investment / cost_per_share
        else:
            # Multi-pozisyon: risk-bazlı
            risk_amount = portfolio_value * self.config.risk_per_trade_pct
            
            if stop_price is not None and stop_price != price:
                risk_per_share = abs(price - stop_price)
                shares = risk_amount / risk_per_share
            elif atr is not None and atr > 0:
                risk_per_share = atr * self.config.atr_stop_multiplier
                shares = risk_amount / risk_per_share
            else:
                shares = (portfolio_value * self.config.max_position_pct) / price
            
            max_shares = (portfolio_value * self.config.max_position_pct) / price
            shares = min(shares, max_shares)
        
        # Nakit limiti
        cost_per_share = price * (1 + self.config.commission_rate + self.config.slippage_rate)
        max_affordable = self.cash / cost_per_share
        shares = min(shares, max_affordable)
        
        return max(0, int(shares))
    
    def _portfolio_value_at_price(self, current_prices: Dict[str, float]) -> float:
        """Portföyün güncel değerini hesapla"""
        value = self.cash
        for ticker, trade in self.open_positions.items():
            price = current_prices.get(ticker, trade.entry_price)
            value += trade.shares * price
        return value
    
    # -------------------------------------------------------------------------
    # İşlem Açma / Kapama
    # -------------------------------------------------------------------------
    
    def _open_position(self, ticker: str, price: float, date: datetime, 
                       side: Side, atr: float = None):
        """Yeni pozisyon aç"""
        if self._halted:
            return None
        if ticker in self.open_positions:
            return None
        if len(self.open_positions) >= self.config.max_open_positions:
            return None
        
        # Stop loss fiyatı (ATR-bazlı)
        if self.config.use_atr_stops and atr is not None and atr > 0:
            if side == Side.LONG:
                stop_price = price - (atr * self.config.atr_stop_multiplier)
            else:
                stop_price = price + (atr * self.config.atr_stop_multiplier)
        else:
            if side == Side.LONG:
                stop_price = price * (1 - self.config.stop_loss_pct)
            else:
                stop_price = price * (1 + self.config.stop_loss_pct)
        
        shares = self._calc_position_size(price, stop_price, atr)
        if shares <= 0:
            return None
        
        # Maliyet hesapla
        entry_value = shares * price
        commission = entry_value * self.config.commission_rate
        slippage = entry_value * self.config.slippage_rate
        total_cost = entry_value + commission + slippage
        
        if total_cost > self.cash:
            # Nakitten karşılanabilen kadarını al
            shares = int(self.cash / (price * (1 + self.config.commission_rate + self.config.slippage_rate)))
            if shares <= 0:
                return None
            entry_value = shares * price
            commission = entry_value * self.config.commission_rate
            slippage = entry_value * self.config.slippage_rate
            total_cost = entry_value + commission + slippage
        
        self.trade_counter += 1
        trade = TradeRecord(
            trade_id=self.trade_counter,
            ticker=ticker,
            side="LONG" if side == Side.LONG else "SHORT",
            entry_date=date,
            entry_price=price,
            shares=shares,
            entry_value=entry_value,
            commission_paid=commission,
            slippage_cost=slippage,
        )
        
        self.cash -= total_cost
        self.open_positions[ticker] = trade
        return trade
    
    def _close_position(self, ticker: str, price: float, date: datetime, 
                        reason: ExitReason):
        """Pozisyonu kapat"""
        if ticker not in self.open_positions:
            return None
        
        trade = self.open_positions[ticker]
        
        exit_value = trade.shares * price
        commission = exit_value * self.config.commission_rate
        slippage = exit_value * self.config.slippage_rate
        
        # P&L
        if trade.side == "LONG":
            gross_pnl = (price - trade.entry_price) * trade.shares
        else:
            gross_pnl = (trade.entry_price - price) * trade.shares
        
        net_pnl = gross_pnl - commission - slippage - trade.commission_paid - trade.slippage_cost
        
        # Trade kaydını güncelle
        trade.exit_date = date
        trade.exit_price = price
        trade.exit_value = exit_value
        trade.commission_paid += commission
        trade.slippage_cost += slippage
        trade.gross_pnl = gross_pnl
        trade.net_pnl = net_pnl
        trade.return_pct = (net_pnl / trade.entry_value) * 100 if trade.entry_value > 0 else 0
        trade.holding_days = (date - trade.entry_date).days if isinstance(date, datetime) else 0
        trade.exit_reason = reason.value
        trade.risk_reward_actual = abs(trade.max_favorable / trade.max_adverse) if trade.max_adverse != 0 else 0
        
        self.cash += exit_value - commission - slippage
        self.trades.append(trade)
        del self.open_positions[ticker]
        
        return trade
    
    # -------------------------------------------------------------------------
    # Ana Backtest Loop
    # -------------------------------------------------------------------------
    
    def run(self, data: pd.DataFrame, signals: pd.Series, 
            ticker: str = "THYAO.IS") -> Dict:
        """
        Ana backtest çalıştırıcı.
        
        Args:
            data: OHLCV DataFrame (DatetimeIndex)
            signals: Sinyal serisi (1=AL, -1=SAT, 0=BEKLE)
            ticker: Hisse kodu
            
        Returns:
            dict: Kapsamlı sonuçlar
        """
        self.reset()
        
        df = TechnicalEngine.compute_indicators(data.copy())
        
        # Sinyalleri hizala
        signals = signals.reindex(df.index).fillna(0).astype(int)
        
        last_exit_date = None  # Re-entry cooldown tracking
        
        for i in range(1, len(df)):
            date = df.index[i]
            row = df.iloc[i]
            price = row['Close']
            high = row['High']
            low = row['Low']
            signal = signals.iloc[i]
            atr = row.get('atr', price * 0.02)
            if pd.isna(atr) or atr == 0:
                atr = price * 0.02
            
            current_prices = {ticker: price}
            
            # 1) Açık pozisyonları güncelle (stop/tp/trailing)
            self._update_open_positions(ticker, high, low, price, date, atr)
            
            # Track last exit for cooldown
            if ticker not in self.open_positions and self.trades:
                last_trade = self.trades[-1]
                if last_trade.ticker == ticker:
                    last_exit_date = last_trade.exit_date
            
            # 2) Portföy drawdown kontrolü
            portfolio_val = self._portfolio_value_at_price(current_prices)
            if portfolio_val > self.peak_value:
                self.peak_value = portfolio_val
            dd = (self.peak_value - portfolio_val) / self.peak_value
            if dd >= self.config.max_portfolio_drawdown:
                # Tüm pozisyonları kapat
                for t in list(self.open_positions.keys()):
                    self._close_position(t, price, date, ExitReason.MAX_DRAWDOWN)
                self._halted = True
            
            # 3) Sinyal bazlı işlem
            if not self._halted:
                in_position = ticker in self.open_positions
                
                # Re-entry cooldown kontrolü
                can_enter = True
                if self.config.allow_reentry and last_exit_date is not None:
                    if isinstance(date, datetime) and isinstance(last_exit_date, datetime):
                        days_since_exit = (date - last_exit_date).days
                        if days_since_exit < self.config.reentry_cooldown_days:
                            can_enter = False
                
                if signal == 1 and not in_position and can_enter:
                    self._open_position(ticker, price, date, Side.LONG, atr)
                elif signal == -1 and in_position:
                    self._close_position(ticker, price, date, ExitReason.SIGNAL)
                    last_exit_date = date
            
            # 4) Portföy geçmişi
            pv = self._portfolio_value_at_price(current_prices)
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': pv,
                'cash': self.cash,
                'positions_value': pv - self.cash,
                'n_positions': len(self.open_positions),
                'price': price,
            })
        
        # Son açık pozisyonları kapat
        if self.open_positions:
            last_price = df['Close'].iloc[-1]
            last_date = df.index[-1]
            for t in list(self.open_positions.keys()):
                self._close_position(t, last_price, last_date, ExitReason.END_OF_DATA)
        
        return self._compile_results(df, ticker)
    
    def _update_open_positions(self, ticker: str, high: float, low: float,
                                close: float, date: datetime, atr: float):
        """Açık pozisyonlar için ATR-bazlı stop/tp/trailing kontrolü"""
        if ticker not in self.open_positions:
            return
        
        trade = self.open_positions[ticker]
        entry = trade.entry_price
        
        if trade.side == "LONG":
            # MFE/MAE güncelle
            favorable = (high - entry) / entry
            adverse = (entry - low) / entry
            trade.max_favorable = max(trade.max_favorable, favorable)
            trade.max_adverse = max(trade.max_adverse, adverse)
            
            # ATR-bazlı veya sabit stop (hangisi daha geniş ise onu kullan)
            if self.config.use_atr_stops and atr > 0:
                atr_stop = entry - (atr * self.config.atr_stop_multiplier)
                fixed_stop = entry * (1 - self.config.stop_loss_pct)
                stop_price = min(atr_stop, fixed_stop)  # Daha geniş olan (düşük fiyat)
            else:
                stop_price = entry * (1 - self.config.stop_loss_pct)
            
            # Stop loss kontrolü
            if low <= stop_price:
                self._close_position(ticker, stop_price, date, ExitReason.STOP_LOSS)
                return
            
            # Take profit (0 ise kapalı - trend devam etsin)
            if self.config.take_profit_pct > 0:
                tp_price = entry * (1 + self.config.take_profit_pct)
                if high >= tp_price:
                    self._close_position(ticker, tp_price, date, ExitReason.TAKE_PROFIT)
                    return
            
            # Trailing stop - ATR bazlı
            if self.config.use_trailing_stop:
                # Kârda olduğumuzda trailing başlasın
                current_profit_pct = (close - entry) / entry
                if current_profit_pct > 0.02:  # Min %2 kârda trailing aktif
                    if self.config.use_atr_stops and atr > 0:
                        trail_price = high * (1 - max(atr * self.config.atr_trail_multiplier / high, 
                                                       self.config.trailing_stop_pct))
                    else:
                        trail_price = high * (1 - self.config.trailing_stop_pct)
                    
                    # Trail sadece entry'nin üstündeyse uygula (kârı koru)
                    if trail_price > entry and low <= trail_price:
                        self._close_position(ticker, trail_price, date, ExitReason.TRAILING_STOP)
                        return
            
            # Time stop
            if self.config.use_time_stop and self.config.time_stop_days > 0:
                holding = (date - trade.entry_date).days if isinstance(date, datetime) else 0
                if holding >= self.config.time_stop_days:
                    self._close_position(ticker, close, date, ExitReason.TIME_STOP)
                    return
    
    # -------------------------------------------------------------------------
    # Sonuç Derleme
    # -------------------------------------------------------------------------
    
    def _compile_results(self, df: pd.DataFrame, ticker: str) -> Dict:
        """Kapsamlı backtest sonuçlarını derle"""
        ph = pd.DataFrame(self.portfolio_history)
        if ph.empty:
            return self._empty_results()
        
        ph.set_index('date', inplace=True)
        
        # Portföy metrikleri
        initial = self.config.initial_capital
        final = ph['portfolio_value'].iloc[-1]
        total_return = (final - initial) / initial
        
        # Daily returns
        daily_returns = ph['portfolio_value'].pct_change().dropna()
        
        # Yıllıklaştırılmış getiri
        n_days = len(ph)
        n_years = n_days / 252
        if n_years > 0 and total_return > -1:
            cagr = (1 + total_return) ** (1 / n_years) - 1
        else:
            cagr = 0
        
        # Risk metrikleri
        annual_vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
        
        # Sharpe Ratio
        rf_daily = self.config.risk_free_rate / 252
        excess = daily_returns - rf_daily
        sharpe = (excess.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Sortino Ratio (sadece negatif volatilite)
        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
        sortino = (daily_returns.mean() * 252 - self.config.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Calmar Ratio
        cummax = ph['portfolio_value'].cummax()
        drawdown = (ph['portfolio_value'] - cummax) / cummax
        max_dd = drawdown.min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Max Drawdown detay
        dd_end_idx = drawdown.idxmin()
        dd_start_idx = ph['portfolio_value'][:dd_end_idx].idxmax() if dd_end_idx is not None else None
        
        # Trade metrikleri
        trades_df = self._trades_to_dataframe()
        trade_stats = self._compute_trade_stats(trades_df)
        
        # Buy & Hold karşılaştırma
        bh_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        alpha = total_return - bh_return
        
        # Drawdown serisi
        dd_series = drawdown.copy()
        
        # Aylık getiriler
        monthly_returns = ph['portfolio_value'].resample('ME').last().pct_change().dropna()
        
        # Win/Loss streak
        if not trades_df.empty:
            wins = (trades_df['net_pnl'] > 0).astype(int)
            max_win_streak = self._max_consecutive(wins, 1)
            max_loss_streak = self._max_consecutive(wins, 0)
        else:
            max_win_streak = max_loss_streak = 0
        
        results = {
            # Genel
            'ticker': ticker,
            'start_date': ph.index[0],
            'end_date': ph.index[-1],
            'trading_days': n_days,
            'years': round(n_years, 2),
            
            # Sermaye
            'initial_capital': initial,
            'final_value': round(final, 2),
            'net_profit': round(final - initial, 2),
            
            # Getiri
            'total_return_pct': round(total_return * 100, 2),
            'cagr_pct': round(cagr * 100, 2),
            'buy_hold_return_pct': round(bh_return * 100, 2),
            'alpha_pct': round(alpha * 100, 2),
            
            # Risk
            'annual_volatility_pct': round(annual_vol * 100, 2),
            'sharpe_ratio': round(sharpe, 3),
            'sortino_ratio': round(sortino, 3),
            'calmar_ratio': round(calmar, 3),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'max_dd_start': dd_start_idx,
            'max_dd_end': dd_end_idx,
            
            # İşlem
            **trade_stats,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            
            # Serileri de ekle (grafik için)
            '_portfolio_history': ph,
            '_drawdown_series': dd_series,
            '_daily_returns': daily_returns,
            '_monthly_returns': monthly_returns,
            '_trades_df': trades_df,
            '_ohlcv': df,  # Grafik için OHLCV verisi
            
            # Config
            '_config': self.config,
        }
        
        return results
    
    def _empty_results(self) -> Dict:
        return {
            'ticker': '', 'total_return_pct': 0, 'sharpe_ratio': 0,
            'max_drawdown_pct': 0, 'total_trades': 0, 'win_rate_pct': 0,
        }
    
    def _trades_to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        records = []
        for t in self.trades:
            records.append({
                'trade_id': t.trade_id,
                'ticker': t.ticker,
                'side': t.side,
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'shares': t.shares,
                'entry_value': t.entry_value,
                'exit_value': t.exit_value,
                'commission': t.commission_paid,
                'slippage': t.slippage_cost,
                'gross_pnl': t.gross_pnl,
                'net_pnl': t.net_pnl,
                'return_pct': t.return_pct,
                'holding_days': t.holding_days,
                'exit_reason': t.exit_reason,
                'mfe': t.max_favorable,
                'mae': t.max_adverse,
            })
        return pd.DataFrame(records)
    
    def _compute_trade_stats(self, trades_df: pd.DataFrame) -> Dict:
        """Detaylı trade istatistikleri"""
        if trades_df.empty:
            return {
                'total_trades': 0, 'win_rate_pct': 0, 'profit_factor': 0,
                'avg_trade_pnl': 0, 'avg_win': 0, 'avg_loss': 0,
                'largest_win': 0, 'largest_loss': 0,
                'avg_holding_days': 0, 'expectancy': 0,
                'payoff_ratio': 0, 'total_commission': 0,
                'exit_reasons': {},
            }
        
        wins = trades_df[trades_df['net_pnl'] > 0]
        losses = trades_df[trades_df['net_pnl'] <= 0]
        
        total_profit = wins['net_pnl'].sum() if len(wins) > 0 else 0
        total_loss = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 0
        
        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0
        payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Expectancy (beklenen değer)
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Exit reason dağılımı
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate_pct': round(win_rate * 100, 2),
            'profit_factor': round(profit_factor, 3),
            'avg_trade_pnl': round(trades_df['net_pnl'].mean(), 2),
            'avg_trade_return_pct': round(trades_df['return_pct'].mean(), 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'largest_win': round(trades_df['net_pnl'].max(), 2),
            'largest_loss': round(trades_df['net_pnl'].min(), 2),
            'avg_holding_days': round(trades_df['holding_days'].mean(), 1),
            'median_holding_days': round(trades_df['holding_days'].median(), 1),
            'expectancy': round(expectancy, 2),
            'payoff_ratio': round(payoff, 3),
            'total_commission': round(trades_df['commission'].sum(), 2),
            'total_slippage': round(trades_df['slippage'].sum(), 2),
            'exit_reasons': exit_reasons,
            'avg_mfe': round(trades_df['mfe'].mean() * 100, 2),
            'avg_mae': round(trades_df['mae'].mean() * 100, 2),
        }
    
    @staticmethod
    def _max_consecutive(series, value):
        """Seri içinde ardışık 'value' sayısının max'ını bul"""
        max_count = 0
        count = 0
        for v in series:
            if v == value:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count


# =============================================================================
# WALK-FORWARD OPTİMİZASYON
# =============================================================================

class WalkForwardAnalyzer:
    """
    Walk-Forward Analiz (WFA)
    
    Eğitim penceresinde en iyi stratejiyi seç,
    test penceresinde doğrula. Gerçekçi out-of-sample test.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.windows: List[Dict] = []
        self.results: List[Dict] = []
    
    def run(self, data: pd.DataFrame, strategies: List[str] = None,
            ticker: str = "THYAO.IS") -> Dict:
        """
        Walk-forward analiz çalıştır.
        
        Args:
            data: OHLCV DataFrame
            strategies: Test edilecek strateji listesi
            ticker: Hisse kodu
        """
        if strategies is None:
            strategies = ["rsi", "ma_crossover", "macd", "bollinger", "momentum", "trend_follow", "adaptive", "vol_filtered", "ml_enhanced", "combined"]
        
        self.windows = []
        self.results = []
        
        train_days = self.config.wf_train_days
        test_days = self.config.wf_test_days
        step_days = self.config.wf_step_days
        
        n = len(data)
        start = 0
        window_id = 0
        
        while start + train_days + test_days <= n:
            window_id += 1
            train_end = start + train_days
            test_end = min(train_end + test_days, n)
            
            train_data = data.iloc[start:train_end]
            test_data = data.iloc[train_end:test_end]
            
            if len(test_data) < 10:
                break
            
            # Her stratejiyi eğitim verisiyle test et
            best_strategy = None
            best_sharpe = -999
            train_results_all = {}
            
            for strat in strategies:
                engine = BacktestEngine(self.config)
                signals = TechnicalEngine.generate_signals(train_data, strat)
                result = engine.run(train_data, signals, ticker)
                train_results_all[strat] = result
                
                if result.get('sharpe_ratio', -999) > best_sharpe:
                    best_sharpe = result['sharpe_ratio']
                    best_strategy = strat
            
            # En iyi stratejiyi test verisinde çalıştır
            engine = BacktestEngine(self.config)
            test_signals = TechnicalEngine.generate_signals(test_data, best_strategy)
            test_result = engine.run(test_data, test_signals, ticker)
            
            window_info = {
                'window_id': window_id,
                'train_start': data.index[start],
                'train_end': data.index[train_end - 1],
                'test_start': data.index[train_end],
                'test_end': data.index[test_end - 1],
                'best_strategy': best_strategy,
                'train_sharpe': round(best_sharpe, 3),
                'test_sharpe': round(test_result.get('sharpe_ratio', 0), 3),
                'test_return_pct': test_result.get('total_return_pct', 0),
                'test_max_dd_pct': test_result.get('max_drawdown_pct', 0),
                'test_win_rate': test_result.get('win_rate_pct', 0),
                'test_trades': test_result.get('total_trades', 0),
            }
            
            self.windows.append(window_info)
            self.results.append(test_result)
            
            start += step_days
        
        return self._compile_wf_results()
    
    def _compile_wf_results(self) -> Dict:
        """Walk-forward sonuçlarını derle"""
        if not self.windows:
            return {'windows': [], 'summary': {}}
        
        wf_df = pd.DataFrame(self.windows)
        
        # Out-of-sample performans
        avg_return = wf_df['test_return_pct'].mean()
        avg_sharpe = wf_df['test_sharpe'].mean()
        avg_dd = wf_df['test_max_dd_pct'].mean()
        
        # Walk-forward efficiency (OOS / IS oran)
        is_sharpes = wf_df['train_sharpe'].values
        oos_sharpes = wf_df['test_sharpe'].values
        wfe = np.mean(oos_sharpes / (is_sharpes + 1e-10)) if len(is_sharpes) > 0 else 0
        
        # En çok seçilen strateji
        strategy_counts = wf_df['best_strategy'].value_counts().to_dict()
        
        # Pozitif OOS pencere oranı
        positive_windows = (wf_df['test_return_pct'] > 0).sum()
        positive_ratio = positive_windows / len(wf_df)
        
        summary = {
            'total_windows': len(self.windows),
            'avg_oos_return_pct': round(avg_return, 2),
            'avg_oos_sharpe': round(avg_sharpe, 3),
            'avg_oos_max_dd_pct': round(avg_dd, 2),
            'wf_efficiency': round(wfe, 3),
            'positive_windows_pct': round(positive_ratio * 100, 1),
            'strategy_selection': strategy_counts,
        }
        
        return {
            'windows': self.windows,
            'windows_df': wf_df,
            'summary': summary,
        }


# =============================================================================
# MONTE CARLO SİMÜLASYONU
# =============================================================================

class MonteCarloSimulator:
    """
    Monte Carlo simülasyonu ile risk analizi.
    Trade dağılımını bootstrap yöntemiyle yeniden örnekler.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
    
    def run(self, trades_df: pd.DataFrame, n_simulations: int = None) -> Dict:
        """
        Monte Carlo simülasyonu çalıştır.
        
        Args:
            trades_df: Tamamlanmış işlemler DataFrame'i
            n_simulations: Simülasyon sayısı
        """
        n_sim = n_simulations or self.config.mc_simulations
        
        if trades_df.empty:
            return {'error': 'İşlem verisi yok'}
        
        trade_returns = trades_df['return_pct'].values / 100  # Yüzdeyi orana çevir
        n_trades = len(trade_returns)
        
        final_values = []
        max_drawdowns = []
        sharpe_ratios = []
        
        initial = self.config.initial_capital
        
        for _ in range(n_sim):
            # Bootstrap: rastgele sırayla trade'leri yeniden örnekle
            sampled = np.random.choice(trade_returns, size=n_trades, replace=True)
            
            # Equity eğrisi
            equity = [initial]
            for ret in sampled:
                equity.append(equity[-1] * (1 + ret))
            equity = np.array(equity)
            
            final_values.append(equity[-1])
            
            # Max drawdown
            cummax = np.maximum.accumulate(equity)
            dd = (equity - cummax) / cummax
            max_drawdowns.append(dd.min())
            
            # Sharpe (basitleştirilmiş)
            daily_equiv = np.diff(equity) / equity[:-1]
            if daily_equiv.std() > 0:
                sr = np.mean(daily_equiv) / np.std(daily_equiv) * np.sqrt(252 / max(n_trades, 1))
            else:
                sr = 0
            sharpe_ratios.append(sr)
        
        final_values = np.array(final_values)
        max_drawdowns = np.array(max_drawdowns)
        sharpe_ratios = np.array(sharpe_ratios)
        
        conf = self.config.mc_confidence
        lower_pct = (1 - conf) / 2 * 100
        upper_pct = (1 + conf) / 2 * 100
        
        results = {
            'n_simulations': n_sim,
            'confidence': conf,
            
            # Final değer dağılımı
            'final_value_mean': round(np.mean(final_values), 2),
            'final_value_median': round(np.median(final_values), 2),
            'final_value_std': round(np.std(final_values), 2),
            'final_value_ci_lower': round(np.percentile(final_values, lower_pct), 2),
            'final_value_ci_upper': round(np.percentile(final_values, upper_pct), 2),
            'final_value_worst': round(np.min(final_values), 2),
            'final_value_best': round(np.max(final_values), 2),
            'prob_profit': round((final_values > initial).mean() * 100, 1),
            'prob_double': round((final_values > initial * 2).mean() * 100, 1),
            'prob_halve': round((final_values < initial * 0.5).mean() * 100, 1),
            
            # Drawdown dağılımı
            'max_dd_mean_pct': round(np.mean(max_drawdowns) * 100, 2),
            'max_dd_worst_pct': round(np.min(max_drawdowns) * 100, 2),
            'max_dd_ci_pct': round(np.percentile(max_drawdowns, lower_pct) * 100, 2),
            
            # Sharpe dağılımı
            'sharpe_mean': round(np.mean(sharpe_ratios), 3),
            'sharpe_median': round(np.median(sharpe_ratios), 3),
            'sharpe_ci_lower': round(np.percentile(sharpe_ratios, lower_pct), 3),
            'sharpe_ci_upper': round(np.percentile(sharpe_ratios, upper_pct), 3),
            
            # Ham veriler (histogram için)
            '_final_values': final_values,
            '_max_drawdowns': max_drawdowns,
            '_sharpe_ratios': sharpe_ratios,
        }
        
        return results


# =============================================================================
# ÇOKLU VARLIK BACKTEST
# =============================================================================

class MultiAssetBacktester:
    """
    Birden fazla BIST hissesini aynı anda backtest et.
    Karşılaştırmalı performans analizi.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.results: Dict[str, Dict] = {}
    
    def run(self, tickers: List[str], start_date: str, end_date: str,
            strategy: str = "combined") -> Dict:
        """
        Çoklu varlık backtest.
        
        Args:
            tickers: Hisse kodu listesi (örn: ["THYAO.IS", "GARAN.IS", "TUPRS.IS"])
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            strategy: Kullanılacak strateji
        """
        self.results = {}
        
        for ticker in tickers:
            try:
                if YF_AVAILABLE:
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    data = data.dropna()
                else:
                    continue
                
                if len(data) < 60:
                    continue
                
                engine = BacktestEngine(self.config)
                signals = TechnicalEngine.generate_signals(data, strategy)
                result = engine.run(data, signals, ticker)
                self.results[ticker] = result
                
            except Exception as e:
                self.results[ticker] = {'error': str(e)}
        
        return self._compile_comparison()
    
    def _compile_comparison(self) -> Dict:
        """Karşılaştırma tablosu oluştur"""
        rows = []
        for ticker, res in self.results.items():
            if 'error' in res:
                continue
            rows.append({
                'Hisse': ticker.replace('.IS', ''),
                'Getiri (%)': res.get('total_return_pct', 0),
                'CAGR (%)': res.get('cagr_pct', 0),
                'Sharpe': res.get('sharpe_ratio', 0),
                'Sortino': res.get('sortino_ratio', 0),
                'Max DD (%)': res.get('max_drawdown_pct', 0),
                'Win Rate (%)': res.get('win_rate_pct', 0),
                'Profit Factor': res.get('profit_factor', 0),
                'İşlem Sayısı': res.get('total_trades', 0),
                'Alpha (%)': res.get('alpha_pct', 0),
            })
        
        comparison_df = pd.DataFrame(rows)
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('Sharpe', ascending=False)
        
        return {
            'comparison': comparison_df,
            'individual_results': self.results,
        }


# =============================================================================
# STRATEJİ KARŞILAŞTIRMA
# =============================================================================

class StrategyComparator:
    """Farklı stratejileri aynı veri üzerinde karşılaştır"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
    
    def run(self, data: pd.DataFrame, strategies: List[str] = None,
            ticker: str = "THYAO.IS") -> pd.DataFrame:
        """Tüm stratejileri karşılaştır"""
        if strategies is None:
            strategies = ["rsi", "ma_crossover", "macd", "bollinger", "momentum", "trend_follow", "adaptive", "vol_filtered", "ml_enhanced", "combined"]
        
        rows = []
        for strat in strategies:
            engine = BacktestEngine(self.config)
            signals = TechnicalEngine.generate_signals(data, strat)
            result = engine.run(data, signals, ticker)
            
            rows.append({
                'Strateji': strat.upper(),
                'Getiri (%)': result.get('total_return_pct', 0),
                'CAGR (%)': result.get('cagr_pct', 0),
                'Sharpe': result.get('sharpe_ratio', 0),
                'Sortino': result.get('sortino_ratio', 0),
                'Calmar': result.get('calmar_ratio', 0),
                'Max DD (%)': result.get('max_drawdown_pct', 0),
                'Win Rate (%)': result.get('win_rate_pct', 0),
                'Profit Factor': result.get('profit_factor', 0),
                'İşlem Sayısı': result.get('total_trades', 0),
                'Ort. İşlem Süresi': result.get('avg_holding_days', 0),
                'Expectancy': result.get('expectancy', 0),
                'Alpha (%)': result.get('alpha_pct', 0),
            })
        
        df = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
        return df


# =============================================================================
# RAPOR ÜRETİCİ - HTML
# =============================================================================

class ReportGenerator:
    """Detaylı HTML backtest raporu üretir"""
    
    @staticmethod
    def generate_html_report(results: Dict, 
                             wf_results: Dict = None,
                             mc_results: Dict = None,
                             strategy_comparison: pd.DataFrame = None,
                             output_path: str = "backtest_report.html") -> str:
        """Kapsamlı HTML rapor üret"""
        
        r = results
        html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OmenQuant Backtest Raporu - {r.get('ticker', 'N/A')}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ 
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background: #0a0a1a; color: #e0e0e0; line-height: 1.6;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    
    /* Header */
    .header {{
        text-align: center; padding: 40px 20px; margin-bottom: 30px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px; border: 1px solid #333;
    }}
    .header h1 {{
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .header .subtitle {{ color: #888; font-size: 1.1rem; margin-top: 8px; }}
    .header .meta {{ color: #666; font-size: 0.9rem; margin-top: 15px; }}
    
    /* KPI Cards */
    .kpi-grid {{
        display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 15px; margin-bottom: 30px;
    }}
    .kpi-card {{
        background: #1a1a2e; border-radius: 12px; padding: 20px;
        border: 1px solid #2a2a4a; text-align: center;
        transition: transform 0.2s;
    }}
    .kpi-card:hover {{ transform: translateY(-2px); border-color: #667eea; }}
    .kpi-value {{ font-size: 1.8rem; font-weight: 700; margin: 8px 0; }}
    .kpi-label {{ font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
    .positive {{ color: #38ef7d; }}
    .negative {{ color: #f45c43; }}
    .neutral {{ color: #ffd700; }}
    
    /* Sections */
    .section {{
        background: #1a1a2e; border-radius: 16px; padding: 25px;
        margin-bottom: 25px; border: 1px solid #2a2a4a;
    }}
    .section h2 {{
        font-size: 1.4rem; margin-bottom: 20px; padding-bottom: 10px;
        border-bottom: 2px solid #333;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    
    /* Tables */
    table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
    th {{ 
        background: #16213e; padding: 12px 15px; text-align: left;
        font-weight: 600; font-size: 0.9rem; color: #aaa;
        border-bottom: 2px solid #333;
    }}
    td {{ 
        padding: 10px 15px; border-bottom: 1px solid #222;
        font-size: 0.95rem;
    }}
    tr:hover td {{ background: #16213e; }}
    
    /* Two column layout */
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 25px; }}
    @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
    
    /* Stat row */
    .stat-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #222; }}
    .stat-label {{ color: #888; }}
    .stat-value {{ font-weight: 600; }}
    
    /* Badge */
    .badge {{
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.8rem; font-weight: 600;
    }}
    .badge-green {{ background: rgba(56, 239, 125, 0.15); color: #38ef7d; }}
    .badge-red {{ background: rgba(244, 92, 67, 0.15); color: #f45c43; }}
    .badge-yellow {{ background: rgba(255, 215, 0, 0.15); color: #ffd700; }}
    
    /* Rating */
    .rating-box {{
        text-align: center; padding: 30px; border-radius: 12px; margin-top: 20px;
    }}
    .rating-score {{ font-size: 4rem; font-weight: 800; }}
    .rating-label {{ font-size: 1.2rem; margin-top: 10px; }}
    
    /* Progress bar */
    .progress-bar {{
        background: #222; border-radius: 10px; height: 8px; margin: 5px 0;
    }}
    .progress-fill {{
        height: 100%; border-radius: 10px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }}
    
    /* Footer */
    .footer {{
        text-align: center; padding: 20px; color: #555; font-size: 0.85rem;
        margin-top: 30px;
    }}
</style>
</head>
<body>
<div class="container">
"""
        
        # ---- HEADER ----
        ticker_name = r.get('ticker', 'N/A').replace('.IS', '')
        total_ret = r.get('total_return_pct', 0)
        ret_class = 'positive' if total_ret > 0 else 'negative'
        
        html += f"""
<div class="header">
    <h1>⚡ OmenQuant Backtest Raporu</h1>
    <div class="subtitle">{ticker_name} | Profesyonel Performans Analizi</div>
    <div class="meta">
        📅 {r.get('start_date', 'N/A')} → {r.get('end_date', 'N/A')} | 
        📊 {r.get('trading_days', 0)} işlem günü ({r.get('years', 0)} yıl) |
        💰 Başlangıç: {r.get('initial_capital', 0):,.0f} TL
    </div>
</div>
"""
        
        # ---- KPI CARDS ----
        sharpe = r.get('sharpe_ratio', 0)
        sharpe_class = 'positive' if sharpe > 0.5 else ('neutral' if sharpe > 0 else 'negative')
        
        max_dd = r.get('max_drawdown_pct', 0)
        dd_class = 'positive' if max_dd > -15 else ('neutral' if max_dd > -25 else 'negative')
        
        wr = r.get('win_rate_pct', 0)
        wr_class = 'positive' if wr > 55 else ('neutral' if wr > 45 else 'negative')
        
        alpha = r.get('alpha_pct', 0)
        alpha_class = 'positive' if alpha > 0 else 'negative'
        
        html += f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Net Getiri</div>
        <div class="kpi-value {ret_class}">{total_ret:+.2f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">CAGR</div>
        <div class="kpi-value {ret_class}">{r.get('cagr_pct', 0):+.2f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Sharpe Ratio</div>
        <div class="kpi-value {sharpe_class}">{sharpe:.3f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Max Drawdown</div>
        <div class="kpi-value {dd_class}">{max_dd:.2f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Win Rate</div>
        <div class="kpi-value {wr_class}">{wr:.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Alpha (vs B&H)</div>
        <div class="kpi-value {alpha_class}">{alpha:+.2f}%</div>
    </div>
</div>
"""
        
        # ---- GRAFİKLER (Plotly) ----
        chart_html = ReportGenerator._build_charts(r)
        html += chart_html
        
        # ---- GETİRİ & RİSK DETAY ----
        html += """<div class="two-col">"""
        
        # Getiri bölümü
        html += f"""
<div class="section">
    <h2>📈 Getiri Analizi</h2>
    <div class="stat-row"><span class="stat-label">Toplam Getiri</span><span class="stat-value {ret_class}">{total_ret:+.2f}%</span></div>
    <div class="stat-row"><span class="stat-label">CAGR</span><span class="stat-value">{r.get('cagr_pct', 0):+.2f}%</span></div>
    <div class="stat-row"><span class="stat-label">Buy & Hold Getiri</span><span class="stat-value">{r.get('buy_hold_return_pct', 0):+.2f}%</span></div>
    <div class="stat-row"><span class="stat-label">Alpha</span><span class="stat-value {alpha_class}">{alpha:+.2f}%</span></div>
    <div class="stat-row"><span class="stat-label">Net Kâr/Zarar</span><span class="stat-value {ret_class}">{r.get('net_profit', 0):+,.2f} TL</span></div>
    <div class="stat-row"><span class="stat-label">Bitiş Değeri</span><span class="stat-value">{r.get('final_value', 0):,.2f} TL</span></div>
</div>
"""
        
        # Risk bölümü
        sortino = r.get('sortino_ratio', 0)
        calmar = r.get('calmar_ratio', 0)
        html += f"""
<div class="section">
    <h2>⚠️ Risk Analizi</h2>
    <div class="stat-row"><span class="stat-label">Sharpe Ratio</span><span class="stat-value">{sharpe:.3f}</span></div>
    <div class="stat-row"><span class="stat-label">Sortino Ratio</span><span class="stat-value">{sortino:.3f}</span></div>
    <div class="stat-row"><span class="stat-label">Calmar Ratio</span><span class="stat-value">{calmar:.3f}</span></div>
    <div class="stat-row"><span class="stat-label">Yıllık Volatilite</span><span class="stat-value">{r.get('annual_volatility_pct', 0):.2f}%</span></div>
    <div class="stat-row"><span class="stat-label">Max Drawdown</span><span class="stat-value {dd_class}">{max_dd:.2f}%</span></div>
    <div class="stat-row"><span class="stat-label">Drawdown Başlangıcı</span><span class="stat-value">{r.get('max_dd_start', 'N/A')}</span></div>
    <div class="stat-row"><span class="stat-label">Drawdown Sonu</span><span class="stat-value">{r.get('max_dd_end', 'N/A')}</span></div>
</div>
"""
        html += """</div>"""
        
        # ---- İŞLEM İSTATİSTİKLERİ ----
        html += """<div class="two-col">"""
        
        html += f"""
<div class="section">
    <h2>🎯 İşlem İstatistikleri</h2>
    <div class="stat-row"><span class="stat-label">Toplam İşlem</span><span class="stat-value">{r.get('total_trades', 0)}</span></div>
    <div class="stat-row"><span class="stat-label">Kazançlı İşlem</span><span class="stat-value positive">{r.get('winning_trades', 0)}</span></div>
    <div class="stat-row"><span class="stat-label">Zararlı İşlem</span><span class="stat-value negative">{r.get('losing_trades', 0)}</span></div>
    <div class="stat-row"><span class="stat-label">Win Rate</span><span class="stat-value">{r.get('win_rate_pct', 0):.1f}%</span></div>
    <div class="stat-row"><span class="stat-label">Profit Factor</span><span class="stat-value">{r.get('profit_factor', 0):.3f}</span></div>
    <div class="stat-row"><span class="stat-label">Payoff Ratio</span><span class="stat-value">{r.get('payoff_ratio', 0):.3f}</span></div>
    <div class="stat-row"><span class="stat-label">Expectancy</span><span class="stat-value">{r.get('expectancy', 0):+.2f} TL</span></div>
</div>
"""
        
        html += f"""
<div class="section">
    <h2>📊 İşlem Detayları</h2>
    <div class="stat-row"><span class="stat-label">Ort. İşlem Getirisi</span><span class="stat-value">{r.get('avg_trade_return_pct', 0):+.2f}%</span></div>
    <div class="stat-row"><span class="stat-label">Ort. Kazanç</span><span class="stat-value positive">{r.get('avg_win', 0):+,.2f} TL</span></div>
    <div class="stat-row"><span class="stat-label">Ort. Kayıp</span><span class="stat-value negative">{r.get('avg_loss', 0):+,.2f} TL</span></div>
    <div class="stat-row"><span class="stat-label">En Büyük Kazanç</span><span class="stat-value positive">{r.get('largest_win', 0):+,.2f} TL</span></div>
    <div class="stat-row"><span class="stat-label">En Büyük Kayıp</span><span class="stat-value negative">{r.get('largest_loss', 0):+,.2f} TL</span></div>
    <div class="stat-row"><span class="stat-label">Ort. Pozisyon Süresi</span><span class="stat-value">{r.get('avg_holding_days', 0):.1f} gün</span></div>
    <div class="stat-row"><span class="stat-label">Max Kazanç Serisi</span><span class="stat-value positive">{r.get('max_win_streak', 0)}</span></div>
    <div class="stat-row"><span class="stat-label">Max Kayıp Serisi</span><span class="stat-value negative">{r.get('max_loss_streak', 0)}</span></div>
</div>
"""
        html += """</div>"""
        
        # ---- MALİYET ANALİZİ ----
        html += f"""
<div class="section">
    <h2>💸 Maliyet & MFE/MAE Analizi</h2>
    <div class="two-col">
        <div>
            <div class="stat-row"><span class="stat-label">Toplam Komisyon</span><span class="stat-value">{r.get('total_commission', 0):,.2f} TL</span></div>
            <div class="stat-row"><span class="stat-label">Toplam Slippage</span><span class="stat-value">{r.get('total_slippage', 0):,.2f} TL</span></div>
            <div class="stat-row"><span class="stat-label">Toplam Maliyet</span><span class="stat-value">{r.get('total_commission', 0) + r.get('total_slippage', 0):,.2f} TL</span></div>
        </div>
        <div>
            <div class="stat-row"><span class="stat-label">Ort. MFE (Max Kazanç Potansiyeli)</span><span class="stat-value">{r.get('avg_mfe', 0):.2f}%</span></div>
            <div class="stat-row"><span class="stat-label">Ort. MAE (Max Kayıp Noktası)</span><span class="stat-value">{r.get('avg_mae', 0):.2f}%</span></div>
        </div>
    </div>
</div>
"""
        
        # ---- EXIT REASON DAĞILIMI ----
        exit_reasons = r.get('exit_reasons', {})
        if exit_reasons:
            html += """<div class="section"><h2>🚪 Çıkış Sebepleri</h2><table><tr><th>Sebep</th><th>Sayı</th><th>Oran</th></tr>"""
            total = sum(exit_reasons.values())
            for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
                pct = (count / total) * 100 if total > 0 else 0
                emoji = {'SIGNAL': '📊', 'STOP_LOSS': '🛑', 'TAKE_PROFIT': '🎯', 
                         'TRAILING_STOP': '📉', 'TIME_STOP': '⏰', 'END_OF_DATA': '🏁',
                         'MAX_DRAWDOWN': '💥'}.get(reason, '❓')
                html += f"""<tr><td>{emoji} {reason}</td><td>{count}</td><td>{pct:.1f}%</td></tr>"""
            html += """</table></div>"""
        
        # ---- STRATEJİ KARŞILAŞTIRMA ----
        if strategy_comparison is not None and not strategy_comparison.empty:
            html += """<div class="section"><h2>🔄 Strateji Karşılaştırma</h2><table><tr>"""
            for col in strategy_comparison.columns:
                html += f"<th>{col}</th>"
            html += "</tr>"
            for _, row in strategy_comparison.iterrows():
                html += "<tr>"
                for col in strategy_comparison.columns:
                    val = row[col]
                    if isinstance(val, float):
                        css = ""
                        if col in ['Getiri (%)', 'Alpha (%)', 'Sharpe', 'Sortino', 'Calmar']:
                            css = ' class="positive"' if val > 0 else ' class="negative"'
                        html += f"<td{css}>{val:.2f}</td>"
                    else:
                        html += f"<td>{val}</td>"
                html += "</tr>"
            html += "</table></div>"
        
        # ---- WALK-FORWARD ----
        if wf_results and wf_results.get('summary'):
            wf = wf_results['summary']
            html += f"""
<div class="section">
    <h2>🔀 Walk-Forward Analiz</h2>
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Pencere Sayısı</div>
            <div class="kpi-value neutral">{wf.get('total_windows', 0)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">OOS Ort. Getiri</div>
            <div class="kpi-value {'positive' if wf.get('avg_oos_return_pct', 0) > 0 else 'negative'}">{wf.get('avg_oos_return_pct', 0):+.2f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">OOS Ort. Sharpe</div>
            <div class="kpi-value">{wf.get('avg_oos_sharpe', 0):.3f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">WF Efficiency</div>
            <div class="kpi-value">{wf.get('wf_efficiency', 0):.3f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Pozitif Pencere</div>
            <div class="kpi-value">{wf.get('positive_windows_pct', 0):.1f}%</div>
        </div>
    </div>
"""
            # Strateji seçim dağılımı
            strat_sel = wf.get('strategy_selection', {})
            if strat_sel:
                html += """<h3 style="margin-top:20px; color:#aaa;">Seçilen Stratejiler</h3><table><tr><th>Strateji</th><th>Seçilme Sayısı</th></tr>"""
                for s, c in sorted(strat_sel.items(), key=lambda x: -x[1]):
                    html += f"<tr><td>{s.upper()}</td><td>{c}</td></tr>"
                html += "</table>"
            
            # Pencere detayları
            if wf_results.get('windows'):
                html += """<h3 style="margin-top:20px; color:#aaa;">Pencere Detayları</h3><table>
                <tr><th>#</th><th>Test Dönemi</th><th>Strateji</th><th>IS Sharpe</th><th>OOS Sharpe</th><th>OOS Getiri</th><th>OOS Max DD</th></tr>"""
                for w in wf_results['windows']:
                    oos_ret_class = 'positive' if w['test_return_pct'] > 0 else 'negative'
                    html += f"""<tr>
                        <td>{w['window_id']}</td>
                        <td>{w.get('test_start', '')}</td>
                        <td>{w['best_strategy'].upper()}</td>
                        <td>{w['train_sharpe']:.3f}</td>
                        <td>{w['test_sharpe']:.3f}</td>
                        <td class="{oos_ret_class}">{w['test_return_pct']:+.2f}%</td>
                        <td>{w['test_max_dd_pct']:.2f}%</td>
                    </tr>"""
                html += "</table>"
            
            html += "</div>"
        
        # ---- MONTE CARLO ----
        if mc_results and 'n_simulations' in mc_results:
            mc = mc_results
            html += f"""
<div class="section">
    <h2>🎲 Monte Carlo Simülasyonu ({mc['n_simulations']:,} simülasyon)</h2>
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Kâr Olasılığı</div>
            <div class="kpi-value positive">{mc.get('prob_profit', 0):.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">2x Olasılığı</div>
            <div class="kpi-value neutral">{mc.get('prob_double', 0):.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Yarılanma Riski</div>
            <div class="kpi-value negative">{mc.get('prob_halve', 0):.1f}%</div>
        </div>
    </div>
    <div class="two-col" style="margin-top: 15px;">
        <div>
            <h3 style="color:#aaa; margin-bottom:10px;">Final Değer Dağılımı</h3>
            <div class="stat-row"><span class="stat-label">Ortalama</span><span class="stat-value">{mc.get('final_value_mean', 0):,.0f} TL</span></div>
            <div class="stat-row"><span class="stat-label">Medyan</span><span class="stat-value">{mc.get('final_value_median', 0):,.0f} TL</span></div>
            <div class="stat-row"><span class="stat-label">{mc.get('confidence', 0.95)*100:.0f}% CI Alt</span><span class="stat-value">{mc.get('final_value_ci_lower', 0):,.0f} TL</span></div>
            <div class="stat-row"><span class="stat-label">{mc.get('confidence', 0.95)*100:.0f}% CI Üst</span><span class="stat-value">{mc.get('final_value_ci_upper', 0):,.0f} TL</span></div>
            <div class="stat-row"><span class="stat-label">En Kötü</span><span class="stat-value negative">{mc.get('final_value_worst', 0):,.0f} TL</span></div>
            <div class="stat-row"><span class="stat-label">En İyi</span><span class="stat-value positive">{mc.get('final_value_best', 0):,.0f} TL</span></div>
        </div>
        <div>
            <h3 style="color:#aaa; margin-bottom:10px;">Risk Dağılımı</h3>
            <div class="stat-row"><span class="stat-label">Ort. Max DD</span><span class="stat-value">{mc.get('max_dd_mean_pct', 0):.2f}%</span></div>
            <div class="stat-row"><span class="stat-label">En Kötü Max DD</span><span class="stat-value negative">{mc.get('max_dd_worst_pct', 0):.2f}%</span></div>
            <div class="stat-row"><span class="stat-label">Sharpe Ortalaması</span><span class="stat-value">{mc.get('sharpe_mean', 0):.3f}</span></div>
            <div class="stat-row"><span class="stat-label">Sharpe Medyanı</span><span class="stat-value">{mc.get('sharpe_median', 0):.3f}</span></div>
            <div class="stat-row"><span class="stat-label">Sharpe CI</span><span class="stat-value">[{mc.get('sharpe_ci_lower', 0):.3f}, {mc.get('sharpe_ci_upper', 0):.3f}]</span></div>
        </div>
    </div>
</div>
"""
        
        # ---- RATING / DEĞERLENDİRME ----
        score, grade, color, explanation = ReportGenerator._calculate_grade(r, wf_results, mc_results)
        
        html += f"""
<div class="section">
    <h2>📋 Genel Değerlendirme</h2>
    <div class="rating-box" style="background: linear-gradient(135deg, {color}22, {color}11); border: 2px solid {color};">
        <div class="rating-score" style="color: {color};">{grade}</div>
        <div class="rating-label" style="color: {color};">Skor: {score}/100</div>
    </div>
    <div style="margin-top: 20px; padding: 15px; background: #16213e; border-radius: 10px;">
        {explanation}
    </div>
</div>
"""
        
        # ---- FOOTER ----
        html += f"""
<div class="footer">
    <p>OmenQuant v3 - Professional Trading System | Ömer Faruk Şafakoğlu</p>
    <p>Yıldız Teknik Üniversitesi, İstatistik Bölümü, 2025</p>
    <p style="margin-top: 8px; color: #444;">
        ⚠️ Bu rapor eğitim amaçlıdır. Yatırım tavsiyesi değildir.
        Rapor tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </p>
</div>
"""
        
        html += """</div></body></html>"""
        
        # Dosyaya yaz
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
    @staticmethod
    def _build_charts(r: Dict) -> str:
        """Plotly grafikleri için JavaScript kodu üret"""
        
        ohlcv = r.get('_ohlcv')
        trades_df = r.get('_trades_df', pd.DataFrame())
        ph = r.get('_portfolio_history')
        dd_series = r.get('_drawdown_series')
        
        if ohlcv is None or ph is None:
            return ""
        
        # ---- Veri hazırlığı ----
        # OHLCV
        dates_ohlc = [d.strftime('%Y-%m-%d') for d in ohlcv.index]
        opens = ohlcv['Open'].tolist()
        highs = ohlcv['High'].tolist()
        lows = ohlcv['Low'].tolist()
        closes = ohlcv['Close'].tolist()
        
        # SMA50
        sma50 = ohlcv['Close'].rolling(50).mean()
        sma50_vals = [round(v, 2) if not pd.isna(v) else 'null' for v in sma50.tolist()]
        
        # Portfolio history
        ph_dates = [d.strftime('%Y-%m-%d') for d in ph.index]
        ph_vals = [round(v, 2) for v in ph['portfolio_value'].tolist()]
        
        # Buy & Hold equity
        initial = r.get('initial_capital', 100000)
        first_close = closes[0] if closes else 1
        bh_vals = [round(initial * (c / first_close), 2) for c in closes]
        
        # Drawdown
        if dd_series is not None and len(dd_series) > 0:
            dd_dates = [d.strftime('%Y-%m-%d') for d in dd_series.index]
            dd_vals = [round(v * 100, 2) for v in dd_series.tolist()]
        else:
            dd_dates = []
            dd_vals = []
        
        # Trade sinyalleri (AL/SAT okları)
        buy_dates = []
        buy_prices = []
        buy_texts = []
        sell_dates = []
        sell_prices = []
        sell_texts = []
        sell_colors = []
        
        if not trades_df.empty:
            for _, trade in trades_df.iterrows():
                # Entry (AL)
                ed = trade.get('entry_date')
                if ed is not None:
                    buy_dates.append(ed.strftime('%Y-%m-%d') if hasattr(ed, 'strftime') else str(ed))
                    buy_prices.append(round(float(trade['entry_price']), 2))
                    pnl = trade.get('net_pnl', 0)
                    ret = trade.get('return_pct', 0)
                    buy_texts.append(f"AL #{int(trade.get('trade_id', 0))}<br>Fiyat: {trade['entry_price']:.2f}")
                
                # Exit (SAT)
                xd = trade.get('exit_date')
                if xd is not None:
                    sell_dates.append(xd.strftime('%Y-%m-%d') if hasattr(xd, 'strftime') else str(xd))
                    sell_prices.append(round(float(trade['exit_price']), 2))
                    pnl = trade.get('net_pnl', 0)
                    ret = trade.get('return_pct', 0)
                    reason = trade.get('exit_reason', '')
                    emoji = {'SIGNAL': '📊', 'STOP_LOSS': '🛑', 'TAKE_PROFIT': '🎯', 
                             'TRAILING_STOP': '📉', 'END_OF_DATA': '🏁'}.get(reason, '❓')
                    color = '#38ef7d' if pnl > 0 else '#f45c43'
                    sell_texts.append(f"SAT #{int(trade.get('trade_id', 0))} {emoji}<br>Fiyat: {trade['exit_price']:.2f}<br>P&L: {pnl:+,.0f} TL ({ret:+.1f}%)<br>Sebep: {reason}")
                    sell_colors.append(color)
        
        # JSON-safe conversion
        import json as _json
        
        chart_js = f"""
<div class="section">
    <h2>📊 Fiyat Grafiği & İşlem Sinyalleri</h2>
    <div id="priceChart" style="width:100%; height:500px;"></div>
</div>

<div class="two-col">
    <div class="section">
        <h2>💰 Portföy Değeri vs Buy & Hold</h2>
        <div id="equityChart" style="width:100%; height:400px;"></div>
    </div>
    <div class="section">
        <h2>📉 Drawdown</h2>
        <div id="ddChart" style="width:100%; height:400px;"></div>
    </div>
</div>

<script>
(function() {{
    var darkLayout = {{
        paper_bgcolor: '#1a1a2e',
        plot_bgcolor: '#0a0a1a',
        font: {{ color: '#e0e0e0', family: 'Segoe UI, system-ui, sans-serif' }},
        xaxis: {{ gridcolor: '#222', linecolor: '#333', rangeslider: {{ visible: false }} }},
        yaxis: {{ gridcolor: '#222', linecolor: '#333' }},
        margin: {{ l: 60, r: 30, t: 30, b: 40 }},
        legend: {{ bgcolor: 'rgba(26,26,46,0.8)', bordercolor: '#333', borderwidth: 1, 
                   font: {{ size: 11 }} }},
        hovermode: 'x unified'
    }};
    
    // ===== 1) FİYAT GRAFİĞİ + OKLAR =====
    var candlestick = {{
        x: {_json.dumps(dates_ohlc)},
        open: {_json.dumps(opens)},
        high: {_json.dumps(highs)},
        low: {_json.dumps(lows)},
        close: {_json.dumps(closes)},
        type: 'candlestick',
        name: '{r.get("ticker", "").replace(".IS", "")}',
        increasing: {{ line: {{ color: '#38ef7d' }}, fillcolor: '#38ef7d' }},
        decreasing: {{ line: {{ color: '#f45c43' }}, fillcolor: '#f45c43' }}
    }};
    
    var sma50trace = {{
        x: {_json.dumps(dates_ohlc)},
        y: {_json.dumps(sma50_vals).replace('"null"', 'null')},
        type: 'scatter', mode: 'lines',
        name: 'SMA 50',
        line: {{ color: '#ffd700', width: 1.5, dash: 'dot' }},
        connectgaps: false
    }};
    
    var buyMarkers = {{
        x: {_json.dumps(buy_dates)},
        y: {_json.dumps(buy_prices)},
        text: {_json.dumps(buy_texts)},
        type: 'scatter', mode: 'markers',
        name: '🟢 AL',
        marker: {{
            symbol: 'triangle-up',
            size: 14,
            color: '#38ef7d',
            line: {{ color: '#fff', width: 1.5 }}
        }},
        hovertemplate: '%{{text}}<extra></extra>'
    }};
    
    var sellMarkers = {{
        x: {_json.dumps(sell_dates)},
        y: {_json.dumps(sell_prices)},
        text: {_json.dumps(sell_texts)},
        type: 'scatter', mode: 'markers',
        name: '🔴 SAT',
        marker: {{
            symbol: 'triangle-down',
            size: 14,
            color: {_json.dumps(sell_colors) if sell_colors else '["#f45c43"]'},
            line: {{ color: '#fff', width: 1.5 }}
        }},
        hovertemplate: '%{{text}}<extra></extra>'
    }};
    
    var priceLayout = JSON.parse(JSON.stringify(darkLayout));
    priceLayout.yaxis.title = 'Fiyat (TL)';
    priceLayout.legend.orientation = 'h';
    priceLayout.legend.y = 1.12;
    
    Plotly.newPlot('priceChart', [candlestick, sma50trace, buyMarkers, sellMarkers], priceLayout, {{responsive: true}});
    
    // ===== 2) EQUİTY EĞRİSİ =====
    var equityTrace = {{
        x: {_json.dumps(ph_dates)},
        y: {_json.dumps(ph_vals)},
        type: 'scatter', mode: 'lines',
        name: 'Strateji',
        line: {{ color: '#667eea', width: 2 }},
        fill: 'tozeroy',
        fillcolor: 'rgba(102,126,234,0.1)'
    }};
    
    var bhTrace = {{
        x: {_json.dumps(dates_ohlc)},
        y: {_json.dumps(bh_vals)},
        type: 'scatter', mode: 'lines',
        name: 'Buy & Hold',
        line: {{ color: '#ffd700', width: 1.5, dash: 'dash' }}
    }};
    
    var eqLayout = JSON.parse(JSON.stringify(darkLayout));
    eqLayout.yaxis.title = 'Değer (TL)';
    eqLayout.legend.orientation = 'h';
    eqLayout.legend.y = 1.12;
    
    Plotly.newPlot('equityChart', [equityTrace, bhTrace], eqLayout, {{responsive: true}});
    
    // ===== 3) DRAWDOWN =====
    var ddTrace = {{
        x: {_json.dumps(dd_dates)},
        y: {_json.dumps(dd_vals)},
        type: 'scatter', mode: 'lines',
        name: 'Drawdown',
        line: {{ color: '#f45c43', width: 1.5 }},
        fill: 'tozeroy',
        fillcolor: 'rgba(244,92,67,0.2)'
    }};
    
    var ddLayout = JSON.parse(JSON.stringify(darkLayout));
    ddLayout.yaxis.title = 'Drawdown (%)';
    ddLayout.legend.orientation = 'h';
    
    Plotly.newPlot('ddChart', [ddTrace], ddLayout, {{responsive: true}});
}})();
</script>
"""
        return chart_js
    
    @staticmethod
    def _calculate_grade(r: Dict, wf: Dict = None, mc: Dict = None) -> Tuple[int, str, str, str]:
        """Backtest sonuçlarını puanla"""
        score = 50  # Başlangıç puanı
        explanations = []
        
        # Getiri (+/- 15 puan)
        alpha = r.get('alpha_pct', 0)
        if alpha > 10:
            score += 15; explanations.append("✅ Mükemmel alpha: Buy & Hold'dan çok daha iyi")
        elif alpha > 0:
            score += 8; explanations.append("✅ Pozitif alpha: Buy & Hold'u geçiyor")
        else:
            score -= 10; explanations.append("❌ Negatif alpha: Buy & Hold daha iyi")
        
        # Sharpe (+/- 15 puan)
        sharpe = r.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            score += 15; explanations.append("✅ Üstün risk-getiri oranı (Sharpe > 1.5)")
        elif sharpe > 0.75:
            score += 10; explanations.append("✅ İyi risk-getiri oranı (Sharpe > 0.75)")
        elif sharpe > 0:
            score += 3; explanations.append("⚠️ Orta risk-getiri oranı")
        else:
            score -= 10; explanations.append("❌ Negatif Sharpe: Risk karşılığında getiri yetersiz")
        
        # Max DD (+/- 10 puan)
        max_dd = abs(r.get('max_drawdown_pct', 0))
        if max_dd < 10:
            score += 10; explanations.append("✅ Düşük drawdown (<%10)")
        elif max_dd < 20:
            score += 5; explanations.append("⚠️ Kabul edilebilir drawdown (%10-20)")
        elif max_dd < 30:
            score -= 3; explanations.append("⚠️ Yüksek drawdown (%20-30)")
        else:
            score -= 10; explanations.append("❌ Çok yüksek drawdown (>%30)")
        
        # Win Rate (+/- 5 puan)
        wr = r.get('win_rate_pct', 0)
        if wr > 60:
            score += 5; explanations.append("✅ Yüksek win rate (>%60)")
        elif wr > 50:
            score += 2
        elif wr < 40:
            score -= 5; explanations.append("❌ Düşük win rate (<%40)")
        
        # Profit Factor (+/- 5 puan)
        pf = r.get('profit_factor', 0)
        if pf > 2:
            score += 5; explanations.append("✅ Güçlü profit factor (>2)")
        elif pf > 1.5:
            score += 3
        elif pf < 1:
            score -= 5; explanations.append("❌ Profit factor < 1 (zarar eden sistem)")
        
        # Walk-forward bonus
        if wf and wf.get('summary', {}).get('positive_windows_pct', 0) > 60:
            score += 5; explanations.append("✅ Walk-forward: OOS pencerelerinin çoğu pozitif")
        elif wf and wf.get('summary', {}).get('positive_windows_pct', 0) < 40:
            score -= 5; explanations.append("❌ Walk-forward: OOS pencerelerinin çoğu negatif")
        
        # Monte Carlo bonus
        if mc and mc.get('prob_profit', 0) > 70:
            score += 5; explanations.append(f"✅ Monte Carlo: %{mc['prob_profit']:.0f} kâr olasılığı")
        elif mc and mc.get('prob_profit', 0) < 50:
            score -= 5; explanations.append(f"❌ Monte Carlo: %{mc['prob_profit']:.0f} kâr olasılığı (düşük)")
        
        # Sınırla
        score = max(0, min(100, score))
        
        # Grade
        if score >= 85:
            grade, color = "A+", "#38ef7d"
        elif score >= 75:
            grade, color = "A", "#38ef7d"
        elif score >= 65:
            grade, color = "B+", "#ffd700"
        elif score >= 55:
            grade, color = "B", "#ffd700"
        elif score >= 45:
            grade, color = "C", "#ff9f43"
        elif score >= 35:
            grade, color = "D", "#f45c43"
        else:
            grade, color = "F", "#f45c43"
        
        explanation_html = "<br>".join(explanations)
        
        return score, grade, color, explanation_html


# =============================================================================
# KONSOL RAPOR
# =============================================================================

class ConsoleReporter:
    """Terminal çıktısı için raporlayıcı"""
    
    @staticmethod
    def print_results(results: Dict):
        r = results
        
        print("\n" + "═" * 70)
        print("  ⚡ OMENQUANT BACKTEST SONUÇLARI")
        print("═" * 70)
        
        print(f"\n  📊 {r.get('ticker', 'N/A')} | {r.get('start_date', '')} → {r.get('end_date', '')}")
        print(f"     {r.get('trading_days', 0)} gün | {r.get('years', 0)} yıl")
        
        print(f"\n  💰 SERMAYE")
        print(f"     Başlangıç:    {r.get('initial_capital', 0):>15,.2f} TL")
        print(f"     Bitiş:        {r.get('final_value', 0):>15,.2f} TL")
        print(f"     Net Kâr:      {r.get('net_profit', 0):>+15,.2f} TL")
        
        print(f"\n  📈 GETİRİ")
        print(f"     Strateji:     {r.get('total_return_pct', 0):>+15.2f}%")
        print(f"     CAGR:         {r.get('cagr_pct', 0):>+15.2f}%")
        print(f"     Buy & Hold:   {r.get('buy_hold_return_pct', 0):>+15.2f}%")
        print(f"     Alpha:        {r.get('alpha_pct', 0):>+15.2f}%")
        
        print(f"\n  ⚠️  RİSK")
        print(f"     Sharpe:       {r.get('sharpe_ratio', 0):>15.3f}")
        print(f"     Sortino:      {r.get('sortino_ratio', 0):>15.3f}")
        print(f"     Calmar:       {r.get('calmar_ratio', 0):>15.3f}")
        print(f"     Max Drawdown: {r.get('max_drawdown_pct', 0):>15.2f}%")
        print(f"     Volatilite:   {r.get('annual_volatility_pct', 0):>15.2f}%")
        
        print(f"\n  🎯 İŞLEMLER")
        print(f"     Toplam:       {r.get('total_trades', 0):>15}")
        print(f"     Kazançlı:     {r.get('winning_trades', 0):>15}")
        print(f"     Zararlı:      {r.get('losing_trades', 0):>15}")
        print(f"     Win Rate:     {r.get('win_rate_pct', 0):>15.1f}%")
        print(f"     Profit Factor:{r.get('profit_factor', 0):>15.3f}")
        print(f"     Payoff Ratio: {r.get('payoff_ratio', 0):>15.3f}")
        print(f"     Expectancy:   {r.get('expectancy', 0):>+15.2f} TL")
        print(f"     Ort. Süre:    {r.get('avg_holding_days', 0):>15.1f} gün")
        
        print(f"\n  💸 MALİYET")
        print(f"     Komisyon:     {r.get('total_commission', 0):>15,.2f} TL")
        print(f"     Slippage:     {r.get('total_slippage', 0):>15,.2f} TL")
        
        print("\n" + "═" * 70)
        print("  📋 DEĞERLENDİRME")
        
        checks = [
            (r.get('alpha_pct', 0) > 0, "Alpha pozitif (Buy&Hold'u geçiyor)"),
            (r.get('sharpe_ratio', 0) > 0.5, "Sharpe > 0.5"),
            (r.get('sharpe_ratio', 0) > 1.0, "Sharpe > 1.0 (iyi)"),
            (abs(r.get('max_drawdown_pct', 0)) < 20, "Max DD < %20"),
            (r.get('win_rate_pct', 0) > 50, "Win Rate > %50"),
            (r.get('profit_factor', 0) > 1.5, "Profit Factor > 1.5"),
            (r.get('expectancy', 0) > 0, "Pozitif expectancy"),
        ]
        
        for passed, label in checks:
            icon = "  ✅" if passed else "  ❌"
            print(f"     {icon} {label}")
        
        print("═" * 70)


# =============================================================================
# ANA ÇALIŞTIRICI
# =============================================================================

def run_full_backtest(
    ticker: str = "THYAO.IS",
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    trade_start: str = None,
    trade_end: str = None,
    strategy: str = "combined",
    initial_capital: float = 100_000,
    run_walk_forward: bool = True,
    run_monte_carlo: bool = True,
    run_strategy_comparison: bool = True,
    generate_report: bool = True,
    report_path: str = None,
) -> Dict:
    """
    Tek fonksiyonla tam kapsamlı backtest.
    
    Train/Test Split Modu:
        start_date → end_date: Tüm veri (indikatör hesaplama + pattern)
        trade_start → trade_end: Sadece bu aralıkta işlem yap
        
        Örnek: 5 yıl veri çek (2021-2026), sadece son 1 yılda trade et (2025-2026)
        Bu sayede SMA50, RSI vb. indikatörler geçmiş veriyle doğru hesaplanır
        ama performans sadece trade penceresi üzerinden ölçülür.
    
    Args:
        ticker: BIST hisse kodu (örn: "THYAO.IS")
        start_date: Veri başlangıç tarihi (indikatörler için)
        end_date: Veri bitiş tarihi
        trade_start: İşlem başlangıç tarihi (None ise start_date kullanılır)
        trade_end: İşlem bitiş tarihi (None ise end_date kullanılır)
        strategy: Strateji adı
        initial_capital: Başlangıç sermayesi (TL)
    """
    if not YF_AVAILABLE:
        raise ImportError("yfinance yüklü değil: pip install yfinance")
    
    # Trade penceresi
    t_start = trade_start or start_date
    t_end = trade_end or end_date
    
    print(f"\n{'='*70}")
    print(f"  ⚡ OmenQuant v3 - Professional Backtesting Engine")
    print(f"  📊 {ticker} | Strateji: {strategy.upper()}")
    print(f"  📚 Veri Aralığı:  {start_date} → {end_date} (pattern öğrenme)")
    print(f"  🎯 Trade Aralığı: {t_start} → {t_end} (işlem yapma)")
    print(f"  💰 Sermaye: {initial_capital:,.0f} TL")
    print(f"{'='*70}")
    
    # Tüm veriyi çek (indikatör hesaplama için)
    print("\n📥 Veri çekiliyor...")
    full_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(full_data.columns, pd.MultiIndex):
        full_data.columns = full_data.columns.get_level_values(0)
    full_data = full_data.dropna()
    print(f"   ✅ {len(full_data)} gün veri çekildi ({full_data.index[0].strftime('%Y-%m-%d')} → {full_data.index[-1].strftime('%Y-%m-%d')})")
    
    if len(full_data) < 60:
        raise ValueError(f"Yetersiz veri: {len(full_data)} gün (min 60 gün gerekli)")
    
    # İndikatörleri TÜM veri üzerinde hesapla (SMA50 vb. geçmişe ihtiyaç duyar)
    print("📐 İndikatörler hesaplanıyor (tüm veri)...")
    full_data_with_indicators = TechnicalEngine.compute_indicators(full_data.copy())
    
    # Sinyalleri TÜM veri üzerinde üret (state-machine geçmişi görsün)
    print(f"📡 {strategy.upper()} sinyalleri üretiliyor (tüm veri)...")
    full_signals = TechnicalEngine.generate_signals(full_data, strategy)
    
    # Trade penceresine kes
    trade_data = full_data[t_start:t_end].copy()
    trade_signals = full_signals[t_start:t_end].copy()
    
    n_buys = (trade_signals == 1).sum()
    n_sells = (trade_signals == -1).sum()
    print(f"   📊 Trade penceresi: {len(trade_data)} gün")
    print(f"   📊 Sinyaller: {n_buys} AL, {n_sells} SAT, {(trade_signals == 0).sum()} BEKLE")
    
    if len(trade_data) < 10:
        raise ValueError(f"Trade penceresinde yetersiz veri: {len(trade_data)} gün")
    
    # Config
    config = BacktestConfig(
        initial_capital=initial_capital,
        risk_free_rate=0.45,
    )
    
    # 1) Ana Backtest — sadece trade penceresi üzerinde
    print("\n🔄 Ana backtest çalışıyor...")
    engine = BacktestEngine(config)
    results = engine.run(trade_data, trade_signals, ticker)
    ConsoleReporter.print_results(results)
    
    # 2) Strateji Karşılaştırma — trade penceresi üzerinde
    strategy_comp = None
    if run_strategy_comparison:
        print("\n🔄 Strateji karşılaştırması...")
        comparator = StrategyComparator(config)
        # Her strateji için sinyalleri full_data'dan üretip trade window'a kes
        strategies_list = ["rsi", "ma_crossover", "macd", "bollinger", "momentum", "trend_follow", "adaptive", "vol_filtered", "ml_enhanced", "combined"]
        rows = []
        for strat in strategies_list:
            strat_signals_full = TechnicalEngine.generate_signals(full_data, strat)
            strat_signals_window = strat_signals_full[t_start:t_end]
            strat_engine = BacktestEngine(config)
            strat_result = strat_engine.run(trade_data, strat_signals_window, ticker)
            rows.append({
                'Strateji': strat.upper(),
                'Getiri (%)': strat_result.get('total_return_pct', 0),
                'CAGR (%)': strat_result.get('cagr_pct', 0),
                'Sharpe': strat_result.get('sharpe_ratio', 0),
                'Sortino': strat_result.get('sortino_ratio', 0),
                'Calmar': strat_result.get('calmar_ratio', 0),
                'Max DD (%)': strat_result.get('max_drawdown_pct', 0),
                'Win Rate (%)': strat_result.get('win_rate_pct', 0),
                'Profit Factor': strat_result.get('profit_factor', 0),
                'İşlem Sayısı': strat_result.get('total_trades', 0),
                'Ort. İşlem Süresi': strat_result.get('avg_holding_days', 0),
                'Expectancy': strat_result.get('expectancy', 0),
                'Alpha (%)': strat_result.get('alpha_pct', 0),
            })
        strategy_comp = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
        print(strategy_comp.to_string(index=False))
    
    # 3) Walk-Forward — full data üzerinde
    wf_results = None
    if run_walk_forward and len(full_data) >= 400:
        print("\n🔄 Walk-Forward analiz çalışıyor (tüm veri)...")
        wfa = WalkForwardAnalyzer(config)
        wf_results = wfa.run(full_data, ticker=ticker)
        wf_summary = wf_results.get('summary', {})
        print(f"   Pencere: {wf_summary.get('total_windows', 0)}")
        print(f"   OOS Ort. Getiri: {wf_summary.get('avg_oos_return_pct', 0):+.2f}%")
        print(f"   OOS Ort. Sharpe: {wf_summary.get('avg_oos_sharpe', 0):.3f}")
        print(f"   WF Efficiency: {wf_summary.get('wf_efficiency', 0):.3f}")
        print(f"   Pozitif Pencere: {wf_summary.get('positive_windows_pct', 0):.1f}%")
    elif run_walk_forward:
        print("\n⚠️  Walk-Forward için yeterli veri yok (min ~400 gün)")
    
    # 4) Monte Carlo
    mc_results = None
    trades_df = results.get('_trades_df', pd.DataFrame())
    if run_monte_carlo and not trades_df.empty and len(trades_df) >= 5:
        print(f"\n🔄 Monte Carlo simülasyonu ({config.mc_simulations:,} simülasyon)...")
        mc = MonteCarloSimulator(config)
        mc_results = mc.run(trades_df)
        print(f"   Kâr Olasılığı: %{mc_results.get('prob_profit', 0):.1f}")
        print(f"   Medyan Değer: {mc_results.get('final_value_median', 0):,.0f} TL")
        print(f"   %95 CI: [{mc_results.get('final_value_ci_lower', 0):,.0f}, {mc_results.get('final_value_ci_upper', 0):,.0f}]")
    elif run_monte_carlo:
        print(f"\n⚠️  Monte Carlo için yeterli işlem yok ({len(trades_df)} işlem, min 5)")
    
    # 5) HTML Rapor
    report_file = None
    if generate_report:
        if report_path is None:
            ticker_clean = ticker.replace('.IS', '').lower()
            report_path = f"omenquant_backtest_{ticker_clean}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        
        print(f"\n📝 HTML rapor üretiliyor: {report_path}")
        report_file = ReportGenerator.generate_html_report(
            results, wf_results, mc_results, strategy_comp, report_path
        )
        print(f"   ✅ Rapor oluşturuldu: {report_file}")
    
    print(f"\n{'='*70}")
    print("  ✅ Backtest tamamlandı!")
    print(f"{'='*70}\n")
    
    return {
        'results': results,
        'strategy_comparison': strategy_comp,
        'walk_forward': wf_results,
        'monte_carlo': mc_results,
        'report_path': report_file,
    }


# =============================================================================
# TEST & DEMO
# =============================================================================

if __name__ == "__main__":
    # THYAO - 5 yıllık veriyle pattern öğren, son 1 yılda trade et
    all_results = run_full_backtest(
        ticker="THYAO.IS",
        start_date="2021-02-15",       # 5 yıllık veri başlangıcı (indikatörler için)
        end_date="2026-02-15",          # Veri sonu
        trade_start="2025-02-15",       # İşlem başlangıcı (son 1 yıl)
        trade_end="2026-02-15",         # İşlem sonu
        strategy="vol_filtered",
        initial_capital=100_000,
        run_walk_forward=True,
        run_monte_carlo=True,
        run_strategy_comparison=True,
        generate_report=True,
    )
