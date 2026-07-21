"""
Debug script - Check why signals aren't being generated
"""

from omenquant_backtest import BacktestEngine
import pandas as pd

# Create engine
engine = BacktestEngine(
    ticker='THYAO.IS',
    start_date='2022-01-01',
    end_date='2024-12-31'
)

# Fetch and prepare data
print("📊 Veri çekiliyor...")
engine.fetch_data()
print(f"✅ {len(engine.df)} gün veri yüklendi\n")

# Add indicators
print("📈 Teknik göstergeler ekleniyor...")
engine.add_technical_indicators()

# Check data quality
print("📋 VERİ KALİTESİ:")
print(f"  Toplam satır: {len(engine.df)}")
print(f"  NaN değerler (Close): {engine.df['Close'].isna().sum()}")
print(f"  NaN değerler (RSI): {engine.df['RSI'].isna().sum()}")
print(f"  NaN değerler (MA_20): {engine.df['MA_20'].isna().sum()}")

print("\n📊 SON 5 GÜNÜN VERİSİ:")
print(engine.df[['Close', 'RSI', 'MA_5', 'MA_20', 'MACD', 'MACD_Signal']].tail(5))

# Test signal generation on last 10 days
print("\n🎯 SON 10 GÜNÜN SİNYALLERİ:")
print(f"{'Tarih':<12} {'Close':<10} {'RSI':<6} {'Signal':<8} {'Score':<8}")
print("-" * 50)

for idx in range(len(engine.df) - 10, len(engine.df)):
    row = engine.df.iloc[idx]
    signal, score, reasons = engine.generate_signal(idx, None)
    
    print(f"{str(row.name.date()):<12} {row['Close']:<10.2f} {row['RSI']:<6.1f} {signal:<8} {score:<8.2f}")
    
    if len(reasons) > 0:
        for reason in reasons:
            print(f"  → {reason}")

# Check if any trades would be generated
print("\n" + "="*50)
print("SIGNAL İSTATİSTİKLERİ:")
long_signals = 0
short_signals = 0
hold_signals = 0

for idx in range(50, len(engine.df)):  # Start from idx 50
    signal, score, _ = engine.generate_signal(idx, None)
    if signal == 'LONG':
        long_signals += 1
    elif signal == 'SHORT':
        short_signals += 1
    else:
        hold_signals += 1

print(f"  LONG sinyaller: {long_signals}")
print(f"  SHORT sinyaller: {short_signals}")
print(f"  HOLD sinyaller: {hold_signals}")
print(f"  Toplam: {long_signals + short_signals + hold_signals}")

if long_signals + short_signals == 0:
    print("\n⚠️ SORUN: Hiç trade sinyali üretilmiyor!")
    print("\nOlası sebepler:")
    print("  1. RSI hiçbir zaman <30 veya >70 olmayabiliyor")
    print("  2. Moving average'lar yeterince kesişmiyor")
    print("  3. MACD sinyali stabil olmayabiliyor")
    print("\nÇözüm: Parametre eşiklerini düşür (RSI 40/60, etc.)")
else:
    print("\n✅ Sinyaller üretiliyor, sorun pozisyon açmada olabilir")
