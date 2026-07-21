"""
OmenQuant v3 - Backtest Engine
===============================
Production-grade backtesting with:
- Signal generation on historical data
- Trade simulation (entry/exit)
- Risk management (stop-loss, take-profit)
- Performance metrics (Sharpe, Sortino, MDD, etc.)
- Walk-forward validation
- Trade logging

Ömer Faruk Şafakoğlu - OmenQuant Thesis Project
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# TensorFlow (opsiyonel, gerekirse LSTM modeli kullanıyor)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False


class BacktestEngine:
    """
    OmenQuant sinyal sistemini backtest eder.
    
    Kullanım:
    ```python
    engine = BacktestEngine(
        ticker='THYAO.IS',
        start_date='2022-01-01',
        end_date='2024-01-01',
        initial_capital=10000,
        position_size_pct=0.5,  # Her pozisyon max 50%
        use_lstm=True
    )
    results = engine.run()
    engine.print_report()
    ```
    """
    
    def __init__(self, ticker='THYAO.IS', start_date=None, end_date=None,
                 initial_capital=10000, position_size_pct=0.5, use_lstm=True,
                 stop_loss_pct=0.05, take_profit_pct=0.06):
        
        self.ticker = ticker
        self.start_date = start_date or '2022-01-01'
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.use_lstm = use_lstm and TF_AVAILABLE
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Data
        self.df = None
        self.macro_df = None
        
        # Results
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.signals = []
        
        # State
        self.position = None  # {'type': 'LONG'/'SHORT', 'entry_price': X, 'entry_date': D, 'size': S}
        self.cash = initial_capital
        self.equity = initial_capital
        
    def fetch_data(self):
        """Yahoo Finance'ten veri çek"""
        print(f"📊 Veri çekiliyor: {self.ticker} ({self.start_date} - {self.end_date})")
        
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.dropna()
        self.df = df.copy()
        
        print(f"✅ {len(df)} gün veri yüklendi")
        return df
    
    def add_technical_indicators(self):
        """Teknik analiz göstergelerini ekle"""
        df = self.df.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['BB_Upper'] = sma + (std * 2)
        df['BB_Lower'] = sma - (std * 2)
        df['BB_Middle'] = sma
        
        # Volatility & Volume
        df['Volatility'] = df['Close'].pct_change().rolling(20).std()
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Change'] = (df['Volume'] / df['Volume_SMA'] - 1).clip(-1, 1)
        
        # ATR (untuk risk management)
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = abs(df['High'] - df['Close'].shift(1))
        df['Low_Close'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['TR'].rolling(14).mean()
        
        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        self.df = df.copy()
        return df
    
    def build_lstm_model(self, train_df):
        """LSTM modeli eğit (opsiyonel)"""
        if not self.use_lstm or not TF_AVAILABLE:
            return None
        
        try:
            time_steps = 3
            feature_cols = ['RSI', 'Volatility', 'Volume_Change', 'Close']
            
            # Prepare data
            scaler = MinMaxScaler()
            data = train_df[feature_cols].values
            data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
            scaled = scaler.fit_transform(data)
            
            X, y = [], []
            targets = train_df['Target'].values
            for i in range(time_steps, len(scaled)):
                X.append(scaled[i-time_steps:i])
                y.append(targets[i])
            
            if len(X) < 10:
                return None
            
            X, y = np.array(X), np.array(y)
            
            # Build & train
            model = Sequential([
                Input(shape=(time_steps, len(feature_cols))),
                LSTM(32, return_sequences=True),
                Dropout(0.2),
                LSTM(16),
                Dropout(0.2),
                Dense(8, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
            
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, 
                     verbose=0, callbacks=[early_stop])
            
            return (model, scaler, feature_cols, time_steps)
        except:
            return None
    
    def generate_signal(self, idx, lstm_data=None):
        """
        Improved signal generation with strong confirmation filters.
        Requires RSI + Trend alignment + MACD confirmation.
        Less trading, higher quality signals.
        """
        if idx < 50:
            return ('HOLD', 0.0, [])
        
        row = self.df.iloc[idx]
        prev_row = self.df.iloc[idx-1]
        reasons = []
        
        # FILTER 1: RSI Extreme (Must have strong RSI signal)
        rsi_signal = 0.0
        if pd.notna(row['RSI']):
            if row['RSI'] < 25:
                rsi_signal = 1.0
                reasons.append("✅ RSI STRONG oversold (<25)")
            elif row['RSI'] > 75:
                rsi_signal = -1.0
                reasons.append("❌ RSI STRONG overbought (>75)")
            elif row['RSI'] < 35:
                rsi_signal = 0.5
                reasons.append("✅ RSI oversold (<35)")
            elif row['RSI'] > 65:
                rsi_signal = -0.5
                reasons.append("❌ RSI overbought (>65)")
        
        # FILTER 2: Trend Confirmation (MA alignment - price > MA20 > MA50)
        trend_signal = 0.0
        if pd.notna(row['MA_5']) and pd.notna(row['MA_20']) and pd.notna(row['MA_50']):
            if row['Close'] > row['MA_20'] > row['MA_50'] and row['MA_5'] > row['MA_20']:
                trend_signal = 0.5
                reasons.append("✅ Strong uptrend alignment")
            elif row['Close'] < row['MA_20'] < row['MA_50'] and row['MA_5'] < row['MA_20']:
                trend_signal = -0.5
                reasons.append("❌ Strong downtrend alignment")
            elif row['Close'] > row['MA_20'] and row['MA_5'] > row['MA_20']:
                trend_signal = 0.3
                reasons.append("✅ Uptrend")
            elif row['Close'] < row['MA_20'] and row['MA_5'] < row['MA_20']:
                trend_signal = -0.3
                reasons.append("❌ Downtrend")
        
        # FILTER 3: MACD Momentum (Crossovers are strongest)
        macd_signal = 0.0
        if pd.notna(row['MACD']) and pd.notna(row['MACD_Signal']) and pd.notna(prev_row['MACD']) and pd.notna(prev_row['MACD_Signal']):
            macd_hist = row['MACD'] - row['MACD_Signal']
            prev_macd_hist = prev_row['MACD'] - prev_row['MACD_Signal']
            
            # Bullish crossover (MACD crosses above signal line)
            if prev_macd_hist < 0 and macd_hist > 0:
                macd_signal = 0.8
                reasons.append("✅ MACD bullish crossover (strong)")
            # Bearish crossover (MACD crosses below signal line)
            elif prev_macd_hist > 0 and macd_hist < 0:
                macd_signal = -0.8
                reasons.append("❌ MACD bearish crossover (strong)")
            elif row['MACD'] > row['MACD_Signal']:
                macd_signal = 0.3
                reasons.append("✅ MACD positive")
            else:
                macd_signal = -0.3
                reasons.append("❌ MACD negative")
        
        # FILTER 4: Volume Confirmation (Volume must support move)
        volume_signal = 0.0
        if pd.notna(row['Volume']) and pd.notna(row['Volume_SMA']):
            if row['Volume'] > row['Volume_SMA'] * 1.5:
                volume_signal = 0.2
                reasons.append("📊 Strong volume (1.5x average)")
            elif row['Volume'] > row['Volume_SMA'] * 1.2:
                volume_signal = 0.1
                reasons.append("📊 Volume above average")
            else:
                volume_signal = -0.1
        
        # FILTER 5: Bollinger Band Extremes (Mean reversion opportunity)
        bb_signal = 0.0
        if pd.notna(row['BB_Upper']) and pd.notna(row['BB_Lower']):
            bb_pos = (row['Close'] - row['BB_Lower']) / (row['BB_Upper'] - row['BB_Lower'] + 1e-10)
            if bb_pos < 0.1:
                bb_signal = 0.3
                reasons.append("🎯 BB extreme (lower band)")
            elif bb_pos > 0.9:
                bb_signal = -0.3
                reasons.append("🎯 BB extreme (upper band)")
        
        # DECISION: Require RSI + Trend Agreement
        # Only take trades if RSI and Trend signals agree (both bullish or both bearish)
        if rsi_signal > 0 and trend_signal > 0:
            # BULLISH: Both RSI and Trend are bullish
            strength = min(rsi_signal, trend_signal)  # Use weaker signal
            support = macd_signal + volume_signal + bb_signal
            total_score = strength + (support * 0.5)
            
        elif rsi_signal < 0 and trend_signal < 0:
            # BEARISH: Both RSI and Trend are bearish
            strength = max(rsi_signal, trend_signal)  # Use weaker signal
            support = macd_signal + volume_signal + bb_signal
            total_score = strength + (support * 0.5)
            
        else:
            # Conflicting signals - HOLD
            total_score = 0.0
            if rsi_signal != 0 and trend_signal != 0:
                reasons.append("⚠️ Conflicting signals (RSI vs Trend) → HOLD")
        
        # Generate signal with quality threshold
        if total_score > 0.5:  # Higher threshold = fewer but better trades
            signal = 'LONG'
        elif total_score < -0.5:
            signal = 'SHORT'
        else:
            signal = 'HOLD'
        
        return (signal, total_score, reasons)
    
    def process_position(self, idx):
        """
        Position management:
        - Check stop-loss/take-profit
        - Close position eğer gerekiyorsa
        """
        if self.position is None:
            return None
        
        row = self.df.iloc[idx]
        entry_price = self.position['entry_price']
        position_type = self.position['type']
        position_size = self.position['size']
        entry_date = self.position['entry_date']
        
        # Exit logic
        exit_price = None
        exit_reason = None
        
        if position_type == 'LONG':
            pnl_pct = (row['Close'] - entry_price) / entry_price
            
            if pnl_pct <= -self.stop_loss_pct:
                exit_price = entry_price * (1 - self.stop_loss_pct)
                exit_reason = 'STOP_LOSS'
            elif pnl_pct >= self.take_profit_pct:
                exit_price = entry_price * (1 + self.take_profit_pct)
                exit_reason = 'TAKE_PROFIT'
        
        elif position_type == 'SHORT':
            pnl_pct = (entry_price - row['Close']) / entry_price
            
            if pnl_pct <= -self.stop_loss_pct:
                exit_price = entry_price * (1 + self.stop_loss_pct)
                exit_reason = 'STOP_LOSS'
            elif pnl_pct >= self.take_profit_pct:
                exit_price = entry_price * (1 - self.take_profit_pct)
                exit_reason = 'TAKE_PROFIT'
        
        if exit_price is not None:
            self._close_position(idx, exit_price, exit_reason)
            return exit_reason
        
        return None
    
    def _close_position(self, idx, exit_price, exit_reason):
        """Pozisyonu kapat"""
        row = self.df.iloc[idx]
        entry_price = self.position['entry_price']
        position_size = self.position['size']
        position_type = self.position['type']
        entry_date = self.position['entry_date']
        
        if position_type == 'LONG':
            pnl = position_size * (exit_price - entry_price)
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # SHORT
            pnl = position_size * (entry_price - exit_price)
            pnl_pct = (entry_price - exit_price) / entry_price
        
        # Update equity
        self.cash += pnl
        self.equity = self.cash
        
        # Log trade
        trade = {
            'entry_date': entry_date,
            'exit_date': row.name,
            'type': position_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': position_size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'holding_days': (row.name - entry_date).days
        }
        self.trades.append(trade)
        
        self.position = None
    
    def _open_position(self, idx, signal_type):
        """Yeni pozisyon aç - improved position sizing"""
        if self.position is not None:
            return
        
        row = self.df.iloc[idx]
        
        # Dynamic position sizing based on volatility (ATR)
        atr = row['ATR'] if pd.notna(row['ATR']) else row['Close'] * 0.02
        risk_amount = self.cash * 0.02  # Risk 2% per trade
        position_size = risk_amount / (atr + 1e-10)
        
        # Cap position size
        max_position_value = self.cash * self.position_size_pct
        position_quantity = min(position_size, max_position_value / row['Close'])
        
        self.position = {
            'type': signal_type,
            'entry_price': row['Close'],
            'entry_date': row.name,
            'size': position_quantity,
            'entry_idx': idx,
            'atr': atr
        }
    
    def run(self):
        """Ana backtest loop'u çalıştır"""
        print(f"\n{'='*60}")
        print(f"🚀 BACKTEST BAŞLANIYOR")
        print(f"{'='*60}")
        print(f"Sembol: {self.ticker}")
        print(f"Tarih: {self.start_date} - {self.end_date}")
        print(f"İlk Sermaye: ${self.initial_capital:,.2f}")
        print(f"Position Size: {self.position_size_pct*100:.0f}%")
        print(f"Stop-Loss: {self.stop_loss_pct*100:.1f}%")
        print(f"Take-Profit: {self.take_profit_pct*100:.1f}%")
        
        # Fetch & prepare data
        self.fetch_data()
        self.add_technical_indicators()
        
        # Train-test split (80-20)
        split_idx = int(len(self.df) * 0.8)
        train_df = self.df.iloc[:split_idx]
        
        # Build LSTM (opsiyonel)
        lstm_data = None
        if self.use_lstm:
            print("🧠 LSTM modeli eğitiliyor...")
            lstm_data = self.build_lstm_model(train_df)
            if lstm_data:
                print("✅ LSTM başarıyla eğitildi")
            else:
                print("⚠️ LSTM eğitimi başarısız, sadece teknik analiz kullanılacak")
        
        # Backtest loop
        print(f"\n🔄 {len(self.df)} gün için backtest yapılıyor...")
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            
            # Process active position
            if self.position is not None:
                self.process_position(idx)
            
            # Generate signal
            signal, score, reasons = self.generate_signal(idx, lstm_data)
            self.signals.append({
                'date': row.name,
                'signal': signal,
                'score': score,
                'price': row['Close']
            })
            
            # Open new position
            if self.position is None and signal in ['LONG', 'SHORT']:
                self._open_position(idx, signal)
            
            # Record equity
            self.equity_curve.append({
                'date': row.name,
                'equity': self.equity,
                'price': row['Close']
            })
            
            # Daily return
            if len(self.equity_curve) > 1:
                prev_equity = self.equity_curve[-2]['equity']
                ret = (self.equity - prev_equity) / prev_equity
                self.daily_returns.append(ret)
        
        # Close remaining position
        if self.position is not None:
            last_price = self.df.iloc[-1]['Close']
            self._close_position(len(self.df)-1, last_price, 'END_OF_BACKTEST')
        
        print(f"✅ Backtest tamamlandı")
        self._calculate_metrics()
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'signals': self.signals,
            'metrics': self.metrics
        }
    
    def _calculate_metrics(self):
        """Performance metrics'leri hesapla"""
        equity_series = pd.Series(
            [e['equity'] for e in self.equity_curve],
            index=[e['date'] for e in self.equity_curve]
        )
        
        # Basic metrics
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        num_trades = len(self.trades)
        
        if num_trades == 0:
            num_trades = 1  # Avoid division by zero
        
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / 
                           (sum(t['pnl'] for t in losing_trades) + 1e-10)) if losing_trades else float('inf')
        
        # Returns
        daily_rets = np.array(self.daily_returns)
        annual_return = (self.equity / self.initial_capital) ** (252 / len(self.equity_curve)) - 1
        
        # Sharpe Ratio
        if len(daily_rets) > 0 and np.std(daily_rets) > 0:
            sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Sortino Ratio (downside volatility)
        downside_rets = daily_rets[daily_rets < 0]
        if len(downside_rets) > 0:
            sortino = np.mean(daily_rets) / np.std(downside_rets) * np.sqrt(252)
        else:
            sortino = sharpe
        
        # Max Drawdown
        cumulative = (1 + equity_series.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Calmar Ratio
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # Payoff Ratio
        payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Recovery Factor
        equity_list = [e['equity'] for e in self.equity_curve]
        max_loss = max([abs(e - self.initial_capital) for e in equity_list])
        recovery = total_return / (max_loss / self.initial_capital) if max_loss > 0 else 0
        
        self.metrics = {
            'total_return_pct': total_return * 100,
            'annual_return_pct': annual_return * 100,
            'num_trades': num_trades,
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': max_dd * 100,
            'calmar_ratio': calmar,
            'payoff_ratio': payoff,
            'recovery_factor': recovery,
            'final_equity': self.equity,
            'buy_hold_return_pct': ((self.df.iloc[-1]['Close'] / self.df.iloc[0]['Close']) - 1) * 100
        }
    
    def print_report(self):
        """Formatted backtest raporu"""
        print(f"\n{'='*70}")
        print(f"📊 BACKTEST SONUÇLARI - {self.ticker}")
        print(f"{'='*70}\n")
        
        m = self.metrics
        
        # Performance
        print("💰 PERFORMANCE")
        print(f"  İlk Sermaye:        ${self.initial_capital:>14,.2f}")
        print(f"  Son Sermaye:        ${m['final_equity']:>14,.2f}")
        print(f"  Toplam Kazanç:      {m['total_return_pct']:>13.2f}%")
        print(f"  Buy & Hold:         {m['buy_hold_return_pct']:>13.2f}%")
        print(f"  Yıllık Return:      {m['annual_return_pct']:>13.2f}%")
        
        # Risk
        print("\n⚠️ RİSK METRİKLERİ")
        print(f"  Max Drawdown:       {m['max_drawdown_pct']:>13.2f}%")
        print(f"  Sharpe Ratio:       {m['sharpe_ratio']:>14.3f}")
        print(f"  Sortino Ratio:      {m['sortino_ratio']:>14.3f}")
        print(f"  Calmar Ratio:       {m['calmar_ratio']:>14.3f}")
        
        # Trading
        print("\n📈 TRADİNG İSTATİSTİKLERİ")
        print(f"  Trade Sayısı:       {m['num_trades']:>14}")
        print(f"  Kazanan %:          {m['win_rate_pct']:>13.1f}%")
        print(f"  Ort. Kazanç:        ${m['avg_win']:>13,.2f}")
        print(f"  Ort. Kayıp:         ${m['avg_loss']:>13,.2f}")
        print(f"  Profit Factor:      {m['profit_factor']:>14.2f}")
        print(f"  Payoff Ratio:       {m['payoff_ratio']:>14.2f}")
        print(f"  Recovery Factor:    {m['recovery_factor']:>14.2f}")
        
        # Top trades
        if self.trades:
            print("\n🏆 EN İYİ TRADES")
            sorted_trades = sorted(self.trades, key=lambda x: x['pnl'], reverse=True)
            for i, trade in enumerate(sorted_trades[:5]):
                print(f"  {i+1}. {trade['type']:5} | " + 
                      f"{trade['entry_date'].strftime('%Y-%m-%d')} → " +
                      f"{trade['exit_date'].strftime('%Y-%m-%d')} | " +
                      f"PnL: ${trade['pnl']:>10,.2f} ({trade['pnl_pct']:>6.2f}%)")
            
            print("\n🔴 EN KÖTÜ TRADES")
            for i, trade in enumerate(sorted_trades[-5:]):
                print(f"  {i+1}. {trade['type']:5} | " + 
                      f"{trade['entry_date'].strftime('%Y-%m-%d')} → " +
                      f"{trade['exit_date'].strftime('%Y-%m-%d')} | " +
                      f"PnL: ${trade['pnl']:>10,.2f} ({trade['pnl_pct']:>6.2f}%)")
        
        print(f"\n{'='*70}\n")
    
    def export_results(self, output_dir='backtest_results'):
        """Sonuçları dosyaya kaydet"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Trades CSV
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(f'{output_dir}/{self.ticker}_trades.csv', index=False)
        
        # Equity curve CSV
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv(f'{output_dir}/{self.ticker}_equity.csv', index=False)
        
        # Signals CSV
        signals_df = pd.DataFrame(self.signals)
        signals_df.to_csv(f'{output_dir}/{self.ticker}_signals.csv', index=False)
        
        # Metrics JSON
        with open(f'{output_dir}/{self.ticker}_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"✅ Sonuçlar {output_dir}/ klasörüne kaydedildi")


def main():
    """Example usage"""
    # THYAO backtest
    engine = BacktestEngine(
        ticker='THYAO.IS',
        start_date='2022-01-01',
        end_date='2024-01-01',
        initial_capital=10000,
        position_size_pct=0.5,
        use_lstm=True,
        stop_loss_pct=0.05,
        take_profit_pct=0.06
    )
    
    results = engine.run()
    engine.print_report()
    engine.export_results()
    
    # Multiple tickers
    tickers = ['THYAO.IS', 'GARAN.IS', 'AKBNK.IS', 'EREGL.IS']
    for ticker in tickers:
        engine = BacktestEngine(ticker=ticker, start_date='2023-01-01', end_date='2024-01-01')
        engine.run()
        print(f"\n{ticker}: {engine.metrics['total_return_pct']:.2f}% return")


if __name__ == '__main__':
    main()
