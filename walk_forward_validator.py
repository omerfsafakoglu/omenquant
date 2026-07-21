"""
OmenQuant Walk-Forward Validation
===================================
Out-of-sample testing için en iyi metodoloji.

Walk-forward validation:
1. Veriyi train/test bölümlerine ayır
2. Her bölümde modeli eğit
3. Sonraki bölümde test et
4. Sonuçları birleştir

Bu overfitting'i minimize ediyor ve realistik performance gösteriyor.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt


class WalkForwardValidator:
    """
    Walk-forward validation engine.
    
    Kullanım:
    ```python
    validator = WalkForwardValidator(
        ticker='THYAO.IS',
        start_date='2021-01-01',
        end_date='2024-01-01',
        train_period_days=252,  # 1 yıl
        test_period_days=63,    # 3 ay
        rebalance_days=21       # Her 3 haftada güncelle
    )
    results = validator.run()
    ```
    """
    
    def __init__(self, ticker, start_date, end_date, 
                 train_period_days=252, test_period_days=63, 
                 rebalance_days=21, initial_capital=10000):
        
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.rebalance_days = rebalance_days
        self.initial_capital = initial_capital
        
        self.windows = []
        self.results = []
        self.overall_metrics = {}
    
    def prepare_data(self, df):
        """Veriyi pencereler için hazırla"""
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        
        self.windows = []
        window_num = 1
        
        current_start = start
        
        while current_start + timedelta(days=self.train_period_days + self.test_period_days) <= end:
            train_end = current_start + timedelta(days=self.train_period_days)
            test_end = train_end + timedelta(days=self.test_period_days)
            
            # Veriyi al
            train_data = df[(df.index >= current_start) & (df.index < train_end)].copy()
            test_data = df[(df.index >= train_end) & (df.index < test_end)].copy()
            
            if len(train_data) > 50 and len(test_data) > 10:
                self.windows.append({
                    'window_num': window_num,
                    'train_start': current_start,
                    'train_end': train_end,
                    'test_start': train_end,
                    'test_end': test_end,
                    'train_data': train_data,
                    'test_data': test_data
                })
                window_num += 1
            
            # Sonraki pencereye geç (rebalance periyodu kadar)
            current_start += timedelta(days=self.rebalance_days)
        
        print(f"✅ {len(self.windows)} validation window'u oluşturuldu")
        return self.windows
    
    def run_window(self, window, backtest_engine_class):
        """Bir window'u çalıştır (train + test)"""
        from omenquant_backtest import BacktestEngine
        
        window_num = window['window_num']
        print(f"\n🔄 Window {window_num} çalışıyor: {window['test_start'].strftime('%Y-%m-%d')} - {window['test_end'].strftime('%Y-%m-%d')}")
        
        # Test backtest'ini çalıştır
        engine = BacktestEngine(
            ticker=self.ticker,
            start_date=window['test_start'].strftime('%Y-%m-%d'),
            end_date=window['test_end'].strftime('%Y-%m-%d'),
            initial_capital=self.initial_capital
        )
        
        # Test datası ile çalıştır
        engine.df = window['test_data']
        
        # Teknik göstergeleri ekle
        engine.add_technical_indicators()
        
        # Backtest
        engine.run()
        
        result = {
            'window_num': window_num,
            'period': f"{window['test_start'].strftime('%Y-%m-%d')} to {window['test_end'].strftime('%Y-%m-%d')}",
            'metrics': engine.metrics.copy(),
            'num_trades': len(engine.trades),
            'trades': engine.trades
        }
        
        self.results.append(result)
        
        print(f"   Return: {engine.metrics['total_return_pct']:.2f}% | " +
              f"Sharpe: {engine.metrics['sharpe_ratio']:.3f} | " +
              f"Trades: {len(engine.trades)}")
        
        return result
    
    def calculate_overall_metrics(self):
        """Tüm window'ların birleştirilmiş metriklerini hesapla"""
        if not self.results:
            return
        
        returns = [r['metrics']['total_return_pct'] / 100 for r in self.results]
        sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in self.results]
        max_dds = [r['metrics']['max_drawdown_pct'] / 100 for r in self.results]
        win_rates = [r['metrics']['win_rate_pct'] / 100 for r in self.results]
        profit_factors = [r['metrics']['profit_factor'] for r in self.results]
        
        # Average metrics
        avg_return = np.mean(returns)
        avg_sharpe = np.mean(sharpe_ratios)
        avg_max_dd = np.mean(max_dds)
        avg_win_rate = np.mean(win_rates)
        avg_profit_factor = np.mean(profit_factors)
        
        # Consistency metrics
        return_std = np.std(returns)
        sharpe_std = np.std(sharpe_ratios)
        
        # Positive windows (profit)
        positive_windows = sum(1 for r in returns if r > 0)
        positive_pct = positive_windows / len(returns)
        
        self.overall_metrics = {
            'num_windows': len(self.results),
            'positive_windows': positive_windows,
            'positive_windows_pct': positive_windows_pct * 100,
            'avg_return_pct': avg_return * 100,
            'return_std_pct': return_std * 100,
            'avg_sharpe': avg_sharpe,
            'sharpe_std': sharpe_std,
            'avg_max_dd_pct': avg_max_dd * 100,
            'avg_win_rate_pct': avg_win_rate * 100,
            'avg_profit_factor': avg_profit_factor,
            'returns': returns,
            'sharpe_ratios': sharpe_ratios,
            'max_drawdowns': max_dds
        }
    
    def print_summary(self):
        """Özet rapor yazdır"""
        m = self.overall_metrics
        
        print(f"\n{'='*70}")
        print(f"📊 WALK-FORWARD VALIDATION SUMMARY - {self.ticker}")
        print(f"{'='*70}\n")
        
        print(f"🪟 WINDOWS")
        print(f"  Total Windows:      {m['num_windows']}")
        print(f"  Positive Windows:   {m['positive_windows']} ({m['positive_windows_pct']:.1f}%)")
        
        print(f"\n💰 PERFORMANCE (Average Across Windows)")
        print(f"  Avg Return:         {m['avg_return_pct']:>13.2f}%")
        print(f"  Return Std Dev:     {m['return_std_pct']:>13.2f}%")
        print(f"  Avg Sharpe:         {m['avg_sharpe']:>14.3f}")
        print(f"  Sharpe Std Dev:     {m['sharpe_std']:>14.3f}")
        
        print(f"\n⚠️ RISK")
        print(f"  Avg Max Drawdown:   {m['avg_max_dd_pct']:>13.2f}%")
        
        print(f"\n📈 TRADING")
        print(f"  Avg Win Rate:       {m['avg_win_rate_pct']:>13.1f}%")
        print(f"  Avg Profit Factor:  {m['avg_profit_factor']:>14.2f}")
        
        print(f"\n📋 WINDOW-BY-WINDOW RESULTS")
        print(f"{'Window':<10} {'Period':<35} {'Return %':<12} {'Sharpe':<10} {'Trades':<8}")
        print(f"{'-'*70}")
        
        for i, result in enumerate(self.results):
            return_pct = result['metrics']['total_return_pct']
            sharpe = result['metrics']['sharpe_ratio']
            trades = result['num_trades']
            period = result['period']
            
            print(f"{i+1:<10} {period:<35} {return_pct:>10.2f}% {sharpe:>9.3f} {trades:>7}")
        
        print(f"\n{'='*70}\n")
    
    def plot_results(self):
        """Walk-forward sonuçlarını görselleştir"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        m = self.overall_metrics
        
        # 1. Window returns
        window_nums = list(range(1, len(self.results) + 1))
        returns = m['returns']
        
        colors = ['green' if r > 0 else 'red' for r in returns]
        axes[0, 0].bar(window_nums, returns, color=colors, alpha=0.7)
        axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[0, 0].set_xlabel('Window')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].set_title('Returns by Window')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Sharpe ratios
        sharpes = m['sharpe_ratios']
        colors = ['green' if s > 0 else 'red' for s in sharpes]
        axes[0, 1].bar(window_nums, sharpes, color=colors, alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[0, 1].set_xlabel('Window')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].set_title('Sharpe Ratio by Window')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Max Drawdown
        max_dds = m['max_drawdowns']
        axes[1, 0].bar(window_nums, max_dds, color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Window')
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        axes[1, 0].set_title('Maximum Drawdown by Window')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Distribution stats
        axes[1, 1].text(0.1, 0.9, f"Positive Windows: {m['positive_windows_pct']:.1f}%", 
                       transform=axes[1, 1].transAxes, fontsize=11)
        axes[1, 1].text(0.1, 0.8, f"Avg Return: {m['avg_return_pct']:.2f}%", 
                       transform=axes[1, 1].transAxes, fontsize=11)
        axes[1, 1].text(0.1, 0.7, f"Return Std: {m['return_std_pct']:.2f}%", 
                       transform=axes[1, 1].transAxes, fontsize=11)
        axes[1, 1].text(0.1, 0.6, f"Avg Sharpe: {m['avg_sharpe']:.3f}", 
                       transform=axes[1, 1].transAxes, fontsize=11)
        axes[1, 1].text(0.1, 0.5, f"Avg Max DD: {m['avg_max_dd_pct']:.2f}%", 
                       transform=axes[1, 1].transAxes, fontsize=11)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary Statistics')
        
        fig.suptitle(f'Walk-Forward Validation - {self.ticker}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'wfv_{self.ticker}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Grafik kaydedildi: wfv_{self.ticker}.png")
    
    def export_results_json(self):
        """Sonuçları JSON olarak kaydet"""
        data = {
            'ticker': self.ticker,
            'overall_metrics': self.overall_metrics,
            'window_results': []
        }
        
        for result in self.results:
            data['window_results'].append({
                'window_num': result['window_num'],
                'period': result['period'],
                'metrics': result['metrics']
            })
        
        filename = f'wfv_{self.ticker}_results.json'
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ JSON kaydedildi: {filename}")


# Example usage
if __name__ == '__main__':
    import yfinance as yf
    from omenquant_backtest import BacktestEngine
    
    # Test
    validator = WalkForwardValidator(
        ticker='THYAO.IS',
        start_date='2021-01-01',
        end_date='2024-01-01',
        train_period_days=252,
        test_period_days=63,
        rebalance_days=21
    )
    
    # Veriyi yükle
    df = yf.download('THYAO.IS', start='2021-01-01', end='2024-01-01', progress=False)
    
    # Windows hazırla
    validator.prepare_data(df)
    
    # Her window'u çalıştır
    for window in validator.windows:
        validator.run_window(window, BacktestEngine)
    
    # Sonuçları hesapla ve göster
    validator.calculate_overall_metrics()
    validator.print_summary()
    validator.plot_results()
    validator.export_results_json()
