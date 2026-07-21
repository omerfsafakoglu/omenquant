"""
OmenQuant Backtest Reporter & Visualizer
==========================================
Backtest sonuçlarını detaylı raporlarla ve grafiklerle sunuyor.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class BacktestReporter:
    """Backtest sonuçlarını raporla ve görselleştir"""
    
    def __init__(self, backtest_results, ticker):
        self.trades = backtest_results['trades']
        self.equity_curve = pd.DataFrame(backtest_results['equity_curve'])
        self.signals = pd.DataFrame(backtest_results['signals'])
        self.metrics = backtest_results['metrics']
        self.ticker = ticker
    
    def create_equity_curve_plot(self):
        """Equity curve grafiği"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(self.equity_curve['date'], self.equity_curve['equity'], 
               linewidth=2, color='#667eea', label='Portfolio Equity')
        
        # Trade markers
        for trade in self.trades:
            entry_equity = self.equity_curve[
                self.equity_curve['date'] == trade['entry_date']
            ]['equity'].values
            if len(entry_equity) > 0:
                ax.scatter(trade['entry_date'], entry_equity[0], 
                          marker='^' if trade['type'] == 'LONG' else 'v',
                          color='green' if trade['type'] == 'LONG' else 'red',
                          s=100, alpha=0.7, zorder=5)
            
            exit_equity = self.equity_curve[
                self.equity_curve['date'] == trade['exit_date']
            ]['equity'].values
            if len(exit_equity) > 0:
                ax.scatter(trade['exit_date'], exit_equity[0],
                          marker='o', color='blue', s=50, alpha=0.5, zorder=5)
        
        ax.set_xlabel('Tarih', fontsize=11, fontweight='bold')
        ax.set_ylabel('Sermaye ($)', fontsize=11, fontweight='bold')
        ax.set_title(f'💰 {self.ticker} - Equity Curve', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'backtest_{self.ticker}_equity.png', dpi=300, bbox_inches='tight')
        print(f"✅ Grafikler kaydedildi")
    
    def create_drawdown_plot(self):
        """Maximum Drawdown grafiği"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Calculate running max and drawdown
        equity = self.equity_curve['equity'].values
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        
        colors = ['red' if dd < 0 else 'green' for dd in drawdown]
        ax.bar(self.equity_curve['date'], drawdown, color=colors, alpha=0.6, width=1)
        
        ax.set_xlabel('Tarih', fontsize=11, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'📉 {self.ticker} - Drawdown Analysis', fontsize=13, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'backtest_{self.ticker}_drawdown.png', dpi=300, bbox_inches='tight')
    
    def create_monthly_returns_heatmap(self):
        """Aylık return heatmap'i"""
        self.equity_curve['date'] = pd.to_datetime(self.equity_curve['date'])
        self.equity_curve['return'] = self.equity_curve['equity'].pct_change() * 100
        self.equity_curve['year'] = self.equity_curve['date'].dt.year
        self.equity_curve['month'] = self.equity_curve['date'].dt.month
        
        # Group by month
        monthly_returns = self.equity_curve.groupby(['year', 'month'])['return'].sum().unstack()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(monthly_returns, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Return (%)'}, ax=ax)
        
        ax.set_title(f'📊 {self.ticker} - Aylık Return Heatmap', fontsize=13, fontweight='bold')
        ax.set_xlabel('Ay', fontsize=11, fontweight='bold')
        ax.set_ylabel('Yıl', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'backtest_{self.ticker}_monthly.png', dpi=300, bbox_inches='tight')
    
    def create_win_loss_distribution(self):
        """Kazanan/Kayıp Trade'lerin dağılımı"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Adaptive bins based on number of trades
        num_trades = len(self.trades)
        bins = max(5, min(20, num_trades // 2))
        
        # PnL histogram
        wins = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trades if t['pnl'] < 0]
        
        if wins:
            axes[0].hist(wins, bins=bins, alpha=0.7, label='Wins', color='green')
        if losses:
            axes[0].hist(losses, bins=bins, alpha=0.7, label='Losses', color='red')
        axes[0].set_xlabel('PnL ($)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('PnL Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # PnL % histogram
        wins_pct = [t['pnl_pct']*100 for t in self.trades if t['pnl'] > 0]
        losses_pct = [t['pnl_pct']*100 for t in self.trades if t['pnl'] < 0]
        
        if wins_pct:
            axes[1].hist(wins_pct, bins=bins, alpha=0.7, label='Wins', color='green')
        if losses_pct:
            axes[1].hist(losses_pct, bins=bins, alpha=0.7, label='Losses', color='red')
        axes[1].set_xlabel('PnL (%)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1].set_title('PnL % Distribution', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(f'🎯 {self.ticker} - Trade Distribution', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'backtest_{self.ticker}_distribution.png', dpi=300, bbox_inches='tight')
    
    def create_signal_chart(self):
        """Sinyal ve fiyat grafiği"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart
        ax1.plot(self.signals['date'], self.signals['price'], 
                linewidth=1.5, color='black', label='Price', alpha=0.7)
        
        # Signals
        long_signals = self.signals[self.signals['signal'] == 'LONG']
        short_signals = self.signals[self.signals['signal'] == 'SHORT']
        
        ax1.scatter(long_signals['date'], long_signals['price'], 
                   marker='^', color='green', s=100, label='LONG Signal', alpha=0.7)
        ax1.scatter(short_signals['date'], short_signals['price'], 
                   marker='v', color='red', s=100, label='SHORT Signal', alpha=0.7)
        
        ax1.set_ylabel('Price', fontsize=11, fontweight='bold')
        ax1.set_title(f'📈 {self.ticker} - Fiyat & Sinyaller', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Score chart
        ax2.bar(self.signals['date'], self.signals['score'], 
               color=['green' if s > 0 else 'red' for s in self.signals['score']], 
               alpha=0.7, width=1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Tarih', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Signal Score', fontsize=11, fontweight='bold')
        ax2.set_title('Signal Score Timeline', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'backtest_{self.ticker}_signals.png', dpi=300, bbox_inches='tight')
    
    def create_html_report(self, filename=None):
        """HTML raporu oluştur"""
        if filename is None:
            filename = f'backtest_{self.ticker}_report.html'
        
        m = self.metrics
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>OmenQuant Backtest Report - {self.ticker}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: #f5f5f5;
                    color: #333;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                }}
                .header p {{
                    margin: 5px 0 0 0;
                    font-size: 1.1em;
                    opacity: 0.9;
                }}
                .section {{
                    background: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                .section h2 {{
                    color: #667eea;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                    margin-top: 0;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                .metric {{
                    background: #f9f9f9;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 4px solid #667eea;
                }}
                .metric-label {{
                    font-size: 0.9em;
                    color: #666;
                    margin-bottom: 5px;
                }}
                .metric-value {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #333;
                }}
                .positive {{ color: #10b981; }}
                .negative {{ color: #ef4444; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 15px;
                }}
                th {{
                    background: #667eea;
                    color: white;
                    padding: 10px;
                    text-align: left;
                    font-weight: 600;
                }}
                td {{
                    padding: 10px;
                    border-bottom: 1px solid #eee;
                }}
                tr:hover {{
                    background: #f9f9f9;
                }}
                .image-section {{
                    text-align: center;
                    margin-top: 15px;
                }}
                .image-section img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    margin: 10px 0;
                }}
                .footer {{
                    text-align: center;
                    color: #999;
                    margin-top: 30px;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>⚡ OmenQuant Backtest Report</h1>
                <p>{self.ticker} | {self.equity_curve['date'].min().strftime('%Y-%m-%d')} to {self.equity_curve['date'].max().strftime('%Y-%m-%d')}</p>
            </div>
            
            <div class="section">
                <h2>💰 Performance Summary</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-label">Total Return</div>
                        <div class="metric-value {'positive' if m['total_return_pct'] > 0 else 'negative'}">
                            {m['total_return_pct']:.2f}%
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Annual Return</div>
                        <div class="metric-value {'positive' if m['annual_return_pct'] > 0 else 'negative'}">
                            {m['annual_return_pct']:.2f}%
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Buy & Hold Return</div>
                        <div class="metric-value {'positive' if m['buy_hold_return_pct'] > 0 else 'negative'}">
                            {m['buy_hold_return_pct']:.2f}%
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value negative">{m['max_drawdown_pct']:.2f}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{m['sharpe_ratio']:.3f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{m['win_rate_pct']:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Profit Factor</div>
                        <div class="metric-value">{m['profit_factor']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Total Trades</div>
                        <div class="metric-value">{m['num_trades']}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Trade Statistics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Average Win</td>
                        <td class="positive">${m['avg_win']:,.2f}</td>
                    </tr>
                    <tr>
                        <td>Average Loss</td>
                        <td class="negative">${m['avg_loss']:,.2f}</td>
                    </tr>
                    <tr>
                        <td>Payoff Ratio</td>
                        <td>{m['payoff_ratio']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Sortino Ratio</td>
                        <td>{m['sortino_ratio']:.3f}</td>
                    </tr>
                    <tr>
                        <td>Calmar Ratio</td>
                        <td>{m['calmar_ratio']:.3f}</td>
                    </tr>
                    <tr>
                        <td>Recovery Factor</td>
                        <td>{m['recovery_factor']:.2f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>🏆 Best Trades</h2>
                <table>
                    <tr>
                        <th>Entry Date</th>
                        <th>Exit Date</th>
                        <th>Type</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>PnL</th>
                        <th>Return %</th>
                    </tr>
        """
        
        # Best trades
        best_trades = sorted(self.trades, key=lambda x: x['pnl'], reverse=True)[:5]
        for trade in best_trades:
            html += f"""
                    <tr>
                        <td>{trade['entry_date'].strftime('%Y-%m-%d')}</td>
                        <td>{trade['exit_date'].strftime('%Y-%m-%d')}</td>
                        <td>{trade['type']}</td>
                        <td>${trade['entry_price']:.2f}</td>
                        <td>${trade['exit_price']:.2f}</td>
                        <td class="positive">${trade['pnl']:,.2f}</td>
                        <td class="positive">{trade['pnl_pct']*100:.2f}%</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>📈 Charts</h2>
                <div class="image-section">
                    <h3>Equity Curve</h3>
                    <img src="backtest_{}_equity.png" alt="Equity Curve">
                </div>
                <div class="image-section">
                    <h3>Drawdown Analysis</h3>
                    <img src="backtest_{}_drawdown.png" alt="Drawdown">
                </div>
                <div class="image-section">
                    <h3>Price & Signals</h3>
                    <img src="backtest_{}_signals.png" alt="Signals">
                </div>
                <div class="image-section">
                    <h3>Trade Distribution</h3>
                    <img src="backtest_{}_distribution.png" alt="Distribution">
                </div>
            </div>
            
            <div class="footer">
                <p>Generated on {}</p>
                <p>OmenQuant - Professional Trading Intelligence</p>
            </div>
        </body>
        </html>
        """.format(self.ticker, self.ticker, self.ticker, self.ticker, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ HTML Raporu {filename}'e kaydedildi")
    
    def create_all_reports(self):
        """Tüm grafikleri ve raporları oluştur"""
        print(f"\n📊 {self.ticker} için raporlar oluşturuluyor...")
        self.create_equity_curve_plot()
        self.create_drawdown_plot()
        self.create_monthly_returns_heatmap()
        self.create_win_loss_distribution()
        self.create_signal_chart()
        self.create_html_report()
        print(f"✅ Tüm raporlar tamamlandı!")


if __name__ == '__main__':
    # Example
    from omenquant_backtest import BacktestEngine
    
    engine = BacktestEngine(
        ticker='THYAO.IS',
        start_date='2022-01-01',
        end_date='2024-01-01'
    )
    
    results = engine.run()
    
    reporter = BacktestReporter(results, 'THYAO.IS')
    reporter.create_all_reports()
