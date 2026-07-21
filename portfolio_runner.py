"""
OmenQuant Portfolio Runner - Çoklu BIST Hisse Portföy Sistemi
==============================================================
backtester_pro.py motorunu kullanarak birden fazla BIST hissesini
paralel test eder ve portföy düzeyinde performans hesaplar.

Strateji: TREND_FOLLOW (kanıtlanmış: THYAO +%26, 3 trade, Sharpe 4.24)
Hedef: Ayda ~10.000 TL (yılda ~%30+)

Ömer Faruk Şafakoğlu - OmenQuant Trading System
Yıldız Teknik Üniversitesi, İstatistik Bölümü, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    print("❌ pip install yfinance")
    sys.exit(1)

# backtester_pro.py import
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from backtester_pro import (
        BacktestConfig, BacktestEngine, TechnicalEngine,
        WalkForwardAnalyzer, MonteCarloSimulator,
        ReportGenerator, ConsoleReporter
    )
    print("✅ backtester_pro.py yüklendi")
except ImportError:
    print("❌ backtester_pro.py bulunamadı!")
    print("   Bu dosyayı backtester_pro.py ile aynı klasöre koy.")
    sys.exit(1)


# =============================================================================
# BIST HİSSE EVRENİ
# =============================================================================

# Likit, trend oluşturan BIST hisseleri
BIST_UNIVERSE = {
    # Havacılık & Savunma
    "THYAO.IS": {"name": "Türk Hava Yolları", "sector": "Havacılık"},
    "ASELS.IS": {"name": "Aselsan", "sector": "Savunma"},
    
    # Bankacılık
    "GARAN.IS": {"name": "Garanti BBVA", "sector": "Banka"},
    "AKBNK.IS": {"name": "Akbank", "sector": "Banka"},
    "ISCTR.IS": {"name": "İş Bankası C", "sector": "Banka"},
    "YKBNK.IS": {"name": "Yapı Kredi", "sector": "Banka"},
    
    # Sanayi
    "EREGL.IS": {"name": "Ereğli Demir Çelik", "sector": "Metal"},
    "TUPRS.IS": {"name": "Tüpraş", "sector": "Enerji"},
    "SISE.IS":  {"name": "Şişecam", "sector": "Cam"},
    "TOASO.IS": {"name": "Tofaş", "sector": "Otomotiv"},
    
    # Holding
    "KCHOL.IS": {"name": "Koç Holding", "sector": "Holding"},
    "SAHOL.IS": {"name": "Sabancı Holding", "sector": "Holding"},
    
    # Perakende & Telekom
    "BIMAS.IS": {"name": "BİM", "sector": "Perakende"},
    "TCELL.IS": {"name": "Turkcell", "sector": "Telekom"},
    
    # Teknoloji
    "LOGO.IS":  {"name": "Logo Yazılım", "sector": "Teknoloji"},
}


# =============================================================================
# PORTFÖY SİSTEMİ
# =============================================================================

class PortfolioBacktester:
    """
    Çoklu hisse portföy backtester.
    
    Her hisseyi bağımsız test eder, sonra portföy düzeyinde
    sermaye dağılımı simülasyonu yapar.
    
    Sermaye dağılım stratejileri:
    1. Equal Weight: Her hisseye eşit sermaye
    2. Momentum Rank: Son 3 ay momentum'a göre ağırlık
    3. Top-N: En iyi N hisseye odaklan
    """
    
    def __init__(self, 
                 initial_capital: float = 100_000,
                 strategy: str = "trend_follow",
                 max_positions: int = 5,
                 allocation: str = "equal",
                 data_start: str = "2020-01-01",
                 data_end: str = "2026-02-15",
                 trade_start: str = "2025-02-15",
                 trade_end: str = "2026-02-15"):
        
        self.initial_capital = initial_capital
        self.strategy = strategy
        self.max_positions = max_positions
        self.allocation = allocation
        self.data_start = data_start
        self.data_end = data_end
        self.trade_start = trade_start
        self.trade_end = trade_end
        
        self.individual_results = {}
        self.portfolio_result = {}
    
    def run(self, tickers: dict = None) -> dict:
        """Ana çalıştırıcı"""
        if tickers is None:
            tickers = BIST_UNIVERSE
        
        print(f"\n{'='*70}")
        print(f"  🏦 OmenQuant Portföy Backtester")
        print(f"  📊 {len(tickers)} hisse | Strateji: {self.strategy.upper()}")
        print(f"  💰 Sermaye: ₺{self.initial_capital:,.0f} | Max {self.max_positions} pozisyon")
        print(f"  📅 Veri: {self.data_start} → {self.data_end}")
        print(f"  🎯 Trade: {self.trade_start} → {self.trade_end}")
        print(f"{'='*70}\n")
        
        # 1. Her hisseyi ayrı ayrı test et
        self._run_individual_backtests(tickers)
        
        # 2. Portföy simülasyonu
        self._run_portfolio_simulation()
        
        # 3. Rapor üret
        self._generate_reports(tickers)
        
        return {
            'individual': self.individual_results,
            'portfolio': self.portfolio_result,
        }
    
    def _run_individual_backtests(self, tickers: dict):
        """Her hisseyi bağımsız backtest et"""
        
        config = BacktestConfig()
        config.initial_capital = self.initial_capital / self.max_positions
        config.max_position_pct = 0.95
        config.max_open_positions = 1
        
        print(f"📊 Bireysel backtestler çalışıyor...\n")
        print(f"  {'Hisse':<10} {'Getiri':>8} {'Alpha':>8} {'Sharpe':>8} {'WR':>6} {'PF':>6} {'Trade':>6} {'Hold':>6}")
        print(f"  {'─'*68}")
        
        for ticker, info in tickers.items():
            try:
                # Veri çek
                data = yf.download(ticker, start=self.data_start, end=self.data_end, progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data = data.dropna()
                
                if len(data) < 100:
                    print(f"  {ticker.replace('.IS',''):<10} ⚠️ Yetersiz veri ({len(data)} bar)")
                    continue
                
                # Trade penceresi
                trade_data = data[self.trade_start:self.trade_end].copy()
                if len(trade_data) < 20:
                    print(f"  {ticker.replace('.IS',''):<10} ⚠️ Trade penceresi çok kısa")
                    continue
                
                # Sinyal üret (tüm veri üzerinde)
                full_signals = TechnicalEngine.generate_signals(data, self.strategy)
                trade_signals = full_signals[self.trade_start:self.trade_end]
                
                # Backtest
                engine = BacktestEngine(config)
                result = engine.run(trade_data, trade_signals, ticker)
                
                self.individual_results[ticker] = {
                    'result': result,
                    'info': info,
                    'data': trade_data,
                }
                
                # Konsol çıktı
                name = ticker.replace('.IS', '')
                ret = result.get('total_return_pct', 0)
                alpha = result.get('alpha_pct', 0)
                sharpe = result.get('sharpe_ratio', 0)
                wr = result.get('win_rate_pct', 0)
                pf = result.get('profit_factor', 0)
                trades = result.get('total_trades', 0)
                hold = result.get('avg_holding_days', 0)
                
                ret_icon = '🟢' if ret > 0 else '🔴'
                print(f"  {ret_icon} {name:<8} {ret:+7.2f}% {alpha:+7.2f}% {sharpe:+7.3f} {wr:5.1f}% {pf:5.2f} {trades:5d} {hold:5.1f}d")
                
            except Exception as e:
                print(f"  ❌ {ticker.replace('.IS',''):<8} Hata: {e}")
        
        print(f"\n  ✅ {len(self.individual_results)}/{len(tickers)} hisse başarıyla test edildi")
    
    def _run_portfolio_simulation(self):
        """
        Portföy düzeyinde sermaye dağılımı simülasyonu.
        
        Mantık: trend_follow sinyali aktif olan hisselere sermaye dağıt.
        Max N pozisyon aynı anda açık olabilir.
        """
        if not self.individual_results:
            return
        
        print(f"\n📈 Portföy simülasyonu çalışıyor (max {self.max_positions} pozisyon)...\n")
        
        # Tüm hisselerin günlük getiri serileri
        daily_returns = {}
        active_signals = {}
        
        for ticker, data in self.individual_results.items():
            result = data['result']
            trade_data = data['data']
            
            # Günlük getiri
            dr = trade_data['Close'].pct_change().dropna()
            daily_returns[ticker] = dr
            
            # Strateji sinyalleri — pozisyonda olduğu günler
            signals = TechnicalEngine.generate_signals(trade_data, self.strategy)
            
            # Sinyal state: pozisyonda mı değil mi
            in_position = pd.Series(0, index=trade_data.index)
            state = 0
            for i in range(len(signals)):
                s = signals.iloc[i]
                if s == 1:
                    state = 1
                elif s == -1:
                    state = 0
                in_position.iloc[i] = state
            
            active_signals[ticker] = in_position
        
        # Günlük portföy getirisi hesapla
        all_dates = sorted(set().union(*[set(dr.index) for dr in daily_returns.values()]))
        
        portfolio_balance = [self.initial_capital]
        portfolio_dates = [all_dates[0]]
        
        for date in all_dates[1:]:
            # Bu gün hangi hisseler aktif?
            active_tickers = []
            for ticker in self.individual_results:
                if ticker in active_signals and date in active_signals[ticker].index:
                    if active_signals[ticker].loc[date] == 1:
                        active_tickers.append(ticker)
            
            # Max pozisyon sınırla
            if len(active_tickers) > self.max_positions:
                # Momentum bazlı seç: son 20 günde en iyi performans gösterenler
                momentum = {}
                for t in active_tickers:
                    if t in daily_returns:
                        dr = daily_returns[t]
                        recent = dr[dr.index <= date].tail(20)
                        momentum[t] = recent.sum() if len(recent) > 0 else 0
                active_tickers = sorted(momentum, key=momentum.get, reverse=True)[:self.max_positions]
            
            # Günlük getiri
            if active_tickers:
                weight = 1.0 / len(active_tickers)
                daily_ret = 0
                for t in active_tickers:
                    if t in daily_returns and date in daily_returns[t].index:
                        daily_ret += daily_returns[t].loc[date] * weight
                
                new_balance = portfolio_balance[-1] * (1 + daily_ret)
            else:
                new_balance = portfolio_balance[-1]
            
            portfolio_balance.append(new_balance)
            portfolio_dates.append(date)
        
        # Portföy metrikleri
        port_series = pd.Series(portfolio_balance, index=portfolio_dates)
        port_returns = port_series.pct_change().dropna()
        
        total_return = (port_series.iloc[-1] - port_series.iloc[0]) / port_series.iloc[0]
        
        n_days = len(port_series)
        n_years = n_days / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 and total_return > -1 else 0
        
        ann_vol = port_returns.std() * np.sqrt(252)
        rf = 0.45 / 252
        excess = port_returns - rf
        sharpe = (excess.mean() / port_returns.std()) * np.sqrt(252) if port_returns.std() > 0 else 0
        
        cummax = port_series.cummax()
        dd = (port_series - cummax) / cummax
        max_dd = dd.min()
        
        # Benchmark: BIST100
        try:
            bist = yf.download("XU100.IS", start=self.trade_start, end=self.trade_end, progress=False)
            if isinstance(bist.columns, pd.MultiIndex):
                bist.columns = bist.columns.get_level_values(0)
            bist_ret = (bist['Close'].iloc[-1] - bist['Close'].iloc[0]) / bist['Close'].iloc[0]
        except:
            bist_ret = 0
        
        alpha = total_return - bist_ret
        
        self.portfolio_result = {
            'initial_capital': self.initial_capital,
            'final_capital': port_series.iloc[-1],
            'total_return_pct': round(total_return * 100, 2),
            'cagr_pct': round(cagr * 100, 2),
            'annual_volatility_pct': round(ann_vol * 100, 2),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'bist100_return_pct': round(bist_ret * 100, 2),
            'alpha_pct': round(alpha * 100, 2),
            'n_stocks': len(self.individual_results),
            'max_positions': self.max_positions,
            'strategy': self.strategy,
            '_balance_series': port_series,
            '_drawdown_series': dd,
        }
        
        # Portföy özet
        print(f"  {'='*60}")
        print(f"  🏦 PORTFÖY SONUÇLARI")
        print(f"  {'='*60}")
        print(f"  💰 Sermaye: ₺{self.initial_capital:,.0f} → ₺{port_series.iloc[-1]:,.0f}")
        print(f"  📈 Toplam Getiri: {total_return*100:+.2f}%")
        print(f"  📊 CAGR: {cagr*100:+.2f}%")
        print(f"  📊 Sharpe Ratio: {sharpe:.3f}")
        print(f"  📉 Max Drawdown: {max_dd*100:.2f}%")
        print(f"  🏛️  BIST100: {bist_ret*100:+.2f}%")
        print(f"  🎯 Alpha: {alpha*100:+.2f}%")
        
        tl_profit = port_series.iloc[-1] - self.initial_capital
        monthly = tl_profit / max(n_years * 12, 1)
        print(f"\n  💵 Net Kâr: ₺{tl_profit:+,.0f}")
        print(f"  📅 Aylık Ortalama: ₺{monthly:+,.0f}")
        print(f"  {'='*60}")
    
    def _generate_reports(self, tickers):
        """HTML rapor üret"""
        
        if not self.individual_results:
            return
        
        # Hisse karşılaştırma tablosu
        rows = []
        for ticker, data in self.individual_results.items():
            r = data['result']
            info = data['info']
            rows.append({
                'Hisse': ticker.replace('.IS', ''),
                'İsim': info['name'],
                'Sektör': info['sector'],
                'Getiri (%)': r.get('total_return_pct', 0),
                'Alpha (%)': r.get('alpha_pct', 0),
                'Sharpe': r.get('sharpe_ratio', 0),
                'Max DD (%)': r.get('max_drawdown_pct', 0),
                'Win Rate (%)': r.get('win_rate_pct', 0),
                'PF': r.get('profit_factor', 0),
                'İşlem': r.get('total_trades', 0),
                'Ort. Hold': r.get('avg_holding_days', 0),
            })
        
        comp_df = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
        
        # HTML rapor
        p = self.portfolio_result
        ret_class = 'positive' if p.get('total_return_pct', 0) > 0 else 'negative'
        alpha_class = 'positive' if p.get('alpha_pct', 0) > 0 else 'negative'
        
        html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>OmenQuant Portföy Raporu</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0a0a1a; color:#e0e0e0; line-height:1.6; }}
    .container {{ max-width:1200px; margin:0 auto; padding:20px; }}
    .header {{ text-align:center; padding:30px; margin-bottom:20px; background:linear-gradient(135deg,#1a1a2e,#16213e); border-radius:16px; border:1px solid #333; }}
    .header h1 {{ font-size:2em; background:linear-gradient(45deg,#38ef7d,#11998e); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
    .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:15px; margin-bottom:20px; }}
    .kpi-card {{ background:linear-gradient(135deg,#1e1e3a,#2a2a4a); padding:20px; border-radius:12px; text-align:center; border:1px solid #333; }}
    .kpi-label {{ font-size:0.8em; color:#888; text-transform:uppercase; }}
    .kpi-value {{ font-size:1.5em; font-weight:bold; margin-top:5px; }}
    .positive {{ color:#38ef7d; }}
    .negative {{ color:#f45c43; }}
    .section {{ background:#1a1a2e; border-radius:12px; padding:20px; margin-bottom:15px; border:1px solid #333; }}
    .section h2 {{ font-size:1.2em; margin-bottom:15px; color:#38ef7d; }}
    .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:15px; margin-bottom:15px; }}
    table {{ width:100%; border-collapse:collapse; font-size:0.85em; }}
    th {{ background:#2a2a4a; padding:10px; text-align:left; }}
    td {{ padding:8px 10px; border-bottom:1px solid #222; }}
    .footer {{ text-align:center; padding:20px; color:#555; font-size:0.8em; margin-top:20px; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>🏦 OmenQuant Portföy Raporu</h1>
    <p>{len(self.individual_results)} hisse | {self.strategy.upper()} | Max {self.max_positions} pozisyon | {self.trade_start}</p>
</div>

<div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-label">Portföy Getiri</div><div class="kpi-value {ret_class}">{p.get('total_return_pct',0):+.2f}%</div></div>
    <div class="kpi-card"><div class="kpi-label">Sharpe Ratio</div><div class="kpi-value">{p.get('sharpe_ratio',0):.3f}</div></div>
    <div class="kpi-card"><div class="kpi-label">Max Drawdown</div><div class="kpi-value negative">{p.get('max_drawdown_pct',0):.2f}%</div></div>
    <div class="kpi-card"><div class="kpi-label">BIST100</div><div class="kpi-value">{p.get('bist100_return_pct',0):+.2f}%</div></div>
    <div class="kpi-card"><div class="kpi-label">Alpha</div><div class="kpi-value {alpha_class}">{p.get('alpha_pct',0):+.2f}%</div></div>
    <div class="kpi-card"><div class="kpi-label">Sermaye</div><div class="kpi-value">₺{p.get('final_capital',0):,.0f}</div></div>
</div>
"""
        
        # Portföy equity grafiği
        port_series = p.get('_balance_series')
        dd_series = p.get('_drawdown_series')
        
        if port_series is not None:
            import json as _json
            eq_dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in port_series.index]
            eq_vals = [round(v, 0) for v in port_series.values]
            dd_dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in dd_series.index]
            dd_vals = [round(v*100, 2) for v in dd_series.values]
            
            html += f"""
<div class="two-col">
    <div class="section">
        <h2>💰 Portföy Equity</h2>
        <div id="eqChart" style="width:100%;height:350px;"></div>
    </div>
    <div class="section">
        <h2>📉 Drawdown</h2>
        <div id="ddChart" style="width:100%;height:350px;"></div>
    </div>
</div>
<script>
var dark = {{paper_bgcolor:'#1a1a2e',plot_bgcolor:'#0a0a1a',font:{{color:'#e0e0e0'}},
    xaxis:{{gridcolor:'#222'}},yaxis:{{gridcolor:'#222'}},margin:{{l:60,r:30,t:30,b:40}}}};
Plotly.newPlot('eqChart',[{{x:{_json.dumps(eq_dates)},y:{_json.dumps(eq_vals)},type:'scatter',mode:'lines',
    name:'Portföy',line:{{color:'#38ef7d',width:2}},fill:'tozeroy',fillcolor:'rgba(56,239,125,0.1)'}}],
    Object.assign({{}},dark,{{yaxis:{{title:'₺'}}}}),{{responsive:true}});
Plotly.newPlot('ddChart',[{{x:{_json.dumps(dd_dates)},y:{_json.dumps(dd_vals)},type:'scatter',mode:'lines',
    name:'Drawdown',line:{{color:'#f45c43',width:1.5}},fill:'tozeroy',fillcolor:'rgba(244,92,67,0.2)'}}],
    Object.assign({{}},dark,{{yaxis:{{title:'DD (%)'}}}}),{{responsive:true}});
</script>
"""
        
        # Hisse karşılaştırma tablosu
        html += '<div class="section"><h2>📊 Hisse Bazlı Performans</h2><table>'
        html += '<tr>' + ''.join(f'<th>{c}</th>' for c in comp_df.columns) + '</tr>'
        for _, row in comp_df.iterrows():
            ret_val = row.get('Getiri (%)', 0)
            row_class = 'style="background:rgba(56,239,125,0.05)"' if ret_val > 0 else 'style="background:rgba(244,92,67,0.05)"' if ret_val < 0 else ''
            html += f'<tr {row_class}>' + ''.join(f'<td>{v}</td>' for v in row.values) + '</tr>'
        html += '</table></div>'
        
        # Sektör analizi
        sector_perf = comp_df.groupby('Sektör').agg({
            'Getiri (%)': 'mean', 'Sharpe': 'mean', 'İşlem': 'sum'
        }).round(2).sort_values('Sharpe', ascending=False)
        
        html += '<div class="section"><h2>🏭 Sektör Performansı</h2><table>'
        html += '<tr><th>Sektör</th><th>Ort. Getiri (%)</th><th>Ort. Sharpe</th><th>Toplam İşlem</th></tr>'
        for sector, row in sector_perf.iterrows():
            html += f'<tr><td>{sector}</td><td>{row["Getiri (%)"]:.2f}</td><td>{row["Sharpe"]:.3f}</td><td>{int(row["İşlem"])}</td></tr>'
        html += '</table></div>'
        
        # Puanlama
        score = 0
        items = []
        
        ret = p.get('total_return_pct', 0)
        s = 20 if ret > 20 else 15 if ret > 10 else 10 if ret > 0 else 5 if ret > -10 else 0
        score += s; items.append(('📈 Portföy Getiri', f'{ret:+.2f}%', s, 20))
        
        sh = p.get('sharpe_ratio', 0)
        s = 20 if sh > 2 else 15 if sh > 1 else 10 if sh > 0.5 else 5 if sh > 0 else 0
        score += s; items.append(('📊 Sharpe', f'{sh:.3f}', s, 20))
        
        mdd = abs(p.get('max_drawdown_pct', 0))
        s = 15 if mdd < 5 else 12 if mdd < 10 else 8 if mdd < 15 else 4 if mdd < 25 else 0
        score += s; items.append(('📉 Max DD', f'{mdd:.2f}%', s, 15))
        
        al = p.get('alpha_pct', 0)
        s = 15 if al > 10 else 12 if al > 5 else 8 if al > 0 else 4 if al > -5 else 0
        score += s; items.append(('🎯 Alpha', f'{al:+.2f}%', s, 15))
        
        # Diversifikasyon
        pos_stocks = sum(1 for _, d in self.individual_results.items() if d['result'].get('total_return_pct', 0) > 0)
        div_ratio = pos_stocks / max(len(self.individual_results), 1)
        s = 15 if div_ratio > 0.6 else 10 if div_ratio > 0.4 else 5 if div_ratio > 0.2 else 0
        score += s; items.append(('🔄 Diversifikasyon', f'{pos_stocks}/{len(self.individual_results)} kârlı', s, 15))
        
        # Tutarlılık — kaç hisse pozitif Sharpe
        pos_sharpe = sum(1 for _, d in self.individual_results.items() if d['result'].get('sharpe_ratio', 0) > 0)
        s = 15 if pos_sharpe / max(len(self.individual_results), 1) > 0.6 else 10 if pos_sharpe > 3 else 5 if pos_sharpe > 1 else 0
        score += s; items.append(('📏 Tutarlılık', f'{pos_sharpe} hisse Sharpe>0', s, 15))
        
        if score >= 80: grade, gc, ge = 'A+', '#38ef7d', '🏆'
        elif score >= 65: grade, gc, ge = 'A', '#38ef7d', '⭐'
        elif score >= 50: grade, gc, ge = 'B', '#ffd700', '👍'
        elif score >= 35: grade, gc, ge = 'C', '#ffa500', '⚠️'
        elif score >= 20: grade, gc, ge = 'D', '#f45c43', '👎'
        else: grade, gc, ge = 'F', '#f45c43', '💀'
        
        html += f"""
<div class="section">
    <h2>🏅 Portföy Puanlama</h2>
    <div style="text-align:center;margin-bottom:20px;">
        <div style="font-size:3em;font-weight:bold;color:{gc};">{ge} {grade}</div>
        <div style="font-size:1.5em;color:{gc};">{score} / 100</div>
    </div>
    <table><tr><th>Metrik</th><th>Değer</th><th>Puan</th><th>Max</th><th>Bar</th></tr>"""
        
        for name, val, pts, mx in items:
            pct = pts / mx * 100
            bc = '#38ef7d' if pct >= 70 else '#ffd700' if pct >= 40 else '#f45c43'
            html += f'<tr><td>{name}</td><td>{val}</td><td><b>{pts}</b></td><td>{mx}</td>'
            html += f'<td><div style="background:#222;border-radius:4px;width:120px;height:16px;display:inline-block;">'
            html += f'<div style="background:{bc};height:100%;width:{pct}%;border-radius:4px;"></div></div></td></tr>'
        
        html += '</table></div>'
        
        # Aylık TL hesabı
        tl_profit = p.get('final_capital', 0) - self.initial_capital
        months = max(len(port_series) / 21, 1) if port_series is not None else 12
        monthly_tl = tl_profit / months
        
        html += f"""
<div class="section">
    <h2>💵 Aylık Gelir Projeksiyonu</h2>
    <div style="text-align:center;padding:20px;">
        <div style="font-size:1.2em;color:#888;">₺{self.initial_capital:,.0f} sermaye ile</div>
        <div style="font-size:2.5em;font-weight:bold;color:{'#38ef7d' if monthly_tl > 0 else '#f45c43'};">₺{monthly_tl:+,.0f} / ay</div>
        <div style="font-size:1em;color:#888;margin-top:10px;">Yıllık: ₺{tl_profit:+,.0f} ({p.get('total_return_pct',0):+.2f}%)</div>
    </div>
</div>
"""
        
        html += f"""
<div class="footer">
    OmenQuant Portföy Sistemi | Ömer Faruk Şafakoğlu | YTÜ İstatistik 2025<br>
    {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
</div></body></html>"""
        
        filepath = f"omenquant_portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"\n✅ Rapor: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    # Top 10 likit BIST hissesi
    portfolio = PortfolioBacktester(
        initial_capital=100_000,          # ₺100K sermaye
        strategy="trend_follow",           # Kanıtlanmış strateji
        max_positions=5,                   # Aynı anda max 5 hisse
        allocation="equal",
        data_start="2020-01-01",           # 5 yıl veri (indikatör eğitimi)
        data_end="2026-02-15",
        trade_start="2025-02-15",          # Son 1 yıl trade
        trade_end="2026-02-15",
    )
    
    portfolio.run(BIST_UNIVERSE)
