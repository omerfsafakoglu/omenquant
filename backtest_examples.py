"""
OmenQuant Backtest System - Quick Start Guide
==============================================

Bu belge OmenQuant'ın backtest sistemini nasıl kullanacağınızı gösterir.
"""

# ============================================================================
# 1. BASIT BACKTEST
# ============================================================================

def example_1_simple_backtest():
    """En basit backtest örneği"""
    from omenquant_backtest import BacktestEngine
    from backtest_reporter import BacktestReporter
    
    # Engine oluştur
    engine = BacktestEngine(
        ticker='THYAO.IS',
        start_date='2023-01-01',
        end_date='2024-01-01',
        initial_capital=10000,
        position_size_pct=0.5,
        use_lstm=False,  # Hız için LSTM'i kapatabilirsin
        stop_loss_pct=0.05,
        take_profit_pct=0.06
    )
    
    # Backtest çalıştır
    results = engine.run()
    
    # Rapor yazdır
    engine.print_report()
    
    # Grafikleri oluştur
    reporter = BacktestReporter(results, 'THYAO.IS')
    reporter.create_all_reports()  # HTML, PNG, JSON çıktı


# ============================================================================
# 2. MULTIPLE TİCKER BACKTEST
# ============================================================================

def example_2_multiple_tickers():
    """Birden fazla sembol için backtest"""
    from omenquant_backtest import BacktestEngine
    import pandas as pd
    
    tickers = ['THYAO.IS', 'GARAN.IS', 'AKBNK.IS', 'EREGL.IS', 'SISE.IS']
    results_summary = []
    
    for ticker in tickers:
        print(f"\n{'='*70}")
        print(f"Testing {ticker}...")
        
        engine = BacktestEngine(
            ticker=ticker,
            start_date='2023-01-01',
            end_date='2024-01-01'
        )
        
        results = engine.run()
        
        results_summary.append({
            'Ticker': ticker,
            'Return %': engine.metrics['total_return_pct'],
            'Sharpe': engine.metrics['sharpe_ratio'],
            'Max DD %': engine.metrics['max_drawdown_pct'],
            'Win Rate %': engine.metrics['win_rate_pct'],
            'Trades': engine.metrics['num_trades']
        })
    
    # Sonuçları tablo olarak göster
    summary_df = pd.DataFrame(results_summary)
    print(f"\n{'='*70}")
    print("📊 MULTIPLE TICKER SUMMARY")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))
    
    # CSV olarak kaydet
    summary_df.to_csv('backtest_summary.csv', index=False)
    print("\n✅ Özet CSV'ye kaydedildi: backtest_summary.csv")


# ============================================================================
# 3. WALK-FORWARD VALIDATION
# ============================================================================

def example_3_walk_forward_validation():
    """
    Overfitting'i kontrol etmek için walk-forward validation.
    Bu, gerçek trading performansını tahmin etmek için en iyi metod.
    """
    from walk_forward_validator import WalkForwardValidator
    import yfinance as yf
    
    validator = WalkForwardValidator(
        ticker='THYAO.IS',
        start_date='2021-01-01',
        end_date='2024-01-01',
        train_period_days=252,    # 1 yıllık training
        test_period_days=63,      # 3 aylık test
        rebalance_days=21         # Her 3 haftada güncelle
    )
    
    # Veriyi yükle
    print("📊 Veri yükleniyor...")
    df = yf.download('THYAO.IS', start='2021-01-01', end='2024-01-01', 
                    progress=False)
    
    # Windows hazırla
    print("🪟 Validation window'ları hazırlanıyor...")
    validator.prepare_data(df)
    
    # Her window'u çalıştır
    print("🔄 Backtest'ler çalışıyor...")
    for window in validator.windows:
        from omenquant_backtest import BacktestEngine
        validator.run_window(window, BacktestEngine)
    
    # Sonuçları göster
    validator.calculate_overall_metrics()
    validator.print_summary()
    validator.plot_results()
    validator.export_results_json()


# ============================================================================
# 4. PARAMETER OPTIMIZATION (Sensitivity Analysis)
# ============================================================================

def example_4_parameter_sensitivity():
    """
    Parametrelerin etkisini test et.
    Hangi stop-loss/take-profit optimal?
    """
    from omenquant_backtest import BacktestEngine
    import pandas as pd
    
    # Test parametreleri
    stop_losses = [0.03, 0.05, 0.07, 0.10]
    take_profits = [0.04, 0.06, 0.08, 0.10]
    
    results = []
    
    for sl in stop_losses:
        for tp in take_profits:
            if tp <= sl:
                continue  # TP > SL olmalı
            
            engine = BacktestEngine(
                ticker='THYAO.IS',
                start_date='2023-01-01',
                end_date='2024-01-01',
                stop_loss_pct=sl,
                take_profit_pct=tp
            )
            
            engine.run()
            
            results.append({
                'Stop Loss %': sl * 100,
                'Take Profit %': tp * 100,
                'Return %': engine.metrics['total_return_pct'],
                'Sharpe': engine.metrics['sharpe_ratio'],
                'Win Rate %': engine.metrics['win_rate_pct'],
                'Trades': engine.metrics['num_trades']
            })
            
            print(f"SL: {sl*100:.0f}% TP: {tp*100:.0f}% → Return: {engine.metrics['total_return_pct']:.2f}%")
    
    # Sonuçları göster
    results_df = pd.DataFrame(results)
    print("\n📊 SENSITIVITY ANALYSIS")
    print(results_df.to_string(index=False))
    
    # Best configuration
    best = results_df.loc[results_df['Return %'].idxmax()]
    print(f"\n🏆 En iyi: SL={best['Stop Loss %']:.0f}% TP={best['Take Profit %']:.0f}% → {best['Return %']:.2f}%")


# ============================================================================
# 5. PERIOD COMPARISON
# ============================================================================

def example_5_period_comparison():
    """
    Farklı zaman periyotlarında sistem performansını karşılaştır.
    Bull market vs Bear market'te nasıl çalışıyor?
    """
    from omenquant_backtest import BacktestEngine
    import pandas as pd
    
    periods = [
        ('2022 (Bear)', '2022-01-01', '2022-12-31'),
        ('2023 (Bull)', '2023-01-01', '2023-12-31'),
        ('2024 (Early)', '2024-01-01', '2024-03-31')
    ]
    
    results = []
    
    for period_name, start, end in periods:
        engine = BacktestEngine(
            ticker='THYAO.IS',
            start_date=start,
            end_date=end
        )
        
        results_dict = engine.run()
        engine._calculate_metrics()
        
        results.append({
            'Period': period_name,
            'Return %': engine.metrics['total_return_pct'],
            'Sharpe': engine.metrics['sharpe_ratio'],
            'Max DD %': engine.metrics['max_drawdown_pct'],
            'Win Rate %': engine.metrics['win_rate_pct'],
            'Buy Hold %': engine.metrics['buy_hold_return_pct']
        })
    
    # Göster
    results_df = pd.DataFrame(results)
    print("\n📊 PERIOD COMPARISON")
    print(results_df.to_string(index=False))


# ============================================================================
# 6. THESIS RAPORUNDA KULLANMAK IÇIN
# ============================================================================

def example_6_thesis_report():
    """
    Thesis'e eklenecek comprehensive backtest report'u oluştur.
    """
    from omenquant_backtest import BacktestEngine
    from backtest_reporter import BacktestReporter
    from walk_forward_validator import WalkForwardValidator
    import yfinance as yf
    
    print("\n" + "="*70)
    print("📋 THESIS IÇIN COMPREHENSIVE BACKTEST")
    print("="*70)
    
    ticker = 'THYAO.IS'
    
    # 1. In-sample backtest (with LSTM)
    print("\n1️⃣ In-Sample Backtest (2022-2023)")
    engine_insample = BacktestEngine(
        ticker=ticker,
        start_date='2022-01-01',
        end_date='2023-12-31',
        use_lstm=True
    )
    results_insample = engine_insample.run()
    engine_insample.print_report()
    
    # 2. Out-of-sample backtest
    print("\n2️⃣ Out-of-Sample Backtest (2024)")
    engine_outsample = BacktestEngine(
        ticker=ticker,
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    results_outsample = engine_outsample.run()
    engine_outsample.print_report()
    
    # 3. Walk-forward validation
    print("\n3️⃣ Walk-Forward Validation (2021-2024)")
    df = yf.download(ticker, start='2021-01-01', end='2024-12-31', progress=False)
    
    validator = WalkForwardValidator(
        ticker=ticker,
        start_date='2021-01-01',
        end_date='2024-01-01'
    )
    validator.prepare_data(df)
    
    for window in validator.windows:
        from omenquant_backtest import BacktestEngine
        validator.run_window(window, BacktestEngine)
    
    validator.calculate_overall_metrics()
    validator.print_summary()
    
    # 4. Raporları oluştur
    print("\n4️⃣ Raporlar oluşturuluyor...")
    reporter = BacktestReporter(results_insample, f'{ticker}_InSample')
    reporter.create_all_reports()
    
    reporter_oos = BacktestReporter(results_outsample, f'{ticker}_OutOfSample')
    reporter_oos.create_all_reports()
    
    validator.plot_results()
    
    # 5. Thesis için özet tablosu
    print("\n" + "="*70)
    print("📊 THESIS SUMMARY TABLE")
    print("="*70)
    print(f"\nIn-Sample Performance (2022-2023):")
    print(f"  Total Return:      {engine_insample.metrics['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio:      {engine_insample.metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:      {engine_insample.metrics['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate:          {engine_insample.metrics['win_rate_pct']:.1f}%")
    
    print(f"\nOut-of-Sample Performance (2024):")
    print(f"  Total Return:      {engine_outsample.metrics['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio:      {engine_outsample.metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:      {engine_outsample.metrics['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate:          {engine_outsample.metrics['win_rate_pct']:.1f}%")
    
    print(f"\nWalk-Forward Average:")
    print(f"  Avg Return:        {validator.overall_metrics['avg_return_pct']:.2f}%")
    print(f"  Avg Sharpe:        {validator.overall_metrics['avg_sharpe']:.3f}")
    print(f"  Positive Windows:  {validator.overall_metrics['positive_windows_pct']:.1f}%")


# ============================================================================
# MAIN - Hangi örneği çalıştırmak istiyorsun?
# ============================================================================

if __name__ == '__main__':
    import sys
    
    examples = {
        '1': ('Simple Backtest', example_1_simple_backtest),
        '2': ('Multiple Tickers', example_2_multiple_tickers),
        '3': ('Walk-Forward Validation', example_3_walk_forward_validation),
        '4': ('Parameter Sensitivity', example_4_parameter_sensitivity),
        '5': ('Period Comparison', example_5_period_comparison),
        '6': ('Thesis Report', example_6_thesis_report),
    }
    
    print("\n" + "="*70)
    print("🚀 OmenQuant Backtest Examples")
    print("="*70)
    
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    
    print("\nTercih et (1-6 arası):")
    
    # Eğer command line argument varsa, bunu kullan
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("> ").strip()
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\n▶️ {name} başlatılıyor...\n")
        func()
    else:
        print("❌ Geçersiz tercih")
