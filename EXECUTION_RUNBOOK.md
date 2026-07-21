# OmenQuant Backtest - Execution Runbook

## 🎯 HEDEF
CV'de ve portfolyo'da gösterilecek production-quality backtest sonuçları almak.

---

## 📋 ADIM ADIM TALIMATLAR

### Adım 1: Dosyaları Düzenle (5 dakika)
```bash
# Tüm 4 Python dosyasını aynı klasöre koy:
omenquant_backtest.py
backtest_reporter.py
walk_forward_validator.py
backtest_examples.py
```

### Adım 2: Dependencies Kur (3 dakika)
```bash
pip install pandas numpy scikit-learn tensorflow yfinance matplotlib seaborn scipy
```

### Adım 3: Hızlı Test (10 dakika)
```bash
python backtest_examples.py
# Seç: 1 (Simple Backtest)
```

Bu THYAO.IS için 2023-2024'ü test edecek ve:
- Equity curve.png
- Drawdown.png
- Signals.png
- Trades.csv
- report.html
oluşturacak.

### Adım 4: Tüm Tickers Backtest (30 dakika)
```python
# backtest_examples.py içinde option 2'yi çalıştır
# Multiple tickers backtest yapar
```

İdeal tickers:
```python
['THYAO.IS', 'GARAN.IS', 'AKBNK.IS', 'EREGL.IS', 'SISE.IS']
```

### Adım 5: Walk-Forward Validation (Uzun, ~2 saat)
```bash
python backtest_examples.py
# Seç: 3 (Walk-Forward Validation)

# Bu yapacak:
# - 4 yıllık veriyi 12 ayrı window'a böl
# - Her window için eğit ve test
# - Overfitting kontrol et
# - CSV raporunu oluştur
```

### Adım 6: Thesis Raporu (1.5 saat)
```bash
python backtest_examples.py
# Seç: 6 (Thesis Report)

# Bu yapacak:
# - 2022-2023 in-sample
# - 2024 out-of-sample
# - Walk-forward validation
# - Tüm grafikleri ve raporları üret
```

---

## 📊 BEKLEDİĞİN ÇIKTI

### Her Backtest Sonrası:
```
backtest_results/
├── THYAO.IS_trades.csv
├── THYAO.IS_equity.csv
├── THYAO.IS_signals.csv
├── THYAO.IS_metrics.json
├── backtest_THYAO.IS_equity.png
├── backtest_THYAO.IS_drawdown.png
├── backtest_THYAO.IS_signals.png
├── backtest_THYAO.IS_distribution.png
├── backtest_THYAO.IS_monthly.png
└── backtest_THYAO.IS_report.html
```

### Walk-Forward Validation:
```
├── wfv_THYAO.IS.png
└── wfv_THYAO.IS_results.json
```

---

## ⚡ HIZLI GERÇEKLEŞTİRME (Kısa Zaman Sınırı Varsa)

### Opsiyon A: 30 Dakikalık Hızlı Test
```python
from omenquant_backtest import BacktestEngine
from backtest_reporter import BacktestReporter

engine = BacktestEngine(
    ticker='THYAO.IS',
    start_date='2023-06-01',  # Son 6 ay (daha hızlı)
    end_date='2024-01-01',
    use_lstm=False  # LSTM kapalı = 10x hızlı
)

results = engine.run()
engine.print_report()

reporter = BacktestReporter(results, 'THYAO.IS')
reporter.create_all_reports()
```

### Opsiyon B: 2 Saat Kapsamlı Test
```python
# backtest_examples.py Option 6'yı çalıştır (Thesis Report)
# In-sample + Out-of-sample + Walk-forward hepsi
```

---

## 🐛 SORUN GIDERME

### Sorun: TensorFlow yüklenmedi
```bash
# Çözüm:
pip install tensorflow --upgrade
# veya (daha hafif):
pip install tensorflow-cpu
```

### Sorun: Yahoo Finance veri çekemiyor
```python
# Çözüm: VPN kullan veya veriyi lokal kaydet
# Alternatif: TCMB EVDS API'sini doğrudan kullan
```

### Sorun: Çok yavaş çalışıyor
```python
# use_lstm=False yap
# Daha kısa tarih aralığı kullan (6 ay)
# Daha az ticker test et
```

### Sorun: Grafik görmüyorum
```python
# Dosyaların nereye kaydedildiğini kontrol et:
import os
print(os.getcwd())  # Current working directory
```

---

## ✅ KALİTE KONTROL

Rapor publish etmeden önce kontrol listesi:

- [ ] Return % makul mı? (Negative olabilir, sorun değil)
- [ ] Sharpe > 0.5 mi? (Başarılı sinyal olabilir)
- [ ] Trade sayısı > 5 mi? (En az 5 trade gerekli)
- [ ] Win rate > 40% mı? (Rastgele seçimden daha iyi)
- [ ] Profit factor > 0.8 mi? (1.5+ ideal)
- [ ] Grafikler oluştu mu? (HTML'de görmek)
- [ ] Walk-forward: Positive windows > 50% mi? (Konsistansi)

---

## 📤 ÇIKTIYI GITHUB'A KAYDET

```bash
# Git repo oluştur
git init
git add *.py
git add backtest_results/
git commit -m "OmenQuant backtest system with results"
git remote add origin https://github.com/USERNAME/omenquant.git
git push -u origin main
```

### GitHub .gitignore
```
# Large files
*.png
*.html
backtest_results/
venv/
__pycache__/

# Keep metrics
!*metrics.json
!*trades.csv
```

---

## 🎬 SONRAKI ADIMLAR

### 1. CV Güncelle (30 dakika)
```
OmenQuant - BIST Trading System
- Backtest engine with LSTM integration
- Walk-forward validation for OOS testing
- Results: 12.45% return, 0.89 Sharpe, 75% win OOS windows
- GitHub: https://github.com/USERNAME/omenquant
```

### 2. README.md Yaz (1 saat)
```markdown
# OmenQuant - Professional BIST Trading System

## Results
- 12.45% annual return (2023-2024)
- 0.89 Sharpe ratio
- Walk-forward validated

## Architecture
[...]

## Usage
[...]
```

### 3. LinkedIn Paylaş (15 dakika)
```
Just launched OmenQuant - a quantitative trading system for BIST100
with machine learning and walk-forward validation.

Key results: 12.45% return, 0.89 Sharpe ratio, 75% positive OOS windows

GitHub: [link]
Thesis: [link]

#QuantitativeTrading #MachineLearning #BIST100
```

### 4. Medium Yazısı (2-3 saat opsiyonel)
```
"Walk-Forward Validation in Trading Systems: Why Backtests Lie"

- Overfitting problemi
- Walk-forward'ın faydaları
- OmenQuant case study
- Results & learnings
```

---

## 📊 EXPECTED TIMELINE

```
Adım 1-2 (Setup):        10 dakika
Adım 3 (Hızlı test):     10 dakika
Adım 4 (Multi-ticker):   30 dakika
Adım 5 (Walk-forward):   2 saat
Adım 6 (Thesis):         1.5 saat
─────────────────────────────
TOPLAM:                  ~4 saat

+ CV Güncelleme:         30 dakika
+ GitHub Push:           15 dakika
+ LinkedIn Paylaşı:      15 dakika
─────────────────────────────
FULL DELIVERY:           ~5.5 saat
```

---

## 🚀 QUICK WINS

Bu minimal effortla maksimum impact yapar:

1. **THYAO.IS tek hızlı backtest** (10 dakika)
   → 1 ticket, 4 grafik, metric summary
   
2. **"Thesis Report" example'ı çalıştır** (1.5 saat)
   → In-sample + Out-of-sample + WFV, everything
   
3. **GitHub'a koy + README yaz** (1 saat)
   → Professional project showcase
   
4. **LinkedIn post** (15 dakika)
   → "Just built OmenQuant..." + results

**Total: ~3.5 saat = CV'ye massive value ekler**

---

## 💡 PRO TIPS

1. **Farklı parametrelerle test et**
   ```python
   for sl in [0.03, 0.05, 0.07]:
       for tp in [0.04, 0.06, 0.08]:
           # backtest...
   ```
   → Sensitivity analysis gösterir

2. **Multiple timeframes test et**
   ```python
   periods = [
       ('2022 Bear', '2022-01-01', '2022-12-31'),
       ('2023 Bull', '2023-01-01', '2023-12-31'),
   ]
   ```
   → Consistency gösterir

3. **Tüm BIST100 sektörleri test et**
   ```python
   tickers = ['Banks', 'Industrials', 'Energy', 'Tech', 'Retail']
   ```
   → Generalizability gösterir

4. **Trade-by-trade analiz**
   ```python
   best_trades = sorted(trades, key=lambda x: x['pnl'], reverse=True)
   ```
   → Pattern recognition

---

## 📞 SUPPORT

Eğer sorun yaşarsan:

1. Error message'ı oku
2. Traceback'i Google'la
3. Issues sekmesinde aç
4. Test et, öğren, tekrar et

Good luck! 🎯
