# OmenQuant

A research and backtesting system for systematic trading strategies, built in Python.

The goal is not to find a strategy that looks good on a chart. It is to build a pipeline where a
result can be trusted: clean data in, an honest backtest out, and enough validation to tell the
difference between a real edge and an artefact of how the data was prepared.

## What it does

- **Backtesting engine** with portfolio accounting, commission handling and trade-level bookkeeping.
- **Walk-forward validation**, so a strategy is tested on data it was not tuned on rather than judged by a single in-sample run.
- **Signal generation** for live monitoring, separated from the backtest logic.
- **Portfolio runner** for evaluating several strategies together instead of one at a time.
- **Reporting** with risk-adjusted metrics: Sharpe ratio, maximum drawdown, win rate and trade statistics.
- **Streamlit dashboard** for reviewing results, positions and signals interactively.

Four systematic strategies have been backtested over two years of historical data.

## Stack

Python, pandas, NumPy, statsmodels, scipy, scikit-learn, Plotly, Streamlit, yfinance.

A Rust port of the backtesting core is planned, mainly for speed on larger datasets and
higher-frequency bars.

## Files

| File | Purpose |
|---|---|
| `backtester_pro.py` | Main backtesting engine |
| `omenquant_backtest.py` | `BacktestEngine` used by the other modules |
| `walk_forward_validator.py` | Walk-forward / out-of-sample validation |
| `portfolio_runner.py` | Multi-strategy portfolio evaluation |
| `backtest_reporter.py` | Performance and risk reporting |
| `backtest_examples.py` | Worked examples of the API |
| `live_signals.py` | Signal generation for live monitoring |
| `omenquant_dashboard.py` | Streamlit dashboard |
| `debug_signals.py` | Small script for inspecting signal output |

`BACKTEST_SYSTEM_DOCUMENTATION.md` covers the engine in more detail and
`EXECUTION_RUNBOOK.md` describes how the system is run.

## Getting started

```bash
git clone https://github.com/omerfsafakoglu/omenquant.git
cd omenquant
pip install -r requirements.txt
```

Copy the example settings file and fill in your own values:

```bash
cp omenquant_settings.example.json omenquant_settings.json
```

Run the examples:

```bash
python backtest_examples.py
```

Or open the dashboard:

```bash
streamlit run omenquant_dashboard.py
```

## Example output

<!-- Buraya bir ekran goruntusu ya da equity curve ekle:
![dashboard](docs/dashboard.png)
-->

## Notes and limitations

- Backtest results are not live trading results. Slippage and fill assumptions are simplified.
- Two years of data is a short sample. Results are treated as directional, not conclusive.
- This is a personal research project. Nothing here is investment advice.

## Roadmap

- [ ] Rust implementation of the backtest engine
- [ ] More detailed transaction cost modelling
- [ ] Unit tests around the engine and portfolio accounting
- [ ] Parameter sensitivity analysis across strategies

## License

MIT
