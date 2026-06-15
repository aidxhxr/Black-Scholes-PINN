# Code

## Files

**`PINN_code.ipynb`** — Main notebook. Defines and trains `BlackScholesPINN`, fetches live AAPL options data via `yfinance`, evaluates against held-out market contracts, and generates all four paper figures.

**`AAPL_historical_behavior.ipynb`** — EDA on one year of AAPL daily prices. Used to calibrate the historical volatility parameter (σ=0.14, annualized over 252 trading days).

**`proof_of_concept.ipynb`** — Early single-strike prototype built before extending to the parametric (K-as-input) formulation.

## Running

```bash
pip install torch numpy pandas matplotlib yfinance scipy
```

Run `PINN_code.ipynb` top-to-bottom. GPU is recommended (10,000 epochs takes ~5 min on an RTX 6000) but the notebook falls back to CPU automatically.
