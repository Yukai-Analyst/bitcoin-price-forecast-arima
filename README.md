# Bitcoin Price ARIMA Modeling — Time Series Analysis

> A complete Box-Jenkins ARIMA(p,d,q) modeling workflow applied to Bitcoin monthly price data, implemented in R with automated report generation.

## 📊 Overview

This project demonstrates a full time-series analysis pipeline on Bitcoin daily closing prices:

1. **Daily → Monthly** resampling (mean aggregation)
2. **Stationarity testing** — visual (time series plot, ACF), ADF unit-root test (3 types via `aTSA`)
3. **Differencing** — 1st & 2nd order, with ADF confirmation → **d = 1**
4. **White-noise test** — Ljung-Box at lags 6 and 12
5. **Model identification** — ACF/PACF of differenced series
6. **Grid search** over 16 ARIMA(p,1,q) models (p,q ∈ {0,1,2,3}) with AIC, AICc, BIC
7. **Model diagnostics** — residual white-noise test (`tsdiag`), parameter t-tests
8. **Sparse coefficient models** — constrain insignificant parameters to zero via `fixed`
9. **Model selection** — AIC criterion → best model: **ARIMA(2,1,3)**
10. **Forecast** — 3-month ahead point predictions with 80% and 95% confidence intervals

### Key Findings (from the original analysis)

| Item | Result |
|------|--------|
| Data | 4,250 daily observations → 141 monthly means (Sep 2014 – May 2026) |
| Differencing order | d = 1 |
| Best model | ARIMA(2,1,3), AIC = 2779.65, BIC = 2797.35 |
| All parameters significant | Yes (p < 0.0001 for all 5 coefficients) |
| Residual white noise | Passed (LB lag 6: p = 0.5983, lag 12: p = 0.3009) |
| 3-month forecast direction | Downward trend (~$90,208 → ~$86,003) |

## 📁 Project Structure

```
├── bitcoin_btc.csv          # Raw daily Bitcoin closing prices (Kaggle dataset)
├── bitcoin_arima_modeling.R # Complete R modeling code (Box-Jenkins pipeline)
├── generate_report.py       # Python script to generate the final Word report
├── figures/                 # Auto-generated PNG charts (created by R code)
│   ├── fig1_1_original_time_series.png
│   ├── fig1_2_original_acf.png
│   ├── fig2_1_diff1_time_series.png
│   ├── fig2_2_diff1_acf.png
│   ├── fig2_3_diff2_time_series.png
│   ├── fig2_4_diff2_acf.png
│   ├── fig3_1_diff1_acf.png        # ACF + PACF (2-panel)
│   ├── tsdiag_ARIMA*.png           # Residual diagnostics for all 16 models
│   └── fig8_1_forecast.png         # Historical + forecast plot
├── 比特币价格ARIMA建模结课报告.docx  # Final report (Chinese, ready-to-submit)
├── 应用时间序列分析课程论文范文.docx # Reference paper (style guide)
├── bitcoin_report_text.txt          # Report text content
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites

- **R ≥ 4.4** with packages: `readr`, `tseries`, `aTSA`, `forecast`
- **Python ≥ 3.10** with packages: `python-docx`, `Pillow`

### Step 1: Run the R modeling code

Open `bitcoin_arima_modeling.R` in RStudio and run all lines (Ctrl+Alt+R), or from the command line:

```bash
Rscript bitcoin_arima_modeling.R
```

This will:
- Read `bitcoin_btc.csv` and resample to monthly averages
- Perform all stationarity and white-noise tests
- Fit 16 ARIMA(p,1,q) models and evaluate them
- Identify the optimal model
- Generate forecasts and save all charts to `figures/`

### Step 2: Generate the Word report

```bash
python generate_report.py
```

This produces `比特币价格ARIMA建模结课报告.docx` — a complete Chinese-language academic report with:
- Cover page, abstract, keywords
- 8 chapters aligned with the Box-Jenkins methodology
- All embedded charts from `figures/`
- Formatted tables with actual test statistics
- Full R code in the appendix

## 📖 Methodology

The project follows the standard **Box-Jenkins** ARIMA modeling framework:

```
Data → Visual Inspection → Stationarity Test → Differencing
   → White-Noise Test → Model Identification (ACF/PACF)
   → Parameter Estimation (grid search) → Model Diagnostics
   → Sparse Models → Model Selection (AIC) → Forecasting
```

Each step is fully documented in the R code with inline Chinese comments.

## 🔧 Customization

### Use your own data

Replace `bitcoin_btc.csv` with a CSV containing two columns:

| date | price |
|------|-------|
| YYYY/MM/DD | numeric |

Then update the path on line 17 of `bitcoin_arima_modeling.R`:

```r
raw_data <- read_csv("your_data.csv")
```

### Adjust model grid

Change `p_vals` and `q_vals` (line 155–156) to expand or shrink the search space:

```r
p_vals <- 0:2   # search p = 0, 1, 2
q_vals <- 0:2   # search q = 0, 1, 2
```

### Forecast horizon

Change `h = 3` (line 383) to forecast more or fewer periods.

## 📝 License

This project is released for educational purposes. The Bitcoin price data comes from Kaggle. Feel free to adapt and reuse.
