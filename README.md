# bitcoin-price-forecast-arima
Bitcoin Price Time Series Analysis & Short-Term Forecast | ARIMA Empirical Research

## 1. Project Introduction
This is an independent full-process time series data analysis project based on Python, taking Bitcoin daily closing price data from 2014 to 2026 on Kaggle as the research sample.
Following the classic Box-Jenkins modeling specification, I completed the whole analysis link from raw data cleaning, statistical modeling, model diagnosis to out-of-sample short-term price prediction independently.
Combined with the finite memory property of MA(q) process, the prediction interval is split into core valid short-term forecast period and trend reference period, avoiding the common over-prediction defect of traditional ARIMA model in financial time series.
All codes and analysis results in this repository are fully reproducible, and the project is completed for data analysis post resume practice.

## 2. Technical Stack
### Data Processing & Visualization
Python(Pandas / Matplotlib)：Data filtering, date sequence sorting, abnormal value elimination, time-series trend chart, ACF autocorrelation diagram drawing
### Statistical Inspection
ADF Unit Root Test、Ljung-Box White Noise Test、ARCH LM Heteroscedasticity Test
### Core Model
ARIMA(p,d,q) Time Series Prediction Model, select optimal hyperparameters via AIC minimum criterion

## 3. Repository File Description
| File Name | File Purpose |
| --- | --- |
| `ARIMA_enhanced.py` | Core analysis code, including full process of data preprocessing, stationary verification, model training, parameter selection, residual diagnosis and result visualization |
| `bitcoin_btc.csv` | Original Kaggle Bitcoin daily quotation dataset, covering 2014-2026 4250+ trading day closing price data |
| `Bitcoin_Price_ARIMA_TimeSeries_Forecast_Report.docx` | Full empirical analysis report, including detailed research background, methodology, model derivation, result analysis and conclusion |

## 4. Core Implementation Flow
1. **Data Preprocessing**: Process 4250+ daily trading data, sort date sequence, complete monthly resampling, eliminate abnormal jump data, convert into standardized monthly price analysis series
2. **Stationary Verification**: Combine time series curve, ACF autocorrelation diagram and ADF test, confirm primary difference(d=1) makes the sequence stationary under 5% significance level
3. **Model Confirmation**: Traverse p/q hyperparameter combinations, pick ARIMA(2,1,3) with minimum AIC value as the final optimal prediction model
4. **Model Diagnosis**: Verify residual follows white noise distribution via Ljung-Box test, check parameter significance and residual heteroscedasticity to ensure model validity
5. **Short-term Prediction**: Divide prediction window, output 3-month core effective predicted price and long-term trend reference range

## 5. Subsequent Optimization Plan
1. Introduce GARCH model to fix conditional heteroscedasticity of financial residual data and improve prediction precision
2. Build LSTM deep learning model, compare forecast performance difference between statistical model and machine learning model
3. Add macro influencing factor data to realize multi-feature multivariate prediction

## 6. Author Info
Independent personal project | Applied Statistics, Guangxi University of Finance and Economics
Target Position: Data Analyst / Business Data Analysis Specialist
