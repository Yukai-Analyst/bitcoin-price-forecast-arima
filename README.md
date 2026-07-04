# 比特币价格走势预测 - ARIMA 时间序列分析

基于 ARIMA 模型的比特币（BTC）月度价格预测项目，使用 Python 完成完整的检验步骤与建模流程。

## 项目简介

本项目对比特币从 2014 年 9 月至 2026 年 5 月的历史价格数据进行分析，通过时间序列建模方法（ARIMA）预测未来价格走势。

主要内容包括：
- 数据加载与预处理（日期索引、重采样为月度均值）
- 平稳性检验（ADF 检验）
- 白噪声检验（Ljung-Box 检验）
- ARIMA(p, d, q) 模型参数选择（AIC 准则）
- 模型拟合与残差诊断
- 未来 8 期价格预测
- 可视化分析与结果展示

## 文件说明

| 文件 | 描述 |
|------|------|
| `Bitcoin_Price_ARIMA_TimeSeries_Forecast_Report.ipynb` | 完整的分析 Jupyter Notebook |
| `bitcoin_btc.csv` | 比特币历史价格原始数据 |

## 环境要求

- Python 3.8+
- Jupyter Notebook

## 依赖库

```
numpy
pandas
matplotlib
statsmodels
```

## 使用方法

1. 将 `bitcoin_btc.csv` 放在项目根目录
2. 打开并运行 `Bitcoin_Price_ARIMA_TimeSeries_Forecast_Report.ipynb`
3. 依次执行各单元格以完成数据预处理、模型检验、拟合与预测

## 技术栈

- **数据处理**: pandas
- **时间序列分析**: statsmodels (ARIMA, ADF, Ljung-Box)
- **可视化**: matplotlib
