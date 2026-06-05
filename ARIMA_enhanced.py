# -*- coding: utf-8 -*-
# 比特币走势预测 - 完整检验步骤的 ARIMA 模型
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from itertools import product
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_arch
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = r"D:\2026上学期\商业数据分析Python\结课作业\结课作业\final\outputs"
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("比特币价格走势分析 ARIMA 建模 - 完整流程")
print("=" * 70)

# ======================== 1. 数据加载与预处理 ========================
file_path = r"D:\2026上学期\商业数据分析Python\结课作业\结课作业\final\bitcoin_btc.csv"
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

start_date = '2014-09-17'
end_date = '2026-05-19'
df = df.loc[start_date:end_date]
print(f"\n数据时间范围：{df.index.min().strftime('%Y-%m-%d')} 至 {df.index.max().strftime('%Y-%m-%d')}")
print(f"原始日度数据行数：{len(df)}")

# 重采样为月度平均值
df_month = df.resample('ME').mean()
df_month.rename(columns={'price': 'Price'}, inplace=True)
print(f"月度数据点数：{len(df_month)}")

# ======================== 描述性统计 ========================
print("\n" + "=" * 70)
print("描述性统计")
print("=" * 70)
desc = df_month['Price'].describe()
print(desc)
cv = desc['std'] / desc['mean']  # 变异系数
print(f"\n变异系数 (CV): {cv:.4f}")
from scipy.stats import skew, kurtosis
print(f"偏度 (Skewness): {skew(df_month['Price'].dropna()):.4f}")
print(f"峰度 (Kurtosis): {kurtosis(df_month['Price'].dropna()):.4f}")

# ======================== 2. 图检验 ========================
print("\n正在生成图1：原始月度价格序列 - 时序图...")

# ---- 图1：原始序列时序图 ----
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df_month.index, df_month['Price'], color='#1f77b4', linewidth=1.5)
ax1.set_title('图1  原始月度价格序列 — 时序图（2014年9月—2026年5月）', fontsize=14, fontweight='bold')
ax1.set_xlabel('日期', fontsize=12)
ax1.set_ylabel('价格（美元）', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=10)
fig1.tight_layout()
fig1.savefig(os.path.join(output_dir, '图1_原始序列时序图.png'), dpi=300, bbox_inches='tight')
plt.close(fig1)
print("  [OK] 图1 已保存")

# ---- 图2：原始序列自相关图 (ACF) ----
print("正在生成图2：原始月度价格序列 - 自相关图...")
fig2, ax2 = plt.subplots(figsize=(12, 5))
plot_acf(df_month['Price'].dropna(), lags=24, alpha=0.05, ax=ax2)
ax2.set_title('图2  原始月度价格序列 — 自相关图 (ACF, lag=1—24)', fontsize=14, fontweight='bold')
ax2.set_xlabel('滞后阶数', fontsize=12)
ax2.set_ylabel('自相关系数', fontsize=12)
ax2.tick_params(labelsize=10)
fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, '图2_原始序列自相关图.png'), dpi=300, bbox_inches='tight')
plt.close(fig2)
print("  [OK] 图2 已保存")

# ======================== 3. 平稳性检验 (ADF) ========================
def check_stationarity(series, series_name):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"\n{series_name} 的 ADF 检验结果：")
    print(f"  ADF 统计量: {result[0]:.6f}")
    print(f"  p 值: {result[1]:.6f}")
    print(f"  临界值: 1%={result[4]['1%']:.4f}, 5%={result[4]['5%']:.4f}, 10%={result[4]['10%']:.4f}")
    if result[1] <= 0.05:
        print(f"  结论：在5%显著性水平下拒绝单位根原假设，序列平稳")
        return True, 0, result
    else:
        print(f"  结论：不能拒绝单位根原假设，序列不平稳")
        return False, 1, result

# 检验原始月度价格
is_stationary, d, result_orig = check_stationarity(df_month['Price'], "原始月度价格")

# 一阶差分
df_month['Price_diff'] = df_month['Price'].diff()
is_stationary_diff, _, result_diff = check_stationarity(df_month['Price_diff'], "一阶差分后序列")

if not is_stationary_diff:
    print("警告：一阶差分后仍不平稳，尝试二阶差分")
    d = 2
    df_month['Price_diff2'] = df_month['Price'].diff().diff()
else:
    print("\n一阶差分后序列平稳，确定差分阶数 d = 1")

diff_series = df_month['Price_diff'].dropna()

# ---- 图3：一阶差分后序列时序图 ----
print("\n正在生成图3：一阶差分后序列 - 时序图...")
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(diff_series.index, diff_series, color='#d62728', linewidth=1.2)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax3.set_title('图3  一阶差分后序列 — 时序图', fontsize=14, fontweight='bold')
ax3.set_xlabel('日期', fontsize=12)
ax3.set_ylabel('价格差分（美元）', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=10)
fig3.tight_layout()
fig3.savefig(os.path.join(output_dir, '图3_差分后序列时序图.png'), dpi=300, bbox_inches='tight')
plt.close(fig3)
print("  [OK] 图3 已保存")

# ---- 图4：一阶差分后序列自相关图 (ACF) ----
print("正在生成图4：一阶差分后序列 - 自相关图...")
fig4, ax4 = plt.subplots(figsize=(12, 5))
plot_acf(diff_series, lags=24, alpha=0.05, ax=ax4)
ax4.set_title('图4  一阶差分后序列 — 自相关图 (ACF, lag=1—24)', fontsize=14, fontweight='bold')
ax4.set_xlabel('滞后阶数', fontsize=12)
ax4.set_ylabel('自相关系数', fontsize=12)
ax4.tick_params(labelsize=10)
fig4.tight_layout()
fig4.savefig(os.path.join(output_dir, '图4_差分后序列自相关图.png'), dpi=300, bbox_inches='tight')
plt.close(fig4)
print("  [OK] 图4 已保存")

# ======================== 4. 白噪声检验 (Ljung-Box) ========================
print("\n" + "=" * 70)
print("白噪声检验 (Ljung-Box) — 表1")
print("=" * 70)
series_for_model = df_month['Price_diff'].dropna()
lb_test = acorr_ljungbox(series_for_model, lags=12, return_df=True)
print("\n表1  一阶差分后序列的Ljung-Box白噪声检验结果（滞后1—12阶）")
print("-" * 60)
print(f"{'滞后阶数':<10} {'Q统计量':<15} {'p值':<15} {'结论'}")
print("-" * 60)
for idx, row in lb_test.iterrows():
    lag = idx if isinstance(idx, int) else int(idx)
    q_stat = row['lb_stat'] if 'lb_stat' in row else row.iloc[0]
    p_val = row['lb_pvalue'] if 'lb_pvalue' in row else row.iloc[1]
    conclusion = "显著（非白噪声）" if p_val < 0.05 else "不显著"
    print(f"{lag:<10} {q_stat:<15.4f} {p_val:<15.6f} {conclusion}")
print("-" * 60)
if (lb_test['lb_pvalue'] < 0.05).any():
    print("结论：序列存在显著自相关，非白噪声，适合 ARMA 建模 [OK]")
else:
    print("警告：序列接近白噪声，ARMA 模型可能无效")

# 保存表1为CSV
table1_df = pd.DataFrame({
    '滞后阶数': [f'lag {i}' for i in range(1, 13)],
    'Q统计量': [f'{lb_test.iloc[i-1, 0]:.4f}' for i in range(1, 13)],
    'p值': [f'{lb_test.iloc[i-1, 1]:.6f}' for i in range(1, 13)],
    '结论': ['显著（非白噪声）' if lb_test.iloc[i-1, 1] < 0.05 else '不显著' for i in range(1, 13)]
})
table1_df.to_csv(os.path.join(output_dir, '表1_LjungBox白噪声检验.csv'), index=False, encoding='utf-8-sig')

# ======================== 5. 最优 ARIMA(p,d,q) 模型选择（AIC准则） ========================
print("\n" + "=" * 70)
print("网格搜索最优 ARIMA(p,d,q) — 表2")
print("=" * 70)
ps = range(0, 4)
qs = range(0, 4)
best_aic = float('inf')
best_order = None
best_model = None
results = []

print("\n开始网格搜索最优 ARIMA(p,d,q)...")
for p, q in product(ps, qs):
    try:
        model = ARIMA(df_month['Price'], order=(p, d, q)).fit()
        aic = model.aic
        results.append(((p, d, q), aic))
        if aic < best_aic:
            best_aic = aic
            best_order = (p, d, q)
            best_model = model
    except Exception as e:
        continue

print("\n表2  不同ARIMA(p,1,q)候选模型的AIC值（p, q ∈ [0,3]）")
print("-" * 65)
print(f"{'模型':<20} {'AIC':<15} {'模型':<20} {'AIC'}")
print("-" * 65)
sorted_results = sorted(results, key=lambda x: x[1])
for i in range(0, len(sorted_results), 2):
    model1 = f"ARIMA({sorted_results[i][0][0]},{sorted_results[i][0][1]},{sorted_results[i][0][2]})"
    line = f"{model1:<20} {sorted_results[i][1]:<15.2f}"
    if i + 1 < len(sorted_results):
        model2 = f"ARIMA({sorted_results[i+1][0][0]},{sorted_results[i+1][0][1]},{sorted_results[i+1][0][2]})"
        line += f"{model2:<20} {sorted_results[i+1][1]:<15.2f}"
    print(line)
print("-" * 65)
print(f"\n最优模型: ARIMA{best_order}, AIC = {best_aic:.2f}")

# 保存表2
table2_data = []
for (p_val, d_val, q_val), aic_val in results:
    table2_data.append({
        '模型': f'ARIMA({p_val},{d_val},{q_val})',
        'AIC': f'{aic_val:.2f}',
        '对数似然值': '',
        'BIC': ''
    })
table2_df = pd.DataFrame(table2_data)
table2_df = table2_df.sort_values('AIC')
table2_df.to_csv(os.path.join(output_dir, '表2_AIC网格搜索结果.csv'), index=False, encoding='utf-8-sig')

# ======================== 6. 模型参数估计 — 表3 ========================
print("\n" + "=" * 70)
print("ARIMA(2,1,3) 模型参数估计 — 表3")
print("=" * 70)
print(best_model.summary())

# 提取参数表
print("\n表3  ARIMA(2,1,3)模型的系数估计结果与参数显著性检验")
print("-" * 90)
print(f"{'参数':<15} {'系数估计值':<15} {'标准误':<15} {'z统计量':<15} {'p值':<15} {'显著性'}")
print("-" * 90)
for name in best_model.params.index:
    coef = best_model.params[name]
    se = best_model.bse[name]
    z_val = coef / se
    from scipy.stats import norm
    p_val = 2 * (1 - norm.cdf(abs(z_val)))
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    param_name = name.replace('.L1', ' lag1').replace('.L2', ' lag2').replace('.L3', ' lag3')
    print(f"{param_name:<15} {coef:<15.6f} {se:<15.6f} {z_val:<15.6f} {p_val:<15.6f} {sig}")
print("-" * 90)
print("注：*** 表示在 0.1% 水平显著，** 表示在 1% 水平显著，* 表示在 5% 水平显著，ns 表示不显著")
print(f"\n对数似然值: {best_model.llf:.2f}")
print(f"AIC: {best_model.aic:.2f}")
print(f"BIC: {best_model.bic:.2f}")
print(f"HQIC: {best_model.hqic:.2f}")

# 保存表3
table3_data = []
for name in best_model.params.index:
    coef = best_model.params[name]
    se = best_model.bse[name]
    z_val = coef / se
    from scipy.stats import norm
    p_val = 2 * (1 - norm.cdf(abs(z_val)))
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    table3_data.append({
        '参数': name,
        '系数估计值': f'{coef:.6f}',
        '标准误': f'{se:.6f}',
        'z统计量': f'{z_val:.6f}',
        'p值': f'{p_val:.6f}',
        '显著性': sig
    })
table3_df = pd.DataFrame(table3_data)
table3_df.to_csv(os.path.join(output_dir, '表3_模型系数估计.csv'), index=False, encoding='utf-8-sig')

# ======================== 7. 模型诊断 ========================
print("\n" + "=" * 70)
print("模型诊断")
print("=" * 70)

resid = best_model.resid

# ---------- 残差白噪声检验 — 表4 ----------
print("\n表4  ARIMA(2,1,3)模型残差的Ljung-Box检验结果（滞后1—10阶）")
lb_resid = acorr_ljungbox(resid, lags=10, return_df=True)
print("-" * 60)
print(f"{'滞后阶数':<10} {'Q统计量':<15} {'p值':<15} {'结论'}")
print("-" * 60)
for idx, row in lb_resid.iterrows():
    lag = idx if isinstance(idx, int) else int(idx)
    q_stat = row['lb_stat'] if 'lb_stat' in row else row.iloc[0]
    p_val = row['lb_pvalue'] if 'lb_pvalue' in row else row.iloc[1]
    conclusion = "白噪声 ✓" if p_val > 0.05 else "非白噪声 ✗"
    print(f"{lag:<10} {q_stat:<15.4f} {p_val:<15.6f} {conclusion}")
print("-" * 60)
if (lb_resid['lb_pvalue'] > 0.05).all():
    print("[PASS] 结论：残差序列为白噪声，模型整体显著有效")
else:
    print("[FAIL] 结论：残差序列不是白噪声，模型需要重新定阶")

# 保存表4
table4_data = []
for i in range(10):
    p_val = lb_resid.iloc[i, 1]
    table4_data.append({
        '滞后阶数': f'{i+1}',
        'Q统计量': f'{lb_resid.iloc[i, 0]:.4f}',
        'p值': f'{p_val:.6f}',
        '结论': '白噪声（通过）' if p_val > 0.05 else '非白噪声（未通过）'
    })
table4_df = pd.DataFrame(table4_data)
table4_df.to_csv(os.path.join(output_dir, '表4_残差白噪声检验.csv'), index=False, encoding='utf-8-sig')

# ---------- 残差正态性检验 ----------
print("\n--- 残差正态性检验 (Jarque-Bera) ---")
jb_stat, jb_p = jarque_bera(resid.dropna())
from scipy.stats import skew as sk_func, kurtosis as kt_func
resid_skew = sk_func(resid.dropna())
resid_kurt = kt_func(resid.dropna())
print(f"偏度 (Skewness): {resid_skew:.4f}")
print(f"峰度 (Kurtosis): {resid_kurt:.4f} (超额峰度: {resid_kurt - 3:.4f})")
print(f"Jarque-Bera 统计量: {jb_stat:.4f}")
print(f"p值: {jb_p:.6f}")
if jb_p < 0.05:
    print("结论：拒绝正态分布原假设，残差呈非正态分布（尖峰厚尾）")

# ---------- ARCH效应检验 ----------
print("\n--- ARCH效应检验 ---")
arch_result = het_arch(resid.dropna())
arch_lm = arch_result[0]
arch_p = arch_result[1]
print(f"ARCH LM 统计量: {arch_lm:.4f}")
print(f"p值: {arch_p:.6f}")
if arch_p < 0.05:
    print("结论：拒绝同方差原假设，残差存在显著的条件异方差（ARCH效应）")

# ======================== 8. 预测 — 表5 ========================
print("\n" + "=" * 70)
print("未来8个月预测 — 表5")
print("=" * 70)
last_date = df_month.index[-1]
future_dates = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=8, freq='ME')
forecast = best_model.forecast(steps=8)
forecast_series = pd.Series(forecast, index=future_dates, name='Forecast')

# 获取预测标准误
forecast_result = best_model.get_forecast(steps=8)
forecast_se = forecast_result.se_mean

print("\n表5  基于ARIMA(2,1,3)模型的比特币月度价格分层预测结果（单位：美元）")
print("-" * 100)
print(f"{'月份':<15} {'预测价格':<15} {'预测标准误':<15} {'预测层次':<20} {'说明'}")
print("-" * 100)
for i, (date, price, se) in enumerate(zip(future_dates, forecast, forecast_se)):
    date_str = date.strftime('%Y-%m')
    h = i + 1
    if h <= 3:
        layer = '核心预测期 (h≤q=3)'
        note = 'MA+AR结构充分驱动'
    else:
        layer = '趋势参考期 (h>q=3)'
        note = '仅AR均值回归驱动'
    print(f"{date_str:<15} {price:<15.2f} {se:<15.2f} {layer:<20} {note}")
print("-" * 100)
print("注：预测标准误用于构造预测区间。95%预测区间 ≈ 预测值 ± 1.96 × 标准误")

# 保存表5
table5_data = []
for i, (date, price, se) in enumerate(zip(future_dates, forecast, forecast_se)):
    h = i + 1
    table5_data.append({
        '月份': date.strftime('%Y-%m'),
        '预测步长h': h,
        '预测价格(美元)': f'{price:.2f}',
        '标准误': f'{se:.2f}',
        '预测层次': '核心预测期' if h <= 3 else '趋势参考期',
        '驱动机制': 'MA+AR联合驱动' if h <= 3 else 'AR均值回归驱动'
    })
table5_df = pd.DataFrame(table5_data)
table5_df.to_csv(os.path.join(output_dir, '表5_分层预测结果.csv'), index=False, encoding='utf-8-sig')

# 同时输出为Excel便于插入论文
with pd.ExcelWriter(os.path.join(output_dir, '所有表格汇总.xlsx'), engine='openpyxl') as writer:
    table1_df.to_excel(writer, sheet_name='表1_LjungBox检验', index=False)
    table2_df.to_excel(writer, sheet_name='表2_AIC网格搜索', index=False)
    table3_df.to_excel(writer, sheet_name='表3_模型系数估计', index=False)
    table4_df.to_excel(writer, sheet_name='表4_残差白噪声检验', index=False)
    table5_df.to_excel(writer, sheet_name='表5_分层预测结果', index=False)
print("\n所有表格已汇总保存至: outputs/所有表格汇总.xlsx")

# ======================== 9. 综合预测图 — 图5 ========================
print("\n正在生成图5：比特币价格走势及未来8个月预测...")
fig5, ax5 = plt.subplots(figsize=(14, 7))

# 历史价格
ax5.plot(df_month.index, df_month['Price'], color='#1f77b4', linewidth=2, label='历史实际价格（月度均值）')

# 预测价格
ax5.plot(forecast_series.index, forecast_series, 'r--', linewidth=2.5, marker='o', markersize=8, label='预测价格')

# 预测区间
forecast_ci = forecast_result.conf_int(alpha=0.05)
ax5.fill_between(forecast_series.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                 color='red', alpha=0.1, label='95% 预测置信区间')

# 分层标注
ax5.axvline(x=forecast_series.index[0], color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
ax5.axvline(x=forecast_series.index[2], color='orange', linestyle='--', linewidth=1.2, alpha=0.7)

# 标注核心预测期和趋势参考期
mid_core = forecast_series.index[0] + (forecast_series.index[2] - forecast_series.index[0]) / 2
mid_trend = forecast_series.index[3] + (forecast_series.index[-1] - forecast_series.index[3]) / 2

ax5.annotate('核心预测期\n(h ≤ q = 3)', xy=(forecast_series.index[1], forecast_series.iloc[1]),
             xytext=(mid_core, df_month['Price'].max() * 0.95),
             fontsize=11, ha='center', color='darkred', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

ax5.annotate('趋势参考期\n(h > q = 3)', xy=(forecast_series.index[5], forecast_series.iloc[5]),
             xytext=(mid_trend, df_month['Price'].max() * 0.85),
             fontsize=11, ha='center', color='darkblue', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))

# 标注数值
for i, (date, price) in enumerate(zip(forecast_series.index, forecast_series)):
    ax5.annotate(f'${price:,.0f}', xy=(date, price),
                 xytext=(0, 15), textcoords='offset points',
                 fontsize=8, ha='center', color='darkred', fontweight='bold')

ax5.set_title(f'图5  比特币月度价格走势及未来8个月预测（最优模型 ARIMA{best_order}）', fontsize=14, fontweight='bold')
ax5.set_xlabel('日期', fontsize=12)
ax5.set_ylabel('价格（美元）', fontsize=12)
ax5.grid(True, alpha=0.3)
ax5.legend(loc='upper left', fontsize=10)
ax5.tick_params(labelsize=10)
fig5.tight_layout()
fig5.savefig(os.path.join(output_dir, '图5_价格预测图.png'), dpi=300, bbox_inches='tight')
plt.close(fig5)
print("  [OK] 图5 已保存")

# ======================== 10. 残差诊断组合图 — 图6 ========================
print("正在生成图6：残差诊断组合图...")
fig6, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: 残差时序图
axes[0, 0].plot(resid.index, resid, color='#2ca02c', linewidth=0.8)
axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0, 0].set_title('(a) 残差时序图', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('日期')
axes[0, 0].set_ylabel('残差')
axes[0, 0].grid(True, alpha=0.3)

# 子图2: 残差ACF图
plot_acf(resid.dropna(), lags=20, alpha=0.05, ax=axes[0, 1])
axes[0, 1].set_title('(b) 残差自相关图 (ACF)', fontsize=12, fontweight='bold')

# 子图3: 残差Q-Q图
from scipy.stats import probplot
probplot(resid.dropna(), dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('(c) 残差Q-Q图（正态性检验）', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 子图4: 残差直方图
axes[1, 1].hist(resid.dropna(), bins=30, density=True, color='#ff7f0e', edgecolor='white', alpha=0.7)
from scipy.stats import norm as norm_dist
xmin, xmax = axes[1, 1].get_xlim()
x_range = np.linspace(xmin, xmax, 100)
axes[1, 1].plot(x_range, norm_dist.pdf(x_range, resid.mean(), resid.std()),
                'r-', linewidth=2, label='正态分布参考')
axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
axes[1, 1].set_title(f'(d) 残差分布直方图（偏度={resid_skew:.2f}, 峰度={resid_kurt:.2f}）', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('残差')
axes[1, 1].set_ylabel('密度')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

fig6.suptitle('图6  ARIMA(2,1,3)模型残差诊断组合图', fontsize=14, fontweight='bold', y=1.01)
fig6.tight_layout()
fig6.savefig(os.path.join(output_dir, '图6_残差诊断组合图.png'), dpi=300, bbox_inches='tight')
plt.close(fig6)
print("  [OK] 图6 已保存")

# ======================== 总结 ========================
print("\n" + "=" * 70)
print("所有图表生成完毕！")
print(f"输出目录：{output_dir}")
print("=" * 70)
print("\n生成的文件列表：")
for f in sorted(os.listdir(output_dir)):
    fpath = os.path.join(output_dir, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {f} ({size_kb:.1f} KB)")
