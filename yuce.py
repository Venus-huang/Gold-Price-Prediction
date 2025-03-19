import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

# 加载数据
file_path = 'gold_cpi_interest_rate_daily.csv'
data = pd.read_csv(file_path)

# 将日期转换为时间格式并设置为索引
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
data = data.set_index('Date')

# 确保数据是每日连续的
data = data.resample('D').ffill()

# 定义预测结束日期
end_date = '2024-12-31'

# 定义一个函数用于ADF平稳性检验
def test_stationarity(series):
    result = adfuller(series)
    return result[1]  # 返回 p 值

# 用于存储预测结果
optimized_forecasts = {}

# 对每列数据进行ARIMA建模和预测
for column in ['Gold Price', 'CPI-U', 'Real Interest Rate']:
    # 检测是否平稳
    p_value = test_stationarity(data[column])
    d = 0 if p_value < 0.05 else 1  # 如果数据平稳，差分阶数 d = 0；否则 d = 1
    
    # 使用 auto_arima 自动选择最优参数
    model = auto_arima(
        data[column],
        seasonal=False,  # 不考虑季节性
        d=d,  # 手动设置差分阶数
        stepwise=True,  # 使用逐步搜索
        trace=True,  # 输出搜索过程
        suppress_warnings=True,
        error_action="ignore",
    )
    
    # 创建预测日期范围
    forecast_days = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), end=end_date, freq='D')
    
    # 进行预测
    forecast_values = model.predict(n_periods=len(forecast_days))
    
    # 存储预测结果
    optimized_forecasts[column] = pd.Series(forecast_values, index=forecast_days)

# 将预测结果合并为DataFrame
optimized_forecast_df = pd.concat(optimized_forecasts, axis=1)

# 保存预测结果到CSV
optimized_forecast_df.to_csv('Optimized_Forecast_to_2024.csv')

print("预测完成！预测结果已保存为 'Optimized_Forecast_to_2024.csv'")
