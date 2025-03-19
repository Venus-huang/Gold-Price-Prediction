import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# 加载训练阶段保存的 scaler 和模型
from joblib import load

# 加载 scaler
scaler_input = load('scaler_input.pkl')
scaler_target = load('scaler_target.pkl')

# 加载保存的 LSTM 模型
model = load_model('optimized_gold_price_lstm_model.keras')

# Step 1: 从 CSV 文件读取 ARIMA 预测特征数据
arima_predictions = pd.read_csv('Optimized_Forecast_to_2024.csv')

# 查看数据
print("ARIMA 预测特征数据：")
print(arima_predictions.head())

# 提取特征数据并进行归一化
# 确保特征与训练阶段保持一致
future_features = arima_predictions[['Real Interest Rate', 'CPI-U', 'Gold Price']].values
future_features_scaled = scaler_input.transform(future_features)  # 使用输入特征的 scaler 进行归一化

# 调整形状以适应 LSTM 输入格式
future_features_scaled = future_features_scaled.reshape((future_features_scaled.shape[0], future_features_scaled.shape[1], 1))

# 使用 LSTM 模型预测黄金价格
future_gold_price_predictions = model.predict(future_features_scaled)

# 反归一化预测结果
future_gold_price_predictions = scaler_target.inverse_transform(future_gold_price_predictions)

# 将日期与预测结果组合
future_predictions_df = pd.DataFrame({
    'Date': arima_predictions['Date'],
    'Predicted Gold Price': future_gold_price_predictions.flatten()
})

# 打印未来黄金价格预测结果
print("未来黄金价格预测：")
print(future_predictions_df)

# 保存预测结果为 CSV 文件
future_predictions_df.to_csv('future_gold_price_predictions.csv', index=False)
