import os
import pandas as pd
import numpy as np
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from joblib import dump, load


# 设置环境变量（可选）：禁用 GPU 或限制使用
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 使用 CPU 运行，若需要 GPU 删除此行

# Step 1: 加载数据集
file_path = '/root/autodl-fs/gold_cpi_interest_rate_daily.csv'  # 修改为正确的文件路径
data = pd.read_csv(file_path)

# 检查数据基本信息
print("数据预览：")
print(data.head())
print(f"数据维度: {data.shape}")

# Step 2: 数据预处理
# 确保时间序列排序
data = data.sort_values(by='Date')
data.reset_index(drop=True, inplace=True)

# 提取目标特征和输入变量
target = data['Gold Price']
input_variables = data[['Real Interest Rate', 'CPI-U', 'Gold Price']]

# 数据归一化
scaler_input = MinMaxScaler()
scaler_target = MinMaxScaler()

scaled_inputs = scaler_input.fit_transform(input_variables)
scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    scaled_inputs, scaled_target, test_size=0.2, shuffle=False
)

# 调整形状以适配 LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Step 3: 构建模型
def build_model(hp):
    model = Sequential()
    # LSTM 层
    model.add(LSTM(
        units=hp.Int('units', min_value=32, max_value=128, step=16),
        input_shape=(X_train.shape[1], X_train.shape[2]),
        activation='relu'
    ))
    # Dense 层
    model.add(Dense(
        units=hp.Int('dense_units', min_value=16, max_value=64, step=16),
        activation='relu'
    ))
    model.add(Dense(1))  # 输出层
    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: 超参数调优
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=50,
    factor=3,
    directory='hyperband_tuning',
    project_name='gold_price_prediction',
    max_consecutive_failed_trials=10  # 防止因连续失败中断
)

# Step 5: 进行超参数搜索
try:
    tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)
except Exception as e:
    print(f"Tuning failed with error: {e}")

# Step 6: 获取最佳模型
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
最佳超参数配置：
- LSTM 单元数: {best_hps.get('units')}
- Dense 层单元数: {best_hps.get('dense_units')}
""")

# 构建最佳模型
model = tuner.hypermodel.build(best_hps)

# 训练最佳模型
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

# Step 7: 保存模型
model.save('optimized_gold_price_lstm_model.keras')

# Step 8: 评估模型
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"训练集损失: {train_loss}")
print(f"测试集损失: {test_loss}")

dump(scaler_input, 'scaler_input.pkl')
dump(scaler_target, 'scaler_target.pkl')

# Step 9: 预测并反归一化
train_predictions = scaler_target.inverse_transform(model.predict(X_train))
test_predictions = scaler_target.inverse_transform(model.predict(X_test))
actual_train = scaler_target.inverse_transform(y_train)
actual_test = scaler_target.inverse_transform(y_test)

# 保存结果到 CSV 文件
train_comparison = pd.DataFrame({
    'Actual': actual_train.flatten(),
    'Predicted': train_predictions.flatten()
})
test_comparison = pd.DataFrame({
    'Actual': actual_test.flatten(),
    'Predicted': test_predictions.flatten()
})

train_comparison.to_csv('train_actual_vs_predicted.csv', index=False)
test_comparison.to_csv('test_actual_vs_predicted.csv', index=False)

print("训练与测试结果已保存！")
