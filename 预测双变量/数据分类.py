import pandas as pd
from sklearn.model_selection import train_test_split

# 读取csv全部数据
df = pd.read_csv("data.csv")

# 按 80% 训练，20% 测试划分
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 保存训练集和测试集
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print(f"训练集样本数: {len(train_df)}")
print(f"测试集样本数: {len(test_df)}")
print("训练集和测试集文件已保存:train_data.csv, test_data.csv")