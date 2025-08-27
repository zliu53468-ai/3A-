import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# 建立一個簡單的統計模型 (邏輯迴歸)
# 注意：這是一個範例模型，您應該使用您自己訓練好的模型
model = LogisticRegression()

# 產生一些假的訓練資料
# 輸入特徵維度為 5
# 輸出為 0, 1, 2 (代表莊、閒、和)
X_train = np.random.rand(10, 5)
y_train = np.random.randint(0, 3, 10)

# 訓練模型
model.fit(X_train, y_train)

# 儲存模型
joblib.dump(model, "stat_model.pkl")

print("模型 'stat_model.pkl' 已成功建立。")
