import tensorflow as tf
import numpy as np

# 建立一個簡單的深度學習模型
# 注意：這是一個範例模型，您應該使用您自己訓練好的模型
# 輸入特徵維度為 5 (對應 extract_features 的輸出)
# 輸出維度為 3 (莊、閒、和)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 使用 softmax 輸出機率
])

# 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 產生一些假的訓練資料來讓模型可以儲存
# 在實際應用中，您需要用真實資料來訓練
X_train = np.random.rand(10, 5)
y_train = np.random.randint(0, 3, 10)
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=3)

# 簡單訓練一下
model.fit(X_train, y_train_one_hot, epochs=1, verbose=0)

# 儲存模型
model.save("dl_model.h5")

print("模型 'dl_model.h5' 已成功建立。")
