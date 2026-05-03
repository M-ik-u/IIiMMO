import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X_train, y_train,epochs=50,batch_size=16,validation_data=(X_test,y_test))


y_pred = (model.predict(X_test) > 0.5).astype(int)

print(f"Acuracy: {accuracy_score(y_test,y_pred)}")

plt.plot(history.history['loss'], label="training loss")
plt.plot(history.history['val_loss'], label="validation loss")

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(f"logistic regression accuracy: {accuracy_score(y_test,y_pred)}")

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"random forest accuracy: {accuracy_score(y_test,y_pred_rf)}")
