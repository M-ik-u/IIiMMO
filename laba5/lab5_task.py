import numpy as np
import tensorflow as tf
from keras.src.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt

from data_generation import generate_data

#generate_data()

X = np.loadtxt("laba5/dataIn.txt")
y = np.loadtxt("laba5/dataOut.txt")

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = keras.Sequential([
    keras.layers.Input(shape=(12,)),
    keras.layers.Dense(16,activation= "sigmoid"),
    keras.layers.Dense(2,activation= "softmax"),
])

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=1
)

y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_truth = np.argmax(y_test, axis=1)


print(f"Acuracy: {accuracy_score(y_truth,y_pred)}")

plt.plot(history.history['loss'], label="training loss")
plt.plot(history.history['val_loss'], label="validation loss")

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.savefig("laba5/loss.png")


print(confusion_matrix(y_truth, y_pred))

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("Y_train:", y_train.shape, "Y_test:", y_test.shape)
print("Train acc last epoch:", history.history['accuracy'][-1])
print("Val acc last epoch:", history.history['val_accuracy'][-1])