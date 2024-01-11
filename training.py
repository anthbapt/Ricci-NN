import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import os
os.chdir("/home/anthbapt/Documents/fMNIST_DNN_training/wk")

# Read data
x_test = pd.read_csv("fashion-mnist_test.csv")
y_test = x_test['label']
x_test = x_test.iloc[:, 1:]

x_train = pd.read_csv("fashion-mnist_train.csv")
y_train = x_train['label']
x_train = x_train.iloc[:, 1:]


# Restrict to labels 5 and 9
labels_1_7 = [5, 9]
train_1_7 = np.concatenate([np.where(y_train == label)[0] for label in labels_1_7])
test_1_7 = np.concatenate([np.where(y_test == label)[0] for label in labels_1_7])

y_train = y_train.iloc[train_1_7].values
y_test = y_test.iloc[test_1_7].values

y_test[y_test == 5] = 0
y_test[y_test == 9] = 1

y_train[y_train == 5] = 0
y_train[y_train == 9] = 1

x_train = np.array(x_train.iloc[train_1_7, :])
x_test = np.array(x_test.iloc[test_1_7, :])

# Print dimensions
print("Dimensions of x_train:", x_train.shape)
print("Dimensions of x_test:", x_test.shape)


b = 3
accuracy = list()
model_predict = np.empty(b, dtype = object)

for j in range(b):
    # Define DNN architecturex_test.shapex_test.shape
    model = Sequential()
    model.add(Dense(units=50, activation='relu', input_shape=(x_test.shape[1],)))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # Binary cross-entropy loss function for all models
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    # Train model on training data
    dnn_history = model.fit(x_train, y_train,
                            epochs=50, batch_size=32,
                            validation_split=0.2)

    # Check accuracy on test data
    acc = model.evaluate(x_test, y_test)[1]
    accuracy.append(acc)

    # Output the layers on implementation over test data
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activation_model.save("activation_model"+ str(j) + ".h5")
    model_predict[j] = activation_model.predict(x_test)


np.save("model_predict.npy", model_predict)
np.save("accuracy.npy", accuracy)
pd.DataFrame(x_test).to_csv("x_test.csv", index=False, header = None)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False, header = None)