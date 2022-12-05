from utils import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l2

print('===============================================================')
print('Old Data:\n')
# Train
train_datfile = r"data/simu_20000_0.1_90_140.npy"
train_data = np.load(train_datfile)
# Test
test_datfile = r"data/simu_10000_0.1_140_180.npy"
test_data = np.load(test_datfile)

# print('===============================================================')
# print('New Data:\n')
# # Train
# train_datfile = r"data/day_data_train.npy"
# train_data = np.load(train_datfile)
# # Test
# test_datfile = r"data/day_data_test.npy"
# test_data = np.load(test_datfile)

print('train_data ', train_data.shape)
print('test_data ', test_data.shape)
print('===============================================================')
print('Data: sensor data (100Hz * 10 seconds) + ID + Time + H + R + S + D \n')
print('===============================================================')
print('Data Prep:\n')
train_x = train_data[:, :-6]
train_y = train_data[:, -2:]
print('train_x', train_x.shape)
print('train_y', train_y.shape)

test_x = test_data[:, :-6]
test_y = test_data[:, -2:]
print('test_x', test_x.shape)
print('test_y', test_y.shape)

# Preprocessing X
X_train = xdata_preprocess(train_x)
X_test = xdata_preprocess(test_x)

# Preprocessing y
y_train, mean1, std1 = ydata_preprocess(train_y)
y_test, mean, std = ydata_preprocess(test_y)
print('===============================================================')
print('Reshaped data:\n')
# Reshaping data for CNN
sample_size = X_train.shape[0]  # number of samples in train set
time_steps = X_train.shape[1]  # number of features in train set
input_dimension = 1  # each feature is represented by 1 number

train_data_reshaped = X_train.reshape(sample_size, time_steps, input_dimension)
print("After reshape train data set shape:", train_data_reshaped.shape)
print("1 Sample shape:", train_data_reshaped[0].shape)

test_data_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], input_dimension)
print("After reshape test data set shape:", test_data_reshaped.shape)
print("1 Sample shape:", test_data_reshaped[0].shape)

n_timesteps = train_data_reshaped.shape[1]
n_features = train_data_reshaped.shape[2]
print('===============================================================\n')


def build_multi_conv1D_model(n_timesteps, n_features):
    model = keras.Sequential(name="model_conv1D")
    model.add(keras.layers.Input(shape=(n_timesteps, n_features)))

    model.add(keras.layers.Conv1D(32, 1, padding="valid", activation="relu", strides=1, name="Conv1D_1"))
    model.add(keras.layers.Conv1D(32, 3, padding="valid", activation="relu", strides=1, name="Conv1D_2"))
    model.add(keras.layers.Conv1D(32, 5, padding="valid", activation="relu", strides=1,
                                  name="Conv1D_3"))  # kernel_regularizer=l2(0.001)))
    model.add(keras.layers.GlobalMaxPooling1D(name="MaxPooling1D"))
    model.add(keras.layers.Dense(32, activation="relu", name="Dense_1"))
    model.add(keras.layers.Dense(32, activation="relu", name="Dense_2"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2, name="Dense_3"))

    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model


model_conv1D = build_multi_conv1D_model(n_timesteps, n_features)
model_conv1D.summary()

# Training step
EPOCHS = 200
history = model_conv1D.fit(train_data_reshaped, y_train, epochs=EPOCHS,
                           validation_split=0.2, verbose=1)

# Plot of train and validation losses
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.plot(loss_values, 'bo', label='training loss')
plt.plot(val_loss_values, 'r', label='training loss val')
plt.legend()
plt.show()

# Prediction
test_predictions = model_conv1D.predict(test_data_reshaped)
# Reverse scaling the prediction
rev_test_predictions = ydata_revpreprocess(test_predictions, mean, std)
# Calculating MAE
mae_tst = mean_absolute_error(test_y[:, 0], rev_test_predictions[:, 0])
print('\n===============================================================')
print('Systolic MAE on test data using CNN model: ', mae_tst)
print('===============================================================')

# Uncomment to save any new model
# model_conv1D.save('cnn_model.h5')
