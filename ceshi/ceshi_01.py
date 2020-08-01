from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Convolution2D, Flatten

model = Sequential()
model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)
model.add(Flatten())
# now: model.output_shape == (None, 65536)
