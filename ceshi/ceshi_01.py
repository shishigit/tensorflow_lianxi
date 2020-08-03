from tensorflow.python.keras.layers import Flatten

yaping = Flatten()
yaping([1, 1])
print(yaping)
#
# model = Sequential()
# model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
# # now: model.output_shape == (None, 64, 32, 32)
# model.add(Flatten())
# # now: model.output_shape == (None, 65536)
