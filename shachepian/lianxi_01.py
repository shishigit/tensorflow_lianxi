from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

xunlian_wenjianjia = '/home/c/Desktop/shachepian_shuju/20200612'
ceshi_wenjianjia = '/home/c/Desktop/shachepian_shuju/ceshi'

xunlianshuju = image.DirectoryIterator(
    directory=xunlian_wenjianjia,
    image_data_generator=ImageDataGenerator(data_format='channels_first', dtype='float64'),
    target_size=(200, 200)
)

print(xunlianshuju.image_shape)

moxing = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=xunlianshuju.image_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(7)
    ])

moxing.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

moxing.fit(x=xunlianshuju, epochs=1)

ceshi_shuju = image.DirectoryIterator(
    directory=ceshi_wenjianjia,
    image_data_generator=ImageDataGenerator(data_format='channels_first', dtype='float64'),
    target_size=(200, 200),
)

test_loss, test_acc = moxing.evaluate(
    x=ceshi_shuju,
    verbose=2
)

print('\nTest accuracy:', test_acc)
