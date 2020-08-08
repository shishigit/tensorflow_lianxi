from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

#############################################################################################
# 参数
#############################################################################################

xunlian_wenjianjia = '/home/c/Desktop/shachepian_shuju/20200612'
ceshi_wenjianjia = '/home/c/Desktop/shachepian_shuju/ceshi'

#############################################################################################
# 训练
#############################################################################################

xunlian_shuju = image.DirectoryIterator(
    directory=xunlian_wenjianjia,
    image_data_generator=ImageDataGenerator(data_format='channels_first', dtype='float64'),
    target_size=(200, 200)
)

print(xunlian_shuju.image_shape)
print(len(xunlian_shuju))

moxing = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=xunlian_shuju.image_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(7)
    ])

moxing.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

for shuju in xunlian_shuju.next():
    moxing.fit(x=xunlian_shuju, epochs=1)

#############################################################################################
# 测试
#############################################################################################

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
