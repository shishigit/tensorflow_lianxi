from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


#############################################################################################
# 参数
#############################################################################################
class canshu:
    wenjianjia_xunlian = '/home/c/Desktop/CODE/shachepian_shuju/20200612'
    wenjianjia_ceshi = '/home/c/Desktop/CODE/shachepian_shuju/ceshi'


#############################################################################################
# 训练数据
#############################################################################################

xunlian_shuju = image.DirectoryIterator(
    directory=canshu.wenjianjia_xunlian,
    image_data_generator=ImageDataGenerator(
        data_format='channels_first',
        dtype='float64'
    )
)

print('------ 数据信息 ----------------------------------------------------------------------')
print("xunlian_shuju.image_shape:  {}".format(xunlian_shuju.image_shape))
print("len(xunlian_shuju):  {}".format(len(xunlian_shuju)))
print("xunlian_shuju.batch_size:  {}".format(xunlian_shuju.batch_size))
print("xunlian_shuju.filenames:  ".format(xunlian_shuju.n))
print('----------------------------------------------------------------------------')

#############################################################################################
# 训练
#############################################################################################
qianyimoxing = keras.applications.MobileNetV2(
    input_shape=xunlian_shuju.image_shape,
    include_top=False,
    weights='imagenet',
)
qianyimoxing.trainable = False

shuru = keras.layers.Input(shape=xunlian_shuju.image_shape)
x = keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)(shuru)
x = keras.layers.experimental.preprocessing.RandomFlip('horizontal')(x)
x = keras.layers.experimental.preprocessing.RandomRotation(0.1)(x)
x = qianyimoxing(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(100, activation=keras.activations.relu)(x)
x = keras.layers.Dropout(0.2)(x)
shuchu = keras.layers.Dense(xunlian_shuju.num_classes, activation='softmax')(x)

moxing = keras.Model(shuru, shuchu)

moxing.summary()

moxing.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

moxing.fit(
    x=xunlian_shuju,
    epochs=15,
)

#############################################################################################
# 测试
#############################################################################################

ceshi_shuju = image.DirectoryIterator(
    directory=canshu.wenjianjia_ceshi,
    image_data_generator=ImageDataGenerator(data_format='channels_first', dtype='float64'),
)

moxing.evaluate(
    x=ceshi_shuju,
    verbose=2
)
