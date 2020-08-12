import datetime

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

#############################################################################################
# 参数
#############################################################################################

xunlian_wenjianjia = '/home/c/Desktop/shachepian_shuju/20200612'
ceshi_wenjianjia = '/home/c/Desktop/shachepian_shuju/ceshi'

#############################################################################################
# 训练数据
#############################################################################################

xunlian_shuju = image.DirectoryIterator(
    directory=xunlian_wenjianjia,
    image_data_generator=ImageDataGenerator(
        data_format='channels_first',
        dtype='float64'
    )
)

print('------ 数据信息 ----------------------------------------------------------------------')
print("xunlian_shuju.image_shape:  {}".format(xunlian_shuju.image_shape))
print("len(xunlian_shuju):  {}".format(len(xunlian_shuju)))
print("xunlian_shuju.batch_size:  {}".format(xunlian_shuju.batch_size))
print("xunlian_shuju.n:  {}".format(xunlian_shuju.n))
print('----------------------------------------------------------------------------')

#############################################################################################
# 训练
#############################################################################################

moxing = keras.applications.MobileNetV2(input_shape=xunlian_shuju.image_shape,
                                        include_top=False,
                                        weights='imagenet')
moxing.trainable = False

moxing.summary()

moxing.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# moxing.fit(
#     x=xunlian_shuju,
#     epochs=1,
#     verbose=1,
#     callbacks=[tensorboard_callback]
# )
#
# #############################################################################################
# # 测试
# #############################################################################################
#
# ceshi_shuju = image.DirectoryIterator(
#     directory=ceshi_wenjianjia,
#     image_data_generator=ImageDataGenerator(data_format='channels_first', dtype='float64'),
# )
#
# test_loss, test_acc = moxing.evaluate(
#     x=ceshi_shuju,
#     verbose=2
# )
#
# print('\nTest accuracy:', test_acc)
