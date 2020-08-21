import os
from typing import Any

import tensorflow
from tensorflow.python.data.ops.dataset_ops import BatchDataset


#############################################################################################
# 参数
#############################################################################################


class Canshu:
    wenjianjia_xunlian = '/home/c/Desktop/CODE/shachepian_shuju/20200612'
    wenjianjia_ceshi = '/home/c/Desktop/CODE/shachepian_shuju/ceshi'


#############################################################################################
# 数据处理
#############################################################################################

class Shujuchuli:

    def __init__(self, wenjianjia):

        self.__wenjianjia: str = wenjianjia
        self.__onehot: dict = {}
        self.shujuji: BatchDataset = Any
        self.__zero = tensorflow.zeros([2052, 2592, 1])

        self.__shengcheng_zidian()

    def yangben_zhongleishu(self):
        return len(self.__onehot)

    def __shengcheng_zidian(self):

        wenjianjia_set = set()

        for dangqianwenjianjia, wenjianjias, wenjianmings in os.walk(self.__wenjianjia):
            for ziwenjianjia in wenjianjias:
                wenjianjia_set.add(ziwenjianjia)

        lsonehot = tensorflow.one_hot(range(len(wenjianjia_set)), depth=len(wenjianjia_set), dtype=tensorflow.float32)

        for index, wenjianjia in enumerate(wenjianjia_set):
            self.__onehot[wenjianjia] = lsonehot[index]

    def __tupian_generator(self):
        for dangqianwenjianjia, wenjianjias, wenjians in os.walk(self.__wenjianjia):
            for wenjian in wenjians:
                linshiwenjian = tensorflow.io.read_file(os.path.join(dangqianwenjianjia, wenjian))
                linshiwenjian = tensorflow.image.decode_image(contents=linshiwenjian, dtype=tensorflow.float32)
                linshiwenjian = tensorflow.concat([linshiwenjian, self.__zero, self.__zero], axis=2)
                lsbiaoqian = dangqianwenjianjia.split('/')[-1]
                yield linshiwenjian, self.__onehot[lsbiaoqian]

    def shengcheng_shujuji(self):
        """
        获取数据集
        """
        self.shujuji = tensorflow.data.Dataset.from_generator(
            generator=self.__tupian_generator,
            output_types=(tensorflow.dtypes.float32, tensorflow.dtypes.float32),
            output_shapes=((2052, 2592, 3), 7),
        )
        self.shujuji = self.shujuji.batch(batch_size=3)


shujuchuli = Shujuchuli(Canshu.wenjianjia_xunlian)
shujuchuli.shengcheng_shujuji()

#############################################################################################
# 构建模型
#############################################################################################

#############################################################################################
# 训练
#############################################################################################
base_model = tensorflow.keras.applications.MobileNetV2(
    input_shape=(2052, 2592, 3),
    include_top=False,
    weights='imagenet',
)
base_model.trainable = False

shuru = tensorflow.keras.layers.Input(shape=(2052, 2592, 3))
x = tensorflow.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)(shuru)
x = tensorflow.keras.layers.experimental.preprocessing.RandomFlip('horizontal')(x)
x = tensorflow.keras.layers.experimental.preprocessing.RandomRotation(1)(x)
x = base_model(x, training=False)
x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
x = tensorflow.keras.layers.Dense(100, activation=tensorflow.keras.activations.relu)(x)
x = tensorflow.keras.layers.Dropout(0.2)(x)
shuchu = tensorflow.keras.layers.Dense(7, activation='softmax')(x)

moxing = tensorflow.keras.Model(shuru, shuchu)

moxing.summary()

moxing.compile(
    optimizer='adam',
    loss=tensorflow.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

moxing.fit(
    x=shujuchuli.shujuji,
    epochs=15,
)

#############################################################################################
# 测试
#############################################################################################

shujuchuli = Shujuchuli(Canshu.wenjianjia_ceshi)
shujuchuli.shengcheng_shujuji()

test_loss, test_acc = moxing.evaluate(
    x=shujuchuli.shujuji,
    verbose=2
)

print('\nTest accuracy:', test_acc)

#############################################################################################
# 测试
#############################################################################################
