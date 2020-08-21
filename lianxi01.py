import os
from typing import Union

import tensorflow
#############################################################################################
# 参数
#############################################################################################
from tensorflow.python.data.ops.dataset_ops import DatasetV1, DatasetV2


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
        self.shujuji: Union[DatasetV1, DatasetV2] = Union
        self.shape = None

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
                lsbiaoqian = dangqianwenjianjia.split('/')[-1]
                yield tensorflow.image.decode_bmp(contents=linshiwenjian), self.__onehot[lsbiaoqian]

    def shengcheng_shujuji(self):
        """
        获取数据集
        """
        self.shujuji = tensorflow.data.Dataset.from_generator(
            generator=self.__tupian_generator,
            output_types=(tensorflow.dtypes.float32, tensorflow.dtypes.float32),
        ).batch(30)


shujuchuli = Shujuchuli(Canshu.wenjianjia_xunlian)
shujuchuli.shengcheng_shujuji()
ls = shujuchuli.shujuji.as_numpy_iterator().next()
#############################################################################################
# 构建模型
#############################################################################################

moxing = tensorflow.keras.Sequential()
moxing.add(tensorflow.keras.layers.Input(shujuchuli.shujuji))
moxing.add(tensorflow.keras.layers.experimental.preprocessing.RandomFlip('horizontal'))
moxing.add(tensorflow.keras.layers.experimental.preprocessing.RandomRotation(0.2))
moxing.add(tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'))
moxing.add(tensorflow.keras.layers.MaxPooling2D((2, 2)))
moxing.add(tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu'))
moxing.add(tensorflow.keras.layers.MaxPooling2D((2, 2)))
moxing.add(tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu'))
moxing.add(tensorflow.keras.layers.Flatten())
moxing.add(tensorflow.keras.layers.Dense(64, activation='relu'))
moxing.add(tensorflow.keras.layers.Dense(shujuchuli.yangben_zhongleishu()))

moxing.summary()

moxing.compile(
    optimizer='adam',
    loss=tensorflow.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

moxing.fit(
    x=shujuchuli.shujuji,
    epochs=1,
    verbose=1,
)

#############################################################################################
# 训练
#############################################################################################

#############################################################################################
# 测试
#############################################################################################
