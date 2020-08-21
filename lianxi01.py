import os

import tensorflow


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

        self.__shengcheng_zidian()

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
            print(dangqianwenjianjia)
            for wenjian in wenjians:
                linshiwenjian = tensorflow.io.read_file(os.path.join(dangqianwenjianjia, wenjian))
                lsbiaoqian = dangqianwenjianjia.split('/')[-1]
                yield tensorflow.image.decode_image(
                    contents=linshiwenjian,
                    dtype=tensorflow.dtypes.float32,
                ), self.__onehot[lsbiaoqian]

    def huoqu_shujuji(self):
        """
        获取数据集
        """
        return tensorflow.data.Dataset.from_generator(
            generator=self.__tupian_generator,
            output_types=(tensorflow.dtypes.float32, tensorflow.dtypes.float32)
        )


shujuchuli = Shujuchuli(Canshu.wenjianjia_xunlian)
ls = shujuchuli.huoqu_shujuji()
print(ls.take(1).as_numpy_iterator().next())
#############################################################################################
# 构建模型
#############################################################################################

#############################################################################################
# 训练
#############################################################################################

#############################################################################################
# 测试
#############################################################################################
