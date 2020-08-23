from kerastuner.applications import HyperResNet
from kerastuner.tuners import Hyperband
from tensorflow.keras.preprocessing import image


class canshu:
    wenjianjia_xunlian = '/home/c/Desktop/CODE/shachepian_shuju/20200612'
    wenjianjia_ceshi = '/home/c/Desktop/CODE/shachepian_shuju/ceshi'

    @staticmethod
    def huoqushujuji(wenjianjia):
        return image.DirectoryIterator(
            directory=wenjianjia,
            image_data_generator=image.ImageDataGenerator(
                data_format='channels_first',
                dtype='float64'
            )
        )


xunlian_shuju = canshu.huoqushujuji(canshu.wenjianjia_xunlian)
ceshi_shuju = canshu.huoqushujuji(canshu.wenjianjia_ceshi)

hypermodel = HyperResNet(
    include_top=True,
    input_shape=xunlian_shuju.image_shape,
    classes=xunlian_shuju.num_classes
)

tuner = Hyperband(
    hypermodel,
    objective='val_accuracy',
    directory='my_dir',
    project_name='helloworld',
    max_epochs=15
)

tuner.search(
    x=xunlian_shuju,
    epochs=15,
    validation_data=ceshi_shuju
)
