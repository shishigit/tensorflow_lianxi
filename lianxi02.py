import kerastuner
from tensorflow import keras
from tensorflow.keras.preprocessing import image


#############################################################################################
# 参数
#############################################################################################
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


#############################################################################################
# 训练
#############################################################################################

xunlian_shuju = canshu.huoqushujuji(canshu.wenjianjia_xunlian)
qianyimoxing = keras.applications.MobileNetV2(
    input_shape=xunlian_shuju.image_shape,
    include_top=False,
    weights='imagenet',
)
qianyimoxing.trainable = False


def shengchengmoxing(hp):
    shuru = keras.layers.Input(shape=xunlian_shuju.image_shape)
    x = keras.layers.experimental.preprocessing.Rescaling(1. / 127.5)(shuru)
    x = x - 1
    x = keras.layers.experimental.preprocessing.RandomFlip()(x)
    x = keras.layers.experimental.preprocessing.RandomRotation(0.1)(x)
    x = qianyimoxing(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    x = keras.layers.Dense(hp_units, activation=keras.activations.relu)(x)
    x = keras.layers.Dropout(0.2)(x)
    shuchu = keras.layers.Dense(xunlian_shuju.num_classes, activation='softmax')(x)

    moxing = keras.Model(shuru, shuchu)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    moxing.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return moxing


tuner = kerastuner.Hyperband(
    shengchengmoxing,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='intro_to_kt'
)

ceshi_shuju = canshu.huoqushujuji(canshu.wenjianjia_ceshi)

tuner.search(
    xunlian_shuju,
    epochs=10,
    validation_data=ceshi_shuju
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print('units:{}'.format(best_hps.get('units')))
print('learning_rate:{}'.format(best_hps.get('learning_rate')))

model = tuner.hypermodel.build(best_hps)
model.fit(xunlian_shuju, epochs=10, validation_data=ceshi_shuju)
