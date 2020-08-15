import tensorflow as tf

print("Is there a GPU available: "),
print(tf.config.experimental.list_physical_devices("GPU"))