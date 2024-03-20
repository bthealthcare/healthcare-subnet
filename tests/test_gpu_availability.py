import tensorflow as tf

devices = tf.config.list_physical_devices()
print(devices)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs available:", gpus)
else:
    print("No GPUs were found.")

# Perform a simple computation on a tensor
tensor = tf.random.uniform([1000, 1000])
result = tf.matmul(tensor, tf.transpose(tensor))
print(result)