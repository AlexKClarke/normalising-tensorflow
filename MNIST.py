import tensorflow_datasets as tfds
import tensorflow as tf

def get_MNIST_data():
    (train_set,test_set),ds_info = tfds.load(
        'mnist',
        split=['train','test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    def norm_func(data1,data2): 
                data1 = tf.pad(data1, ((2, 2), (2, 2), (0, 0)))
                #dequantise
                data1 = tf.cast(data1, dtype=tf.float32) + \
                    tf.random.uniform(data1.shape, minval=0, maxval=1)
                data1 = (data1 / 256) - 0.5
                data2 = tf.one_hot(data2, 10)
                return data1,data2 
    train_set = train_set.map(norm_func).repeat()
    for data in test_set.map(norm_func).batch(512):
        break
    val_set = data
    for data in test_set.map(norm_func).batch(2048):
        break
    test_set = data
    
    return train_set, val_set, test_set

