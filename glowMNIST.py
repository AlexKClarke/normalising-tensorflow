import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from bijectors import GlowMNIST
from MNIST import get_MNIST_data

class Model(tf.keras.Model):
    def __init__(self, condition_transform=False):
        super(Model, self).__init__()
        tf.keras.backend.set_floatx('float32')
        self.condition_transform = condition_transform
        self.adam = tf.keras.optimizers.Adam(learning_rate=1E-5)
        self.data_check = False
        self.init_check = False
        
        self.num_init_samples = 512
        self.num_test_samples = 32
        
        
    def input_data(self, train_set, val_set, test_set):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.data_check = True
        
    def generate(self, num_samples=10, temperature=0.7, increment=False):
        z = tf.tile(self.z, [num_samples, 1, 1, 1])
        if self.condition_transform:
            if increment:
                y = tf.range(10, dtype=tf.int32)
                z = tf.tile(self.z, [10, 1, 1, 1])
            else:
                y = tf.random.uniform(shape=(num_samples, 1), 
                                      minval=0, 
                                      maxval=10, 
                                      dtype=tf.int32)
            y = tf.cast(tf.one_hot(tf.squeeze(y), 10), dtype=z.dtype)
        else:
            y = None
        x = self.glow(z, 'reverse', y=y, temperature=temperature)
        x = tf.clip_by_value(x, -0.5, 0.5)
        x = tf.math.floor((x + 0.5) * 256)
        return x.numpy()
    
    def test_bijectivity(self):
        for batch in self.train_set.batch(self.num_test_samples): break
        z, loss = self.glow(batch[0], 'forward', batch[1], test=True)
        x = self.glow(z, 'reverse', batch[1], test=True)
        difference = tf.math.reduce_mean(tf.math.abs(batch[0] - x))
        return np.round(difference.numpy(), 3)
    
    def plot_examples(self, temperature=0.3):
        examples = self.generate(increment=True, temperature=temperature)
        plt.figure(figsize=(20,8))
        for digit in range(10):
            plt.subplot(2, 5, digit+1)
            plt.imshow(examples[digit,4:-4,4:-4,0])
            plt.title(str(digit))
        plt.show()

    def fit(self, batch_size=256, num_epochs=1000):#
        if self.data_check is False: raise ValueError('No data in model.')
        if self.init_check is False: self._initialise()
        self._run_fit(batch_size, num_epochs)
    
    def _run_fit(self, batch_size, num_epochs):
        epoch = 0
        template_1 = 'Epoch {}: NLL_BPD = {}'
        template_2 = 'Epoch {}: NLL_BPD = {}, Bijectivity Difference = {}'
        for batch in self.train_set.batch(batch_size).prefetch(1):
            if epoch > num_epochs: break
            epoch = epoch + 1
            self._train_step(batch)
            train_accuracy = np.round(self.train_accuracy.result().numpy(), 3)
            if epoch%50 == 0: 
                difference = self.test_bijectivity()
                print(template_2.format(epoch, train_accuracy, difference))
                self.plot_examples()
            else:
                print(template_1.format(epoch, train_accuracy))
    
    def _initialise(self):
        self.glow = GlowMNIST(num_levels=5, 
                              num_flows_per_level=32, 
                              num_hidden_chans=512, 
                              use_LU_inv=True, 
                              coupling_type='affine', 
                              condition_transform=self.condition_transform)
        self.train_accuracy = tf.keras.metrics.Sum()
        for batch in self.train_set.batch(self.num_init_samples): break
        z, loss = self.glow(batch[0], 'forward', batch[1])
        self.z = tf.zeros_like(z[0:1,:,:,:])
        self.init_check = True
    
    @tf.function
    def _train_step(self, batch):
        with tf.GradientTape() as tape:
            z, NLL_BPD = self.glow(batch[0], 'forward', batch[1])
        gradients = tape.gradient(NLL_BPD, self.trainable_variables)
        gradients = [(tf.clip_by_value(grad, -1., 1.)) for grad in gradients]
        self.adam.apply_gradients(zip(gradients, self.trainable_variables))    
        self.train_accuracy.reset_state()
        self.train_accuracy.update_state(NLL_BPD)
        
#Test code:
if __name__ == "__main__":
    batch_size = 32
    num_epochs = 5000
    condition_transform = True
    
    train_set, val_set, test_set = get_MNIST_data()
    model = Model(condition_transform)
    model.input_data(train_set, val_set, test_set)
    model.fit(batch_size, num_epochs)
    model.plot_examples(temperature=0.7)

        
