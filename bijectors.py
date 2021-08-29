"""
Contains a set of bijectors from:
Glow: Generative Flow with Invertible 1×1 Convolutions
Diederik P. Kingma and Prafulla Dhariwal

Also contains an assembled flow block and example of a full Glow architecture
which works well on the MNIST set
"""

import tensorflow as tf
from math import pi

class ActNorm(tf.keras.layers.Layer):
    """
    Layer performing the activation normalisation step of glow.
    Initialises parameters across first batch.
    """
      
    def __init__(self):
        super(ActNorm, self).__init__()
        self.is_init = tf.Variable(False, trainable=False)
        
    def call(self, x, direction):
        if self.is_init == False:
            self._initialise(x)            
        if direction == 'forward':
            return self._forward(x)
        elif direction == 'reverse':
            return self._reverse(x)
        else:
            raise NameError('Use "forward" or "reverse" to define direction.')
            
    def _initialise(self, x):
        std = tf.math.reduce_std(x, axis=(0,1,2), keepdims=True)
        self.s = tf.Variable(1. / (1E-5 + std))
        mean = tf.math.reduce_mean(x, axis=(0,1,2), keepdims=True)
        self.b = tf.Variable(-mean)
        self.is_init = True

    def _forward(self, x):
        x = (x + self.b) * self.s
        h, w = x.shape[1:3]
        log_det = h * w * \
            tf.math.reduce_sum(tf.math.log((tf.math.abs(self.s))))
        log_det = tf.tile(tf.convert_to_tensor([[log_det]]), [x.shape[0], 1])
        return x, log_det
        
    def _reverse(self, x):
        x = (x / self.s) - self.b
        h, w = x.shape[1:3]
        return x
    
class Invertible1x1Conv(tf.keras.layers.Layer):
    """
    Layer performing the 1x1 convolution step of glow.
    """
    
    def __init__(self):
        super(Invertible1x1Conv, self).__init__()
        
    def build(self, input_shape):
        W = tf.linalg.qr(
            tf.random.normal((input_shape[-1],input_shape[-1])))[0]
        W = tf.expand_dims(tf.expand_dims(W, 0), 0)
        self.W = tf.Variable(W, trainable=True)

    def call(self, x, direction):           
        if direction == 'forward':
            return self._forward(x)
        elif direction == 'reverse':
            return self._reverse(x)
        else:
            raise NameError('Use "forward" or "reverse" to define direction.')
    
    def _forward(self, x):
        x = tf.nn.conv2d(x, self.W, strides=1, padding="SAME")
        h, w = x.shape[1:3]
        det = tf.linalg.det(self.W)
        log_det = tf.math.log(tf.math.abs(det)) * h * w
        log_det = tf.tile(log_det, [x.shape[0],1])
        return x, log_det
        
    def _reverse(self, x):
        x = tf.nn.conv2d(x, tf.linalg.inv(self.W), strides=1, padding="SAME")
        h, w = x.shape[1:3]
        return x  
            
class Invertible1x1ConvLU(tf.keras.layers.Layer):
    """
    Layer performing the 1x1 convolution step of glow.
    Uses the LU form of W (faster determinant solving)
    """
    
    def __init__(self):
        super(Invertible1x1ConvLU, self).__init__()
        
    def build(self, input_shape):
        orthogonal = tf.linalg.qr(
            tf.random.normal((input_shape[-1],input_shape[-1])))[0]
        lu, p = tf.linalg.lu(orthogonal)
        p = tf.concat(
            (tf.expand_dims(p,1),tf.expand_dims(tf.range(p.shape),1)),1)
        sp_input = tf.SparseTensor(
            dense_shape=[p.shape[0], p.shape[0]],
            values=tf.ones(p.shape[0]),
            indices=tf.cast(p,dtype=tf.int64))
        p = tf.sparse.to_dense(tf.sparse.reorder(sp_input))
            
        self.p = tf.Variable(p, trainable=False)
        self.lu = tf.Variable(lu, trainable=True)

    def call(self, x, direction):           
        if direction == 'forward':
            return self._forward(x)
        elif direction == 'reverse':
            return self._reverse(x)
        else:
            raise NameError('Use "forward" or "reverse" to define direction.')
    
    def _get_rotation(self):
        eye = tf.eye(self.lu.shape[0])
        s = tf.linalg.band_part(self.lu, 0, 0)
        u = tf.linalg.band_part(self.lu, 0, -1) * (1 - eye)
        l = tf.linalg.band_part(self.lu, -1, 0) * (1 - eye) + eye
        W = tf.matmul(tf.matmul(self.p, l), (u + s))
        W = tf.expand_dims(tf.expand_dims(W, 0), 0)
        return W, tf.linalg.diag_part(s)
                
    def _forward(self, x):
        W, s = self._get_rotation()
        x = tf.nn.conv2d(x, W, strides=(1,1,1,1), padding="SAME")
        h, w = x.shape[1:3]
        log_det = tf.math.reduce_sum(tf.math.log(tf.math.abs(s))) * h * w
        log_det = tf.tile(tf.convert_to_tensor([[log_det]]),[x.shape[0],1])
        return x, log_det
        
    def _reverse(self, x):
        W, s = self._get_rotation()
        x = tf.nn.conv2d(x, tf.linalg.inv(W), strides=(1,1,1,1), padding="SAME")
        return x
   
class AffineCoupling(tf.keras.layers.Layer):
    """
    Layer performing the affine coupling step of glow.
    Uses as 3 layer CNN to output s and t which are hadamarded / added
    Intialising with a or b selects which half of the input is modified
    Input is split by every other channel (::2)
    Additionally has option to condition with a conditioning vector y
    """
    
    def __init__(self, num_hidden_chans=512, condition_transform=False):
        super(AffineCoupling, self).__init__()
        self.num_hidden_chans = num_hidden_chans
        self.condition_transform = condition_transform

    def build(self, input_shape):
        if self.condition_transform:
            y_dim = tf.math.reduce_prod(input_shape[1:3])
            y_dim = tf.math.sqrt(tf.cast(y_dim, dtype=tf.float32))
            self.out = tf.keras.layers.Dense(y_dim)
        self.conv1 = tf.keras.layers.Conv2D(self.num_hidden_chans, 
                                            kernel_size=3, 
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            bias_initializer='he_normal')
        self.conv2 = tf.keras.layers.Conv2D(self.num_hidden_chans, 
                                            kernel_size=1, 
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            bias_initializer='he_normal')
        self.conv3 = tf.keras.layers.Conv2D(input_shape[3], 
                                            kernel_size=3, 
                                            padding='same',
                                            kernel_initializer='zeros',
                                            bias_initializer='zeros')
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.1)

    def call(self, x, direction, y=None):
        if direction == 'forward':
            return self._forward(x, y)
        elif direction == 'reverse':
            return self._reverse(x, y)
        else:
            raise NameError('Use "forward" or "reverse" to define direction.')
            
    def _get_condition_transform(self, x, y):
        y_transform = self.out(y)
        y_transform = tf.tile(y_transform, [1, y_transform.shape[1]])
        y_transform = tf.reshape(y_transform, (-1, x.shape[1], x.shape[2], 1))
        x = tf.concat((x, y_transform), 3)
        return x
        
    def _get_weights(self, x, y):
        if self.condition_transform: x = self._get_condition_transform(x, y)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        log_s, t = tf.split(x, num_or_size_splits=2, axis=3)
        log_s = tf.clip_by_value(log_s, 1E-32, 1E32)
        s = tf.math.exp(log_s)
        return s, t
        
    def _forward(self, x, y):
        x_a, x_b = tf.split(x, num_or_size_splits=2, axis=3)
        s, t = self._get_weights(x_b, y)
        x_a = (x_a * s) + t
        x = tf.concat((x_a, x_b), 3)
        log_det = tf.expand_dims(
            tf.math.reduce_sum(tf.math.log(tf.math.abs(s)), (1,2,3)), 1)
        return x, log_det
        
    def _reverse(self, x, y):
        x_a, x_b = tf.split(x, num_or_size_splits=2, axis=3)
        s, t = self._get_weights(x_b, y)
        x_a = (x_a - t) / s
        x = tf.concat((x_a, x_b), 3)
        return x 
    
class AdditiveCoupling(tf.keras.layers.Layer):
    """
    Layer performing the affine coupling step of glow.
    Uses as 3 layer CNN to output s and t which are hadamarded / added
    Intialising with a or b selects which half of the input is modified
    Input is split by every other channel (::2)
    Additionally has option to condition with a conditioning vector y
    """
    
    def __init__(self, num_hidden_chans=512, condition_transform=False):
        super(AdditiveCoupling, self).__init__()
        self.num_hidden_chans = num_hidden_chans
        self.condition_transform = condition_transform

    def build(self, input_shape):
        if self.condition_transform:
            y_dim = tf.math.reduce_prod(input_shape[1:3])
            self.out = tf.keras.layers.Dense(y_dim)
        self.conv1 = tf.keras.layers.Conv2D(self.num_hidden_chans, 
                                            kernel_size=3, 
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            bias_initializer='he_normal')
        self.conv2 = tf.keras.layers.Conv2D(self.num_hidden_chans, 
                                            kernel_size=1, 
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            bias_initializer='he_normal')
        self.conv3 = tf.keras.layers.Conv2D(int(input_shape[3]/2), 
                                            kernel_size=3, 
                                            padding='same',
                                            kernel_initializer='zeros',
                                            bias_initializer='zeros')
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, x, direction, y=None):
        if direction == 'forward':
            return self._forward(x, y)
        elif direction == 'reverse':
            return self._reverse(x, y)
        else:
            raise NameError('Use "forward" or "reverse" to define direction.')
            
    def _get_condition_transform(self, x, y):
        y_transform = self.out(y)
        y_transform = tf.reshape(y_transform, (-1,x.shape[1],x.shape[2],1))
        x = tf.concat((x, y_transform), 3)
        return x
        
    def _get_weights(self, x, y):
        if self.condition_transform: x = self._get_condition_transform(x, y)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        return x
        
    def _forward(self, x, y):
        x_a, x_b = tf.split(x, num_or_size_splits=2, axis=3)
        t = self._get_weights(x_b, y)
        x_a = x_a + t
        x = tf.concat((x_a, x_b), 3)
        log_det = tf.zeros((x.shape[0], 1))
        return x, log_det
        
    def _reverse(self, x, y):
        x_a, x_b = tf.split(x, num_or_size_splits=2, axis=3)
        t = self._get_weights(x_b, y)
        x_a = x_a - t
        x = tf.concat((x_a, x_b), 3)
        return x 

class SqueezeExpand(tf.keras.layers.Layer):
    """
    Layer performing the spatial to channel squeeze in glow
    In the reverse direction instead expands from channels to spatial dims.
    The block size (positive integer) is the factor of this transfer
    """
    
    def __init__(self, block_size):
        super(SqueezeExpand, self).__init__()
        self.block_size = block_size
        
    def call(self, x, direction):       
        if direction == 'forward':
            x = tf.nn.space_to_depth(x, self.block_size)
        elif direction == 'reverse':
            x = tf.nn.depth_to_space(x, self.block_size)
        else:
            raise NameError('Use "forward" or "reverse" to define direction.')
        return x
        
class ExitFunction(tf.keras.layers.Layer):
    """
    Performs the channel exit and coupling functions.
    In forward mode will get normal distribution parameterised by
    a convolution operation performed on z_l - the channels staying in the 
    flow, and then get the log prob of z_i - the channels leaving the flow.
    In reverse will instead use the parameterised normal distribution to
    generate the new z_i and then concatenate this with z_l.
    If last=True will instead treat all incoming channels as z_i.
    If last=False will use half the channels as z_i - must be evenly divisible
    Additionally has option to condition with a conditioning vector y
    """

    def __init__(self, last=False, condition_transform=False):
        super(ExitFunction, self).__init__()
        self.last = last
        self.condition_transform = condition_transform
        self.term_3 = -0.5 * tf.math.log(2*pi)
        
    def build(self, input_shape):
        if self.last == True:
            num_z_i_chans = input_shape[3]
        else:
            if input_shape[3]%2 != 0:
                raise ValueError('Must be even number of channels in split.')
            num_z_i_chans = int(input_shape[3]/2)
        if self.condition_transform:
            y_dim = int(tf.math.reduce_prod(input_shape[1:3]))
            self.out = tf.keras.layers.Dense(y_dim)
        self.conv = tf.keras.layers.Conv2D(2*num_z_i_chans, 
                                           kernel_size=3, 
                                           padding='same',
                                           kernel_initializer='zeros',
                                           bias_initializer='zeros')

    def call(self, x, direction, y=None, temperature=1., test=False):
        if direction == 'forward':
            return self._forward(x, y, test)
        elif direction == 'reverse':
            return self._reverse(x, y, temperature, test)
        else:
            raise NameError('Use "forward" or "reverse" to define direction.')
            
    def _get_condition_transform(self, z_l, y):
        y_transform = self.out(y)
        y_transform = tf.reshape(y_transform, (-1,z_l.shape[1],z_l.shape[2],1))
        z_l = tf.concat((z_l, y_transform), 3)
        return z_l    
            
    def _get_params(self, z_l, y, temperature=1.):
        if self.condition_transform: z_l = self._get_condition_transform(z_l, y)
        params = self.conv(z_l)
        mean, log_std = tf.split(params, num_or_size_splits=2, axis=3)
        std = tf.math.exp(log_std) * temperature
        return mean, std
    
    def _get_log_prob_sum(self, z_i, mean, std):
        term_1 = -tf.math.log(std)
        term_2 = -0.5 * ((z_i - mean) / std)**2
        log_prob = term_1 + term_2 + self.term_3
        sum_log_prob = tf.math.reduce_sum(log_prob, (1,2,3))
        return tf.expand_dims(sum_log_prob, 1)

    def _forward(self, x, y, test):
        if self.last == True:
            z_l = tf.zeros_like(x)
            z_i = x
        else:
            z_l, z_i = tf.split(x, num_or_size_splits=2, axis=3)
        mean, std = self._get_params(z_l, y)
        log_prob_sum = self._get_log_prob_sum(z_i, mean, std)
        if test: self.retain = z_i
        return z_l, log_prob_sum
        
    def _reverse(self, x, y, temperature, test):
        mean, std = self._get_params(x, y)
        if test:
            sample = self.retain
        else:
            sample = (tf.random.normal(x.shape) * std) + mean
        if self.last == True: 
            x = sample
        else:
            x = tf.concat((x, sample), 3)
        return x

# ------------------------------------------------------------------------

class Flow(tf.keras.layers.Layer):
    """
    Keras custom layer implementing the flow steps as described in glow paper
    actnorm -> 1x1 convolution -> affine coupling
    Additionally has option to condition with a conditioning vector y
    """
    def __init__(self, num_hidden_chans=512, use_LU_inv=True, 
                 coupling_type='affine', condition_transform=False):
        super(Flow, self).__init__()
        self.condition_transform = condition_transform
        self.norm = ActNorm()
        if use_LU_inv == True:
            self.inv = Invertible1x1ConvLU()
        elif use_LU_inv == False:
            self.inv = Invertible1x1Conv()
        if coupling_type == 'additive':
            self.coupling = AdditiveCoupling(num_hidden_chans, 
                                             condition_transform)
        elif coupling_type == 'affine':
            self.coupling = AffineCoupling(num_hidden_chans,
                                           condition_transform)
        else:
            raise NameError('Use "additive" or "affine" to define coupling type.')
        
    def call(self, x, direction, y=None):  
        if direction == 'forward':
            return self._forward(x, y)
        elif direction == 'reverse':
            return self._reverse(x, y)
        else:
            raise NameError('Use "forward" or "reverse" to define direction.')

    def _forward(self, x, y):
        x, log_det_norm = self.norm(x, 'forward')
        x, log_det_inv = self.inv(x, 'forward')
        x, log_det_coupling = self.coupling(x, 'forward', y)
        return x, log_det_norm + log_det_inv + log_det_coupling
        
    def _reverse(self, x, y):
        x = self.coupling(x, 'reverse', y)
        x = self.inv(x, 'reverse')
        x = self.norm(x, 'reverse')
        return x
    
class Level(tf.keras.layers.Layer):
    """
    Keras custom layer implementing a glow level. First squeezes and 
    then runs multiple flow steps before splitting out half the channels.
    If last=True will instead exit all channels.
    """
    def __init__(self, last=False, num_flows=32, num_hidden_chans=512,
                 use_LU_inv=True, coupling_type='affine', 
                 condition_transform=False):
        super(Level, self).__init__()
        self.num_flows = num_flows
        self.last = last
        self.condition_transform = condition_transform
        self.se = SqueezeExpand(2)
        self.flows = []
        for K in range(self.num_flows):
            self.flows.append(Flow(num_hidden_chans=num_hidden_chans,
                                   use_LU_inv=use_LU_inv,
                                   coupling_type=coupling_type,
                                   condition_transform=condition_transform))
        self.exit = ExitFunction(last, condition_transform)
     
    def call(self, x, direction, y=None, temperature=1., test=False):  
        if direction == 'forward':
            return self._forward(x, y, test)
        elif direction == 'reverse':
            return self._reverse(x, y, temperature, test)
        else:
            raise NameError('Use "forward" or "reverse" to define direction.')

    def _forward(self, x, y, test):
        x = self.se(x, 'forward')
        log_det_sum = tf.zeros((x.shape[0], 1))
        for K in range(self.num_flows):
            x, log_det = self.flows[K](x, 'forward', y)
            log_det_sum = log_det_sum + log_det
        x, log_prob = self.exit(x, 'forward', y, test=test)
        return x, log_det_sum + log_prob
        
    def _reverse(self, x, y, temperature, test):
        x = self.exit(x, 'reverse', y, temperature=temperature, test=test)
        for K in range(self.num_flows-1,-1,-1):
            x = self.flows[K](x, 'reverse', y)
        x = self.se(x, 'reverse')
        return x

class GlowMNIST(tf.keras.layers.Layer):
    """
    Keras custom layer implementing an example glow architecture that works
    (reasonably) well with MNIST (padded to 32x32x1)
    In forward pass will also calculate NLL in bits per dimension
    Additionally has option to condition with a conditioning vector y
    which operates on one-hot encoded digit values
    """
    def __init__(self, num_levels=5, num_flows_per_level=32, 
                 num_hidden_chans=512, use_LU_inv=True, 
                 coupling_type='affine', condition_transform=False):
        super(GlowMNIST, self).__init__()
        self.condition_transform = condition_transform
        self.num_levels = num_levels
        self.levels = []
        last = False
        for L in range(self.num_levels):
            if L == self.num_levels-1: last = True
            self.levels.append(Level(last=last,
                                     num_flows=num_flows_per_level,
                                     num_hidden_chans=num_hidden_chans,
                                     use_LU_inv=use_LU_inv,
                                     coupling_type=coupling_type,
                                     condition_transform=condition_transform))
        
    def call(self, x, direction, y=None, temperature=0.7, test=False):
        if direction == 'forward':
            return self._forward(x, y, test)
        elif direction == 'reverse':
            return self._reverse(x, y, temperature, test)
        else:
            raise NameError('Use "forward" or "reverse" to define direction.')
        
    def _forward(self, x, y, test):
        LL_sum = tf.zeros((x.shape[0], 1))
        dim = tf.cast(tf.math.reduce_prod(x.shape[1:]), dtype=x.dtype)
        for L in range(self.num_levels):
            x, log_likelihood = self.levels[L](x, 'forward', y, test=test)
            LL_sum = LL_sum + log_likelihood
        c = dim * tf.math.log(1./256.)
        NLL_BPD = (-tf.math.reduce_mean(LL_sum) - c) / (dim * tf.math.log(2.))
        return x, NLL_BPD
        
    def _reverse(self, x, y, temperature, test):
        t = 1
        for L in range(self.num_levels-1,-1,-1):
            if L == self.num_levels-1: t = temperature
            x = self.levels[L](x, 'reverse', y, temperature=t, test=test)
        return x