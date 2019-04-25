import tensorflow as tf


class Dense():
    """Fully-connected layer"""
    def __init__(self, scope="dense_layer", size=None, dropout=1.,
                 nonlinearity=tf.identity):
        # (str, int, (float | tf.Tensor), tf.op)
        assert size, "Must specify layer size (num nodes)"
        self.scope = scope
        self.size = size
        self.dropout = dropout # keep_prob
        self.nonlinearity = nonlinearity

    def __call__(self, x):
        """Dense layer currying, to apply layer to any input tensor `x`"""
        # tf.Tensor -> tf.Tensor
        with tf.name_scope(self.scope):
            while True:
                try: # reuse weights if already initialized
                    return self.nonlinearity(tf.matmul(x, self.w) + self.b)
                except(AttributeError):
                    self.w, self.b = self.wbVars(x.get_shape()[1].value, self.size)
                    self.w = tf.nn.dropout(self.w, self.dropout)

    @staticmethod
    def wbVars(fan_in: int, fan_out: int):
        """Helper to initialize weights and biases, via He's adaptation
        of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
        """
        # (int, int) -> (tf.Variable, tf.Variable)
        stddev = tf.cast((2 / fan_in)**0.5, tf.float32)

        initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
        initial_b = tf.zeros([fan_out])

        return (tf.Variable(initial_w, trainable=True, name="weights"),
                tf.Variable(initial_b, trainable=True, name="biases"))


def conv_encoder(input_tensor,IMG_DIM):
    '''Create encoder network.
    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]
    Returns:
        A tensor that expresses the encoder network
    '''
    net = tf.reshape(input_tensor, [-1, IMG_DIM, IMG_DIM, 3])
    net = tf.contrib.layers.conv2d(net, 32, 5, stride=1, padding='VALID')
    net = tf.contrib.layers.conv2d(net, 64, 5, stride=1, padding='VALID')
    net = tf.contrib.layers.conv2d(net, 128, 5, stride=1,padding='VALID')
    #net = layers.dropout(net, keep_prob=0.9)
    net = tf.contrib.layers.flatten(net)
    
    return net

def conv_decoder(input_tensor):
    #net = tf.expand_dims(input_tensor, 1)
    #net = tf.expand_dims(net, 1)  
    net=tf.reshape(input_tensor,[-1,16,16,1])
    net = tf.contrib.layers.conv2d_transpose(net, 128, 5,stride=1,padding='VALID')
    net = tf.contrib.layers.conv2d_transpose(net, 64, 5, stride=1,padding='VALID')
    net = tf.contrib.layers.conv2d_transpose(net, 32, 5, stride=1,padding='VALID')
    net = tf.contrib.layers.conv2d_transpose(
        net, 3, 5, stride=1, padding='VALID',activation_fn=tf.nn.sigmoid)
    net = tf.contrib.layers.flatten(net)
    return net