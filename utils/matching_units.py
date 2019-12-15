import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops.nn_ops import max_pool, leaky_relu
from tensorflow.contrib.layers import batch_norm, layer_norm
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from tensorflow.python.ops.nn_ops import leaky_relu
from utils.network_summary import count_parameters

from utils.sn import spectral_normed_weight, spectral_norm


class g_embedding_bidirectionalLSTM:
    def __init__(self, name, layer_sizes, batch_size):
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer e.g. [100, 100, 100] returns a 3 layer, 100
                                                                                                        neuron bid-LSTM
        :param batch_size: The experiments batch size
        """
        self.reuse = False
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.name = name

    def __call__(self, inputs, training=False):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param inputs: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :param name: Name to give to the tensorflow op
        :param training: Flag that indicates if this is a training or evaluation stage
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope("encoder"):

                fw_lstm_cells_encoder = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
                                         for i in range(len(self.layer_sizes))]
                bw_lstm_cells_encoder = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
                                         for i in range(len(self.layer_sizes))]

                outputs, output_state_fw, output_state_bw = rnn.stack_bidirectional_rnn(
                    fw_lstm_cells_encoder,
                    bw_lstm_cells_encoder,
                    inputs,
                    dtype=tf.float32
                )

            # print("g out shape", tf.stack(outputs, axis=1).get_shape().as_list())

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return outputs

class f_embedding_bidirectionalLSTM:
    def __init__(self, name, layer_size, batch_size):
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer e.g. [100, 100, 100] returns a 3 layer, 100
                                                                                                        neuron bid-LSTM
        :param batch_size: The experiments batch size
        """
        self.reuse = False
        self.batch_size = batch_size
        self.layer_size = layer_size
        self.name = name

    def __call__(self, support_set_embeddings, target_set_embeddings, K, training=False):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param inputs: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :param name: Name to give to the tensorflow op
        :param training: Flag that indicates if this is a training or evaluation stage
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        b, k, h_g_dim = support_set_embeddings.get_shape().as_list()
        b, h_f_dim = target_set_embeddings.get_shape().as_list()
        with tf.variable_scope(self.name, reuse=self.reuse):
            fw_lstm_cells_encoder = rnn.LSTMCell(num_units=self.layer_size, activation=tf.nn.tanh)
            attentional_softmax = tf.ones(shape=(b, k)) * (1.0/k)
            h = tf.zeros(shape=(b, h_g_dim))
            c_h = (h, h)
            c_h = (c_h[0], c_h[1] + target_set_embeddings)
            for i in range(K):
                attentional_softmax = tf.expand_dims(attentional_softmax, axis=2)
                attented_features = support_set_embeddings * attentional_softmax
                attented_features_summed = tf.reduce_sum(attented_features, axis=1)
                c_h = (c_h[0], c_h[1] + attented_features_summed)
                print('1',target_set_embeddings)
                x, h_c = fw_lstm_cells_encoder(inputs=target_set_embeddings, state=c_h)
                attentional_softmax = tf.layers.dense(x, units=k, activation=tf.nn.softmax, reuse=self.reuse)
                self.reuse = True

        outputs = x
        # print("out shape", tf.stack(outputs, axis=0).get_shape().as_list())
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        # print(self.variables)
        return outputs



# class DistanceNetwork:
#     def __init__(self):
#         self.reuse = False

#     def __call__(self, support_set, input_image, name, training=False):
#         with tf.name_scope('distance-module' + name), tf.variable_scope('distance-module', reuse=self.reuse):
#             eps = 1e-10
#             similarities = []
#             for support_image in tf.unstack(support_set, axis=0):
#                 sum_support = tf.reduce_sum(tf.square(support_image), 1, keep_dims=True)
#                 sum_target = tf.reduce_sum(tf.square(input_image), 1, keep_dims=True)
#                 support_magnitude = tf.rsqrt(tf.clip_by_value(sum_support, eps, float("inf")))
#                 target_magnitude = tf.rsqrt(tf.clip_by_value(sum_target, eps, float("inf")))
#                 dot_product = tf.matmul(tf.expand_dims(input_image, 1), tf.expand_dims(support_image, 2))
#                 dot_product = tf.squeeze(dot_product, [1, ])
#                 cosine_similarity = dot_product * support_magnitude * target_magnitude
#                 similarities.append(cosine_similarity)

#         similarities = tf.concat(axis=1, values=similarities)
#         # similarities = tf.nn.softmax(similarities)
#         self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='distance-module')
#         return similarities


class DistanceNetwork:
    def __init__(self):
        self.reuse = False

    def __call__(self, support_set, input_image, name, training=False):
        with tf.name_scope('distance-module' + name), tf.variable_scope('distance-module', reuse=self.reuse):
            eps = 1e-10
            similarities = []
            similarities_standard = []
            for support_image in tf.unstack(support_set, axis=0):
                sum_support = tf.reduce_sum(tf.square(support_image), 1, keep_dims=True)
                support_magnitude = tf.rsqrt(tf.clip_by_value(sum_support, eps, float("inf")))
                sum_target = tf.reduce_sum(tf.square(input_image), 1, keep_dims=True)
                target_magnitude = tf.rsqrt(tf.clip_by_value(sum_target, eps, float("inf")))
                dot_product = tf.matmul(tf.expand_dims(input_image, 1), tf.expand_dims(support_image, 2))
                dot_product = tf.squeeze(dot_product, [1, ])
                cosine_similarity = dot_product * support_magnitude
                cosine_similarity_standard = cosine_similarity * target_magnitude
                similarities.append(cosine_similarity)
                similarities_standard.append(cosine_similarity_standard)

            
        
        similarities = tf.concat(axis=1, values=similarities)
        mean_magnitude = tf.reduce_mean(target_magnitude)
        similarities_standard = tf.concat(axis=1,values=similarities_standard)


        minvalue = tf.reduce_min(similarities_standard,axis=1)
        maxvalue = tf.reduce_max(similarities_standard,axis=1)
        meanvalue, variance = tf.nn.moments(similarities_standard,axes=1)
        minvalue  = tf.expand_dims(minvalue ,axis=1)
        maxvalue=tf.expand_dims(maxvalue,axis=1)
        meanvalue=tf.expand_dims(meanvalue,axis=1)
        variance=tf.expand_dims(variance,axis=1)
       



        all_data = tf.concat([similarities_standard,target_magnitude,minvalue,maxvalue,meanvalue,variance],axis=1)
        scale = tf.random_uniform(shape=[int(similarities.get_shape()[0]),1],minval=1,maxval=10)  
        scale = tf.concat([scale,scale,scale],axis=1) 
        similarities_standard = tf.nn.softmax(similarities)
        # similarities_standard = tf.nn.softmax(tf.multiply(scale,similarities))
        # similarities_standard = tf.nn.softmax(1e1*similarities)
        # similarities_standard = tf.nn.softmax(similarities)
        # similarities_standard = tf.nn.softmax(similarities)
        # similarities_standard = tf.nn.softmax(similarities)
        # similarities_standard = similarities
        # similarities_standard = tf.snn.softmax(similarities)
        # similarities_standard_scale = tf.random_uniform(shape=[1],minval=1,maxval=100)
        # similarities_standard = similarities_standard_scale : * similarities_standard
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='distance-module')
        return similarities, similarities_standard, all_data


# class DistanceNetwork:
#     def __init__(self):
#         self.reuse = False

#     def __call__(self, support_set, input_image, name, training=False):
#         with tf.name_scope('distance-module' + name), tf.variable_scope('distance-module', reuse=self.reuse):
#             eps = 1e-10
#             similarities = []
#             for support_image in tf.unstack(support_set, axis=0)::
#                 distance_current = tf.sqrt(tf.reduce_sum(tf.square(support_image-input_image), 1, keep_dims=True))
#                 similarities.append(distance_current)
#         similarities = tf.concat(axis=1, values=similarities)        
#         sum_value=tf.reduce_sum(similarities,axis=1)
#         similarities_value = []
#         for i in range(similarities.get_shape()[0]):
#             similarities_value.append(similarities[i]/sum_value[i])
#         similarities = tf.stack(similarities_value,axis=0)
#         similarities_standard = tf.nn.softmax(similarities)
#         self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='distance-module')
#         return similarities, similarities_standard

class AttentionalClassify:
    def __init__(self):
        self.reuse = False

    def __call__(self, similarities, support_set_y, name, training=False):
        with tf.name_scope('attentional-classification' + name), tf.variable_scope('attentional-classification',
                                                                                   reuse=self.reuse):
            preds = tf.squeeze(tf.matmul(tf.expand_dims(similarities, 1), support_set_y))
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attentional-classification')
        return preds




class Classifier:
    def __init__(self, name, batch_size, layer_sizes, num_channels=1):
        """
        Builds a CNN to produce embeddings
        :param batch_size: Batch size for experiment
        :param layer_sizes: A list of length 4 containing the layer sizes
        :param num_channels: Number of channels of images
        """
        self.reuse = tf.AUTO_REUSE
        self.name = name
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_sizes = layer_sizes
        assert len(self.layer_sizes) == 4, "layer_sizes should be a list of length 4"

    def __call__(self, image_input, training=False, dropout_rate=0.0):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, 64]
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            outputs = image_input
            with tf.variable_scope('conv_layers'):
                for idx, num_filters in enumerate(self.layer_sizes):
                    with tf.variable_scope('g_conv_{}'.format(idx)):
                        if idx == len(self.layer_sizes) - 1:
                            outputs = tf.layers.conv2d(outputs, num_filters, [2, 2], strides=(1, 1),
                                                       padding='VALID')
                        else:
                            outputs = tf.layers.conv2d(outputs, num_filters, [3, 3], strides=(1, 1),
                                                               padding='VALID')
                        outputs = leaky_relu(outputs)
                        outputs = tf.contrib.layers.batch_norm(outputs, updates_collections=None,
                                                                       decay=0.99,
                                                                       scale=True, center=True,
                                                                       is_training=training)
                        outputs = max_pool(outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

            image_embedding = tf.contrib.layers.flatten(outputs)


        self.reuse = tf.AUTO_REUSE
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return image_embedding




weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope=None):
    with tf.variable_scope(scope):
        # if pad_type == 'zero' :
        #     x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        # if pad_type == 'reflect' :
        #     x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='SAME', name='conv_sn')
            if use_bias:
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias, padding="SAME", name='conv')
        return x


# def conv_layer(self, inputs, num_filters, filter_size, strides, activation=None,
#                transpose=False, w_size=None, h_size=None,scope='scope_0'):
#     self.conv_layer_num += 1
#     if transpose:
#         outputs = self.upscale(inputs, h_size=h_size, w_size=w_size)
#         outputs = tf.layers.conv2d_transpose(outputs, num_filters, filter_size,
#                                              strides=strides,
#                                    padding="SAME", activation=activation)
#     elif not transpose:
#         outputs = tf.layers.conv2d(inputs, num_filters, filter_size, strides=strides,
#                                              padding="SAME", activation=activation)
#     return outputs


def deconv(x, channels, kernel=4, stride=2, use_bias=True, sn=False, scope=None):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]
        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_normed_weight(w), output_shape=output_shape,
                                       strides=[1, stride, stride, 1], padding='SAME', name='deconv_sn')

            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init,
                                           kernel_regularizer=weight_regularizer,
                                           strides=stride, padding='SAME', use_bias=use_bias, name='deconv')

        return x


def fully_conneted(x, units, use_bias=True, sn=False):
    x = tf.layers.flatten(x)
    shape = x.get_shape().as_list()
    channels = shape[-1]

    if sn:
        w = tf.get_variable("kernel", [channels, units], tf.float32,
                            initializer=weight_init, regularizer=weight_regularizer)
        if use_bias:
            bias = tf.get_variable("bias", [units],
                                   initializer=tf.constant_initializer(0.0))

            x = tf.matmul(x, spectral_norm(w)) + bias
        else:
            x = tf.matmul(x, spectral_norm(w))

    else:
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                            use_bias=use_bias)

    return x


def max_pooling(x, kernel=2, stride=2):
    return tf.layers.max_pooling2d(x, pool_size=kernel, strides=stride)


def avg_pooling(x, kernel=2, stride=2):
    return tf.layers.average_pooling2d(x, pool_size=kernel, strides=stride)


def global_avg_pooling(x):
    """
    Incoming Tensor shape must be 4-D
    """
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap


def flatten(x):
    return tf.layers.flatten(x)


def lrelu(x, alpha=0.2):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def sigmoid(x):
    return tf.sigmoid(x)


def tanh(x):
    return tf.tanh(x)


def swish(x):
    return x * sigmoid(x)


def remove_duplicates(input_features):
    """
    Remove duplicate entries from layer list.
    :param input_features: A list of layers
    :return: Returns a list of unique feature tensors (i.e. no duplication).
    """
    feature_name_set = set()
    non_duplicate_feature_set = []
    for feature in input_features:
        if feature.name not in feature_name_set:
            non_duplicate_feature_set.append(feature)
        feature_name_set.add(feature.name)
    return non_duplicate_feature_set






class Unet_encoder:
    def __init__(self,  layer_sizes,inner_layers):
        self.reuse = tf.AUTO_REUSE
        self.layer_sizes = layer_sizes
        self.inner_layers = inner_layers

    def conv_layer(self, inputs, num_filters, filter_size, strides, scope, activation=None,
                   transpose=False, w_size=None, h_size=None, sn=True):
        if transpose:
            outputs = self.upscale(inputs, h_size=h_size, w_size=w_size)
            outputs = deconv(outputs, channels=num_filters, kernel=filter_size[0], stride=strides[0], use_bias=True,
                             sn=sn, scope=scope)
        elif not transpose:
            outputs = conv(inputs, channels=num_filters, kernel=filter_size[0], stride=strides[0], pad=2, sn=sn,
                           scope=scope)
        return outputs

    def add_encoder_layer(self, input, name, training, dropout_rate, layer_to_skip_connect, local_inner_layers,
                          num_features, dim_reduce=False, scope=None):

        """
        Adds a resnet encoder layer.
        :param input: The input to the encoder layer
        :param training: Flag for training or validation
        :param dropout_rate: A float or a placeholder for the dropout rate
        :param layer_to_skip_connect: Layer to skip-connect this layer to
        :param local_inner_layers: A list with the inner layers of the current Multi-Layer
        :param num_features: Number of feature maps for the convolutions
        :param dim_reduce: Boolean value indicating if this is a dimensionality reducing layer or not
        :return: The output of the encoder layer
        """
        [b1, h1, w1, d1] = input.get_shape().as_list()

        if len(layer_to_skip_connect) >= 2:
            layer_to_skip_connect = layer_to_skip_connect[-2]
        else:
            layer_to_skip_connect = None

        if layer_to_skip_connect is not None:
            [b0, h0, w0, d0] = layer_to_skip_connect.get_shape().as_list()
            if h0 > h1:
                skip_connect_layer = self.conv_layer(layer_to_skip_connect, int(layer_to_skip_connect.get_shape()[3]),
                                                     [3, 3], strides=(2, 2), scope='scope1')
            else:
                skip_connect_layer = layer_to_skip_connect
            # print('1',input)
            # print('2',skip_connect_layer)
            current_layers = [input, skip_connect_layer]
        else:
            current_layers = [input]

        current_layers.extend(local_inner_layers)
        current_layers = remove_duplicates(current_layers)
        outputs = tf.concat(current_layers, axis=3)

        if dim_reduce:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(2, 2), scope='scope2')
            outputs = leaky_relu(outputs)
            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                 center=True, is_training=training,
                                 renorm=False, scope='norm_en')
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        else:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(1, 1), scope='scope2')
            outputs = leaky_relu(features=outputs)
            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                 center=True, is_training=training,
                                 renorm=False, scope='norm_en')

        return outputs

    def __call__(self, image_input, scope, training=False, dropout_rate=0.0):
        outputs = image_input
        encoder_layers = []
        current_layers = [outputs]
        with tf.variable_scope(scope):
            for i, layer_size in enumerate(self.layer_sizes):
                encoder_inner_layers = [outputs]
                with tf.variable_scope('g_conv{}'.format(i)):
                    if i == 0:  
                        outputs = self.conv_layer(outputs, num_filters=64,
                                                  filter_size=(3, 3), strides=(2, 2),
                                                  scope='g_conv{}'.format(i))
                        outputs = leaky_relu(features=outputs)
                        outputs = batch_norm(outputs, decay=0.99, scale=True,
                                             center=True, is_training=training,
                                             renorm=True, scope='bn_1')
                        current_layers.append(outputs)
                        encoder_inner_layers.append(outputs)
                    else:
                        for j in range(self.inner_layers[i]):  # Build the inner Layers of the MultiLayer
                            with tf.variable_scope('g_conv_inner_layer{}'.format(j)):
                                outputs = self.add_encoder_layer(input=outputs,
                                                                 training=training,
                                                                 name="encoder_layer_{}_{}".format(i,
                                                                                                   j),
                                                                 layer_to_skip_connect=current_layers,
                                                                 num_features=self.layer_sizes[i],
                                                                 dim_reduce=False,
                                                                 local_inner_layers=encoder_inner_layers,
                                                                 dropout_rate=dropout_rate,
                                                                 scope="encoder_layer_{}_{}".format(i,
                                                                                                    j))
                                encoder_inner_layers.append(outputs)
                                # current_layers.append(outputs)
                        # add final dim reducing conv layer for this MultiLayer
                        outputs = self.add_encoder_layer(input=outputs,
                                                         name="encoder_layer_{}".format(j),
                                                         training=training,
                                                         layer_to_skip_connect=current_layers,
                                                         local_inner_layers=encoder_inner_layers,
                                                         num_features=self.layer_sizes[i],
                                                         dim_reduce=True, dropout_rate=dropout_rate,
                                                         scope="encoder_layer_{}".format(i))
                        current_layers.append(outputs)
                    encoder_layers.append(outputs)

        return outputs, encoder_layers





