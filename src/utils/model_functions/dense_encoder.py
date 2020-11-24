# import necessary modules
import tensorflow as tf
from .scalar_multiplier import ScalarMultiply
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    BatchNormalization,
    Dense,
    Input,
    LSTM,
    Reshape,
)


class DenseBlock(tf.keras.Model):
    def __init__(
        self,
        repetitions,
        layer_width,
        shape_in=(1,),
        num_out=1,
        neurons_out=(1,),
        bn=False,
        init="glorot_uniform",
        act="tanh",
    ):
        """
        Creates a sequence of equal width dense layers
        :param repetitions: int, number of dense layer repetitions
        :param layer_width: int, number of neurons per dense layer
        :param shape_in: tuple, data shape of the input to the DenseBlock
        :param num_out: int, number of outputs
        :param neurons_out: tuple, number of neurons per output
        :param bn: float, the use of batch normalization after a dense layer
        :param init: kernel initialization for dense layers
        :param act: activation function for the dense layers
        """
        # initialize
        super(DenseBlock, self).__init__()

        self.repetitions = repetitions
        self.layer_width = layer_width
        self.shape_in = shape_in
        self.num_out = num_out
        self.neurons_out = neurons_out
        self.bn = bn
        self.init = init
        self.act = act

        self.dense = self.__make_dense()

    def __make_dense(self):
        inp = []

        for i, shape in enumerate(self.shape_in):
            inp.append(Input(shape=shape, name="input_{}".format(i + 1)))
        x = Concatenate(axis=-1)(inp) if len(self.shape_in) > 1 else inp[0]

        # dense layers
        for i in range(self.repetitions):
            x = Dense(
                self.layer_width,
                kernel_initializer=self.init,
                name="dense_{}".format(i + 1),
            )(x)
            x = Activation(self.act, name="{}_{}".format(self.act, i + 1))(x)
            if self.bn:
                x = BatchNormalization(name="bn_{}".format(i + 1))(x)

        # outputs
        out = []
        for i in range(self.num_out):
            o = Dense(
                self.neurons_out[i],
                kernel_initializer=self.init,
                name="out_{}".format(i + 1),
            )(x)
            out.append(o)
        return tf.keras.Model(inputs=inp, outputs=out)

    @tf.function
    def call(self, x):
        return self.dense(x)