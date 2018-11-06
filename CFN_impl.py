import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from keras.initializers import Initializer

from tensorflow.python.layers import base as base_layer

_WEIGHTS_VARIABLE_NAME = "kernel"
_BIAS_VARIABLE_NAME = "bias"
# p = 0.55
# q = 0.45

def initzer(dtype):
    #return init_ops.zeros_initializer(dtype)
    return tf.random_uniform_initializer(minval=-0.07,maxval=0.07,dtype=dtype) # = self.dtype

class MinusOnes(Initializer):
  """Initializer that generates tensors initialized to -1."""

  def __init__(self, dtype=tf.float32):
    self.dtype = tf.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return tf.scalar_mul(-1.0 , array_ops.ones(shape, dtype))

  def get_config(self):
    return {"dtype": self.dtype.name}


#@tf_export("nn.rnn_cell.CFNCell")
class CFNCell(rnn_cell_impl.LayerRNNCell):
  """The implementation of CFN cell.
  Args:
    num_units: int, The number of units in the CFN cell.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self, num_units, p,q, activation=None, reuse=None, name=None):
    super(CFNCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self.p = 1.0 - p
    self.q = 1.0 - q

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    #input_depth = inputs_shape[1].value
    self._weights_U_theta = self.add_variable(
        "gate/U_theta",
        shape=[self._num_units, self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype)   #
        )
    
    self._weights_V_theta = self.add_variable(
        "gate/V_theta",
        shape=[self._num_units, self._num_units],
        initializer=initzer(dtype = self.dtype)   ##tf.random_uniform_initializer(dtype = self.dtype, minval=-0.07,maxval=0.07)
        )
    
    self._bias_theta = self.add_variable(
        "gate/b_theta",
        shape=[self._num_units],
        initializer= init_ops.zeros_initializer(dtype=self.dtype) ##tf.ones_initializer(dtype=self.dtype))
        )
    self._weights_U_n = self.add_variable(
        "gate/U_n",
        shape=[self._num_units, self._num_units],
        initializer=initzer(dtype = self.dtype)
        )
    
    self._weights_V_n = self.add_variable(
        "gate/V_n",
        shape=[self._num_units, self._num_units],
        initializer=initzer(dtype = self.dtype)
        )
    
    self._bias_n = self.add_variable(
        "gate/b_n",
        shape=[self._num_units],
        initializer= init_ops.zeros_initializer(dtype=self.dtype) ##MinusOnes(dtype=self.dtype)
        )
    self._W =  self.add_variable(
        "kernel",
        shape=[self._num_units, self._num_units],
        initializer=initzer(dtype = self.dtype)
       )

    self.built = True

  def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    
    h0_pr = nn_ops.dropout(inputs,self.p) ##[bs, hs]
    h0_pr_gate = nn_ops.dropout(inputs,self.q) ## [bs, hs]
    h1_l_pr_gate = nn_ops.dropout(state,self.q) ## [bs, hs]
    theta_1 = math_ops.add(math_ops.matmul(h1_l_pr_gate, self._weights_U_theta), math_ops.matmul(h0_pr_gate, self._weights_V_theta)) ## [bs, hs]
    theta_1 = tf.add(theta_1, self._bias_theta) #[bs,hs]
    n_1 = math_ops.add(math_ops.matmul(h1_l_pr_gate, self._weights_U_n), math_ops.matmul(h0_pr_gate, self._weights_V_n)) # [bs, hs]
    n_1 = math_ops.add(n_1, self._bias_n) #[bs, hs]
    h1_cur = tf.add(math_ops.multiply(theta_1, math_ops.tanh(state)), math_ops.multiply(n_1, math_ops.tanh(math_ops.matmul(h0_pr, self._W))))

    return h1_cur, h1_cur

  def zero_state(self, batch_size,dtype):
      return tf.zeros([batch_size, self._num_units ], dtype)

