# -*- coding: utf-8 -*-
"""Layers that act as activation functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import activations, initializers, regularizers, constraints
from keras.engine import Layer, InputSpec

from keras import backend as K
from keras.legacy import interfaces
# import tensorflow as tf
from keras.layers.recurrent import RNN
from keras.layers.merge import _Merge

import numpy as np
from veril import plants_via_RNN
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
# import itertools
# import math


class JanetController(RNN):

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=False,
                 external_input=False,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        plant_name = kwargs.pop('plant_name')
        dt = kwargs.pop('dt')
        obs_idx = kwargs.pop('obs_idx')

        cell = JanetControllerCell(units, plant_name, dt, obs_idx,
                                   activation=activation,
                                   recurrent_activation=recurrent_activation,
                                   use_bias=use_bias,
                                   external_input=external_input,
                                   kernel_initializer=kernel_initializer,
                                   recurrent_initializer=recurrent_initializer,
                                   unit_forget_bias=unit_forget_bias,
                                   bias_initializer=bias_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   recurrent_regularizer=recurrent_regularizer,
                                   bias_regularizer=bias_regularizer,
                                   kernel_constraint=kernel_constraint,
                                   recurrent_constraint=recurrent_constraint,
                                   bias_constraint=bias_constraint,
                                   dropout=dropout,
                                   recurrent_dropout=recurrent_dropout,
                                   implementation=implementation)
        super(JanetController, self).__init__(cell, return_sequences=return_sequences,
                                              return_state=return_state,
                                              go_backwards=go_backwards,
                                              stateful=stateful,
                                              unroll=unroll,
                                              **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(JanetController, self).call(inputs, mask=mask,
                                                 training=training,
                                                 initial_state=initial_state)

    def linearize(self):
        return self.cell.linearize()

    @property
    def units(self):
        return self.cell.units

    @property
    def plant_name(self):
        return self.cell.plant_name

    @property
    def dt(self):
        return self.cell.dt

    @property
    def obs_idx(self):
        return self.cell.obs_idx

    @property
    def num_plant_states(self):
        return self.cell.num_plant_states

    @property
    def num_plant_output(self):
        return self.cell.num_plant_output

    @property
    def num_plant_disturb(self):
        return self.cell.num_plant_disturb

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def external_input(self):
        return self.cell.external_input

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    def get_config(self):
        config = {'units': self.units,
                  'plant_name': self.plant_name,
                  'dt': self.dt,
                  'obs_idx': self.obs_idx,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'external_input': self.external_input,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(JanetController, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)


class JanetControllerCell(Layer):

    def __init__(self, units, plant_name, dt, obs_idx,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=False,
                 external_input=False,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(JanetControllerCell, self).__init__(**kwargs)
        self.units = units
        self.plant_name = plant_name
        self.dt = dt
        self.obs_idx = obs_idx
        self.plant = plants_via_RNN.get(plant_name, dt, obs_idx)
        self.num_plant_states = self.plant.num_states
        self.num_plant_output = self.plant.num_outputs
        self.num_plant_disturb = self.plant.num_disturb

        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.external_input = external_input

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        # the closed-loop states are plant states AND RNN cells
        self.state_size = (self.plant.num_states, self.units)
        # self.output_size = self.plant.num_outputs
        self.output_size = self.plant.num_states
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        # y kernel
        self.kernel = self.add_weight(shape=(self.plant.num_outputs, self.units * 2),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.kernel_f = self.kernel[:, :self.units]
        self.kernel_c = self.kernel[:, self.units * 1: self.units * 2]

        # c kernel
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 2),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        self.recurrent_kernel_f = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_c = self.recurrent_kernel[
            :, self.units: self.units * 2]

        # to output u kernel
        self.output_kernel = self.add_weight(shape=(self.units,
                                                    self.plant.num_inputs),
                                             name='output_kernel',
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             constraint=self.kernel_constraint)

        self.feedthrough_kernel = self.add_weight(shape=(self.plant.num_outputs,
                                                         self.plant.num_inputs),
                                                  name='output_kernel',
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)

        # bias
        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 2,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        if self.use_bias:
            self.bias_f = self.bias[:self.units]
            self.bias_c = self.bias[self.units: self.units * 2]
        else:
            self.bias_f = None
            self.bias_c = None

        # input_dim = input_shape[-1]
        # if input_dim is not None:
        if self.external_input:
            # if external input exists
            self.input_kernel = self.add_weight(shape=(input_dim,
                                                       self.plant.num_inputs),
                                                name='input_kernel',
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint)
        self.built = True

    def call(self, inputs, states, training=None):
        plant = self.plant
        x_tm1 = states[0]
        c_tm1 = states[1]  # previous cell memory state
        shift_y_tm1 = plant.get_obs(x_tm1) - plant.y0

        if self.implementation == 1:
            if 0 < self.recurrent_dropout < 1.:
                c_tm1_f = c_tm1 * rec_dp_mask[0]
                c_tm1_c = c_tm1 * rec_dp_mask[1]
            else:
                c_tm1_f = c_tm1
                c_tm1_c = c_tm1

            c_f = K.dot(c_tm1_f, self.recurrent_kernel_f)
            c_c = K.dot(c_tm1_c, self.recurrent_kernel_c)

            # adding the plant's output feedback
            c_f = c_f + K.dot(shift_y_tm1, self.kernel_f)
            c_c = c_c + K.dot(shift_y_tm1, self.kernel_c)

            if self.use_bias:
                c_f = K.bias_add(c_f, self.bias_f)
                c_c = K.bias_add(c_c, self.bias_c)

            tau_f = self.activation(c_f)
            tau_c = self.activation(c_c)

            c_tm2 = (.5 * (c_tm1 + c_tm1 * tau_f + tau_c - tau_c * tau_f))

            u = K.dot(c_tm1, self.output_kernel)
            u = u + K.dot(shift_y_tm1, self.feedthrough_kernel)
            x_tm2 = plant.step(x_tm1, u)
            # y_tm2 = plant.get_obs(x_tm2)

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        return x_tm2, [x_tm2, c_tm2]

    def linearize(self):
        full_dim = self.units + self.plant.num_states
        init_x = K.reshape(K.variable(self.plant.x0),
                           (1, self.plant.num_states))
        init_c = K.zeros((1, self.units))
        tm1 = [init_x, init_c]
        inputs = K.zeros((1, 1, self.plant.num_disturb))
        tm2 = self.call(inputs, tm1)[1]

        J11 = K.eval(K.transpose(K.reshape(jacobian(tm2[0], tm1[0]),
                                           (self.plant.num_states, self.plant.num_states))))
        # tm2[0]:1-by-num_states, tm1[1]:1-by-units
        J12 = K.eval(K.transpose(K.reshape(jacobian(tm2[0], tm1[1]),
                                           (self.plant.num_states, self.units))))
        J21 = K.eval(K.transpose(K.reshape(jacobian(tm2[1], tm1[0]),
                                           (self.units, self.plant.num_states))))
        J22 = K.eval(K.transpose(K.reshape(jacobian(tm2[1], tm1[1]),
                                           (self.units, self.units))))
        # sess = K.get_session()
        # J11 = sess.run(J11, feed_dict={tm1[0]: np.reshape(self.plant.x0,
        # (1,self.plant.num_states)), tm1[1]:np.zeros((1,self.units))})
        # J12 = sess.run(J12, feed_dict={tm1[0]: np.reshape(self.plant.x0,
        # (1,self.plant.num_states)), tm1[1]:np.zeros((1,self.units))})
        # J21 = sess.run(J21, feed_dict={tm1[0]: np.reshape(self.plant.x0,
        # (1,self.plant.num_states)), tm1[1]:np.zeros((1,self.units))})
        # J22 = sess.run(J22, feed_dict={tm1[0]: np.reshape(self.plant.x0,
        # (1,self.plant.num_states)), tm1[1]:np.zeros((1,self.units))})
        # TODO: do the stacking in K and return with K.eval() to get the
        # numerics so as to not rely on import np.

        J = np.vstack((np.hstack((J11, J21)), np.hstack((J12, J22))))
        # for debugging (3 plant states 4 cell states)
        # tm2_np = np.hstack((K.eval(tm2[0]), K.eval(tm2[1])))
        # initx_delta = init_x + K.reshape(K.variable(np.array([1e-6, 1e-6, 1e-6])),
        #                                  (1, self.plant.num_states))
        # initc_delta = init_c + \
        #     K.reshape(K.variable(
        #         np.array([1e-6, 1e-6, 1e-6, 1e-6])), (1, self.units))
        # tm1_delta = [initx_delta, initc_delta]
        # tm2_delta = self.call(inputs, tm1_delta)[1]
        # print(K.eval(tm2_delta[0]))
        # print(K.eval(tm2_delta[1]))
        # tm1_np_delta = 1e-6 * np.ones((1, full_dim))
        # print((tm1_np_delta)@J + tm2_np)
        J -= np.eye(full_dim)
        # since training all state vectors are row vecs, need to transpose
        # such that state_plus approx A@state_plus
        A0 = (J / self.dt).T
        # print(A0)
        return A0

    def get_config(self):
        config = {'units': self.units,
                  'plant_name': self.plant_name,
                  'dt': self.dt,
                  'obs_idx': self.obs_idx,
                  'num_plant_disturb': self.num_plant_disturb,
                  'num_plant_states': self.num_plant_states,
                  'num_plant_output': self.num_plant_output,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'external_input': self.external_input,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(JanetControllerCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# def Jacobian(X, y):
#     J = tf.map_fn(lambda m: tf.gradients(y[:,:,m:m+1], X)[0], tf.range(tf.shape(y)[-1]), tf.float32)
#     J = tf.transpose(tf.squeeze(J), perm = [1,0,2])
#     return J


class JanetCell(Layer):
    """Cell class for the LSTM layer.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).x
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
    """

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 external_input=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(JanetCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.external_input = external_input

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.state_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if self.external_input:
            self.kernel = self.add_weight(shape=(input_dim, self.units * 2),
                                          name='kernel',
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            self.kernel_f = self.kernel[:, :self.units]
            self.kernel_c = self.kernel[:, self.units * 1: self.units * 2]

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 2),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 2,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.recurrent_kernel_f = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_c = self.recurrent_kernel[
            :, self.units: self.units * 2]

        if self.use_bias:
            self.bias_f = self.bias[:self.units]
            self.bias_c = self.bias[self.units: self.units * 2]
        else:
            self.bias_f = None
            self.bias_c = None
        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=2)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=2)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        # c_tm1 = h_tm1  # previous carry state

        if self.implementation == 1:
            if 0 < self.recurrent_dropout < 1.:
                h_tm1_f = h_tm1 * rec_dp_mask[0]
                h_tm1_c = h_tm1 * rec_dp_mask[1]
            else:
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1

            x_f = K.dot(h_tm1_f, self.recurrent_kernel_f)
            x_c = K.dot(h_tm1_c, self.recurrent_kernel_c)

            if 0 < self.dropout < 1.:
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
            else:
                inputs_f = inputs
                inputs_c = inputs

            if self.external_input:
                x_f = x_f + K.dot(inputs_f, self.kernel_f)
                x_c = x_c + K.dot(inputs_c, self.kernel_c)

            if self.use_bias:
                x_f = K.bias_add(x_f, self.bias_f)
                x_c = K.bias_add(x_c, self.bias_c)

            f = self.recurrent_activation(x_f)
            c = f * h_tm1 + (1 - f) * self.activation(x_c)
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            if self.external_input:
                z = K.dot(inputs, self.kernel)
            else:
                z = K.zeors(inputs.shape)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            z += K.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]

            f = self.recurrent_activation(z0)
            c = f * h_tm1 + (1 - f) * self.activation(z1)

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return c, [c]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'external_input': self.external_input,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(JanetCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Janet(RNN):
    """Long Short-Term Memory layer - Hochreiter 1997.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    # References
        - [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 external_input=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = JanetCell(units,
                         activation=activation,
                         recurrent_activation=recurrent_activation,
                         use_bias=use_bias,
                         external_input=external_input,
                         kernel_initializer=kernel_initializer,
                         recurrent_initializer=recurrent_initializer,
                         unit_forget_bias=unit_forget_bias,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         recurrent_regularizer=recurrent_regularizer,
                         bias_regularizer=bias_regularizer,
                         kernel_constraint=kernel_constraint,
                         recurrent_constraint=recurrent_constraint,
                         bias_constraint=bias_constraint,
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout,
                         implementation=implementation)
        super(Janet, self).__init__(cell,
                                    return_sequences=return_sequences,
                                    return_state=return_state,
                                    go_backwards=go_backwards,
                                    stateful=stateful,
                                    unroll=unroll,
                                    **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(Janet, self).call(inputs,
                                       mask=mask,
                                       training=training,
                                       initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def external_input(self):
        return self.cell.external_input

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'external_input': self.external_input,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(Janet, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)


class TransLayer(Layer):
    # @interfaces.legacy_dense_support

    def __init__(self, old_layer,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(TransLayer, self).__init__(**kwargs)
        # self.units = units
        self.old_layer = old_layer
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = K.transpose(self.old_layer.kernel)
        self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        # output_shape = list(input_shape)
        # output_shape[-1] = self.out_dim
        # return tuple(output_shape)
        return self.old_layer.input_shape

    def get_config(self):
        config = {
        }
        base_config = super(TransLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DotKernel(Layer):

    # @interfaces.legacy_dense_support
    def __init__(self, A, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DotKernel, self).__init__(**kwargs)
        self.A = A
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = K.dot(inputs, K.variable(self.A))
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = K.variable(self.A).shape[-1]
        return tuple(output_shape)

    def get_config(self):
        config = {
            'A': self.A,
        }
        base_config = super(DotKernel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Divide(_Merge):
    """Layer that divides two inputs.

    It takes as input a list of tensors of size 2,
    both of the same shape, and returns a single tensor, (inputs[0] / inputs
    [1]),
    also of the same shape.
    """

    def build(self, input_shape):
        super(Divide, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `Subtract` layer should be called '
                             'on exactly 2 inputs')

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `Subtract` layer should be called '
                             'on exactly 2 inputs')
        return inputs[0] / inputs[1]


class SumUp(Layer):
    """Sum the input. Does not affect the batch size.
    """

    def __init__(self, data_format=None, **kwargs):
        super(SumUp, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 1
        return tuple(output_shape)

    def call(self, inputs):
        return K.sum(inputs, axis=-1)

    def get_config(self):
        config = {}
        base_config = super(SumUp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Sigmoid(Layer):
    """sigmoid activation function.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        axis: Integer, axis along which the sigmoid normalization is applied.
    """

    def __init__(self, axis=-1, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return activations.sigmoid(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Normalize(Layer):
    """Normalize activation function.
    """

    def __init__(self, axis=-1, **kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return inputs / K.max(inputs)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Normalize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Sign(Layer):
    """Sign activation function.
    """

    def __init__(self, axis=-1, **kwargs):
        super(Sign, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return K.sign(inputs)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Sign, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Flipped_Saturate(Layer):
    """Flipped_Saturate activation function.
    """

    def __init__(self, axis=-1, **kwargs):
        super(Flipped_Saturate, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        # if inputs greater than zero, return the inputs, otherwise, return one
        return - K.sign(inputs) * K.cast(K.less(inputs, 0), K.floatx()) \
            + inputs * K.cast(K.greater_equal(inputs, 0), K.floatx())

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Flipped_Saturate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Min_Positive(Layer):
    """Min_Positive activation function.
    """

    def __init__(self, axis=-1, **kwargs):
        super(Min_Positive, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        min_pos = K.min(K.max(K.maximum(inputs, 0)) * K.cast(K.less(inputs, 0), K.floatx()) + inputs * K.cast(K.greater_equal(inputs, 0),
                                                                                                              K.floatx()))
        # return K.cast(K.greater_equal(inputs, min_pos), K.floatx())
        return min_pos

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Min_Positive, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Power(Layer):

    def __init__(self, alpha=1, **kwargs):
        super(Power, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = alpha

    def call(self, inputs):
        return K.pow(inputs, self.alpha)

    def get_config(self):
        config = {'alpha': self.alpha}
        base_config = super(Power, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Sign_of_one_minus_input(Layer):

    def __init__(self, **kwargs):
        super(Sign_of_one_minus_input, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.sign(K.ones(K.shape(inputs)) - inputs)

    def get_config(self):
        config = {'alpha': self.alpha}
        base_config = super(Sign_of_one_minus_input, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Greater(_Merge):
    """Layer that Greaters two inputs.

    It takes as input a list of tensors of size 2,
    both of the same shape, and returns a single tensor, (inputs[0] / inputs
    [1]),
    also of the same shape.
    """

    def build(self, input_shape):
        super(Greater, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `Subtract` layer should be called '
                             'on exactly 2 inputs')

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `Subtract` layer should be called '
                             'on exactly 2 inputs')
        return K.cast(K.greater_equal(inputs[0], inputs[1]), K.floatx())


class DenseOnOff(Layer):

    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseOnOff, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        onoff = K.greater_equal(output, K.cast_to_floatx(0.))
        self.onoff = K.cast(onoff, dtype="int32")
        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DenseOnOff, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# class Monomials(Layer):

#     def __init__(self, units,
#                  activation=None,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         if 'input_shape' not in kwargs and 'input_dim' in kwargs:
#             kwargs['input_shape'] = (kwargs.pop('input_dim'),)
#         super(Monomials, self).__init__(**kwargs)
#         self.units = units
#         self.activation = activations.get(activation)
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.activity_regularizer = regularizers.get(activity_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)
#         self.input_spec = InputSpec(min_ndim=2)
#         self.supports_masking = True

#     def build(self, input_shape):
#         assert len(input_shape) >= 2
#         input_dim = input_shape[-1]
#         # adding a power kernel
#         powers = tf.range(1, self.units + 1, 1, dtype='float32')
#         powers = K.reshape(powers, (self.units,))
#         self.pow_kernel = powers
#         self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
#         self.built = True

#     def call(self, inputs):
#         output = K.pow(inputs, self.pow_kernel)
#         return output

#     def compute_output_shape(self, input_shape):
#         assert input_shape and len(input_shape) >= 2
#         assert input_shape[-1]
#         output_shape = list(input_shape)
#         output_shape[-1] = self.units
#         return tuple(output_shape)

#     def get_config(self):
#         config = {
#             'units': self.units,
#             'activation': activations.serialize(self.activation),
#             'use_bias': self.use_bias,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#             'bias_initializer': initializers.serialize(self.bias_initializer),
#             'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer': regularizers.serialize(self.bias_regularizer),
#             'activity_regularizer': regularizers.serialize(self.activity_regularizer),
#             'kernel_constraint': constraints.serialize(self.kernel_constraint),
#             'bias_constraint': constraints.serialize(self.bias_constraint)
#         }
#         base_config = super(Monomials, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


class Polynomials(Layer):

    def __init__(self, n, max_deg,
                 activation=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Polynomials, self).__init__(**kwargs)
        self.max_deg = max_deg
        self.n = n
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        # adding a power kernel
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        y = list(itertools.combinations_with_replacement(
            range(self.n + 1), self.max_deg))
        exponents = [[i.count(j) for j in range(1, self.n + 1)] for i in y][1:]
        output = K.concatenate([K.prod(j, axis=-1, keepdims=True) for j in
                                [K.pow(inputs, i) for i in exponents]])
        # cross = K.expand_dims(K.prod(K.pow(inputs,1),axis=-1))
        # output = K.concatenate([output,cross])
        # output = K.variable([np.linalg.multi_dot(i) for i in aa])
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        # n = input_shape[-1]
        assert self.n
        output_shape = list(input_shape)
        f = lambda x: math.factorial(x)
        output_shape[-1] = f(self.n +
                             self.max_deg) // f(self.max_deg) // f(self.n) - 1
        return tuple(output_shape)

    def get_config(self):
        config = {
            'max_deg': self.max_deg,
            'n': self.n,
        }
        base_config = super(Polynomials, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DiffPoly(Layer):

    def __init__(self, n, max_deg,
                 activation=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DiffPoly, self).__init__(**kwargs)
        self.max_deg = max_deg
        self.n = n
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        # adding a power kernel
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        y = list(itertools.combinations_with_replacement(
            range(self.n + 1), self.max_deg))
        exponents = [[i.count(j) for j in range(1, self.n + 1)] for i in y][1:]
        phi = K.concatenate([K.prod(j, axis=-1, keepdims=True)
                             for j in [K.pow(inputs, i) for i in exponents]])
        output = batch_jacobian(phi, inputs)
        # output = K.concatenate([output,K.ones((1,))])
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        f = lambda x: math.factorial(x)
        output_shape.insert(-1, (f(self.n + self.max_deg) //
                                 f(self.max_deg) // f(self.n)) - 1)
        return tuple(output_shape)

    def get_config(self):
        config = {
            'max_deg': self.max_deg,
            'n': self.n,
        }
        base_config = super(DiffPoly, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ReluOnOff(Layer):

    def __init__(self, onoff=0., **kwargs):
        super(ReluOnOff, self).__init__(**kwargs)
        self.supports_masking = True
        self.theta = K.cast_to_floatx(onoff)
        # self.onoff = []

    def call(self, inputs):
        # print inputs.shape
        onoff = K.greater_equal(inputs, self.theta)
        self.onoff = K.cast(onoff, dtype="int32")
        return K.relu(inputs)

    def get_config(self):
        config = {}
        base_config = super(ReluOnOff, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class CeilingLayer(Layer):

    def __init__(self, theta=10, **kwargs):
        super(CeilingLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.theta = K.cast_to_floatx(theta)

    def call(self, inputs):
        return K.minimum(inputs, self.theta)

    def get_config(self):
        config = {}
        base_config = super(CeilingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Scaling(Layer):

    def __init__(self, alpha=1., **kwargs):
        super(Scaling, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        # print inputs.shape
        onoff = K.greater_equal(inputs, self.theta)
        self.onoff = K.cast(onoff, dtype="int32")
        return K.relu(inputs)

    def get_config(self):
        config = {}
        base_config = super(Scaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
