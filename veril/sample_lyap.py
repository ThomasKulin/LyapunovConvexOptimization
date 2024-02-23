import os
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.layers.advanced_activations import ReLU, Softmax
from keras import backend as K

# from keras.callbacks import TensorBoard
from keras import regularizers, initializers
from keras.utils import CustomObjectScope

from math import factorial as fact
from veril.custom_layers import *
import numpy as np
'''
A note about the DOT layer: if input is 1: (None, a) and 2: (None, a) then no
need to do transpose, direct Dot()(1,2) and the output works correctly with
shape (None, 1).

If input is 1: (None, a, b) and 2: (None, b)

If debug, use kernel_initializer= initializers.Ones()
If regularizing, use kernel_regularizer= regularizers.l2(0.))

    # if write_log:
    #     logs_dir = "/Users/shenshen/veril/data/"
    #     tensorboard = TensorBoard(log_dir=logs_dir, histogram_freq=0,
    #                               write_graph=True, write_images=True)
    #     callbacks.append(tensorboard)
'''


def max_pred(y_true, y_pred):
    return K.max(y_pred)


def mean_pred(y_true, y_pred):
    return y_pred


def neg_percent(y_true, y_pred):
    return K.cast(K.equal(y_true, K.sign(y_pred)), K.floatx())


def mean_pos(y_true, y_pred):
    return K.maximum(y_pred, 0.)


def mean_neg(y_true, y_pred):
    return K.minimum(y_pred, 0.)


def max_min(y_true, y_pred):
    # maximize the V value among all the samples such that Vdot is negative
    # return K.maximum((K.minimum(y_pred,5e-2)), 0.)
    # vdotsign = K.sign(y_pred[-1])
    # signed_V = vdotsign * y_pred[0]
    return -K.min(y_pred)


def sign_loss(y_true, y_pred):
    return K.sign(y_pred)


def flipped_relu(y_true, y_pred):
    return - K.maximum(y_pred, 0.)


def rho_reg(weight_matrix):
    return 0.001 * K.abs(K.sum(weight_matrix))


def model_V(system, over_para, reg, **kwargs):
    sys_dim = system.num_states
    deg_ftrs = system.deg_ftrs
    rm_one = system.rm_one

    if reg is 'l1':
        reg = regularizers.l1(0.)
    elif reg is 'l2':
        reg = regularizers.l2(0.)
    elif reg is 'reg_u' and hasattr(system, 'u_scaling_reg'):
        reg = reg_u(scaling=system.u_scaling_reg())
    elif reg is None:
        reg = None
    else:
        ValueError('Wrong regularizers')

    monomial_dim = get_dim(sys_dim, deg_ftrs, rm_one)
    phi = Input(shape=(monomial_dim,), name='phi')
    layers = [
        Dense(monomial_dim * over_para, use_bias=False,)
        # Dense((monomial_dim * 2), use_bias=False),
    ]
    gram_factor = Sequential(layers, name='gram_factorization')
    phiL = gram_factor(phi)  # (None, monomial_dim)
    V = Dot(1, name='V')([phiL, phiL])  # (None,1)

    loop_closed = system.loop_closed
    # depending on if the loop is closed, only eta term would differ
    if not loop_closed:
        u_dim = system.num_inputs
        deg_u = system.deg_u
        ubasis_dim = get_dim(sys_dim, deg_u, True)
        ubasis = Input(shape=(ubasis_dim,), name='ubasis')
        u = Dense(u_dim, use_bias=False, name='u')(ubasis)
        g = Input(shape=(sys_dim,), name='open_loop_dynamics')
        if hasattr(system, 'B_noneConstant'):
            B = Input(shape=(sys_dim,u_dim), name='B')
            Bu = Dot(-1, name ='Bu')([B,u])
        else:
            B = system.hx(None).T
            Bu = DotKernel(B, name='Bu')(u)
        f_cl = Add(name='closed_loop_dynamics')([g, Bu])
        dphidx = Input(shape=(monomial_dim, sys_dim), name='dphidx')
        eta = Dot(-1, name='eta')([dphidx, f_cl])
        if hasattr(system, 'B_noneConstant'):
            features = [g, B, phi, dphidx, ubasis]
        else:
            features = [g, phi, dphidx, ubasis]

    else:
        eta = Input(shape=(monomial_dim,), name='eta')
        features = [phi, eta]

    # Vdot
    etaL = gram_factor(eta)  # (None, monomial_dim)
    Vdot = Dot(1, name='Vdot')([phiL, etaL])  # (None,1)

    Vsqured = Power(2, name='Vsqured')(V)  # (None,1)

    # Vdot_sign = Sign(name='Vdot_sign')(Vdot)
    # V_signed = Dot(1, name='Vsigned')([V, Vdot_sign])
    # min_pos = Min_Positive(name='V-flipped')(rate)
    # diff = Subtract()([rate, min_pos])
    # rectified_V = ReLu(name = 'rectified_V')(V)
    rate = Divide(name='rate')([Vdot, V])
    model = Model(inputs=features, outputs=rate)
    model.compile(loss=max_pred, metrics=[max_pred, mean_pred, neg_percent],
                  optimizer='adam')
    model.summary()
    print(model.loss)
    return model


def get_V(system, train_or_load, over_para=2, reg=None, **kwargs):
    deg_ftrs = system.deg_ftrs
    rm_one = system.rm_one
    loop_closed = system.loop_closed

    tag = '_degV' + str(2 * deg_ftrs)
    model_dir = '../data/' + system.name
    nx = system.num_states

    if not loop_closed:
        tag = tag + 'degU' + str(system.deg_u)
    if rm_one:
        tag = tag + '_rm'
    x_path = model_dir + '/stableSamplesSlice12.npy'
    if loop_closed and os.path.exists(x_path):
        train_x = np.load(x_path)
        n_samples = train_x.shape[0]
        assert(train_x.shape[1] == nx)
        print('x size %s' % str(train_x.shape))
    else:
        train_x = None

    if train_or_load is 'Train':
        file_path = model_dir + '/features' + tag + '.npz'
        if os.path.exists(file_path):
            features = load_features_dataset(file_path, loop_closed)
        else:
            features = system.features_at_x(train_x, file_path)

        n_samples = features[0].shape[0]
        y = - np.ones((n_samples,))
        model = model_V(system, over_para, reg, **kwargs)
        history = model.fit(features, y, **kwargs)
        # assert (history.history['loss'][-1] <= 0)
        # bad_samples, bad_predictions = eval_model(
        #     model, system, train_x, features=features)
        model_file_name = model_dir + '/V_model' + tag + '.h5'
        model.save(model_file_name)
        print('Saved model ' + model_file_name + ' to disk')

    elif train_or_load is 'Load':
        # TODO: decide if should save the model or directly save V
        # train_x = None
        model = get_V_model(model_dir, tag)

    P, u_weights = get_model_weights(model, loop_closed)
    if not loop_closed:
        system.close_the_loop(u_weights)
    V, Vdot = system.P_to_V(P, samples=train_x)
    # test_model(model, system, V, Vdot, x=None)
    return V, Vdot, system, model, P, u_weights


def get_model_weights(model, loop_closed):
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    u_weights, gram_weights = [], []
    for name, weight in zip(names, weights):
        if name.startswith('u'):
            u_weights.append(weight)
        else:
            gram_weights.append(weight)

    if len(gram_weights) == 1:
        L = gram_weights[0]
    else:
        L = np.linalg.multi_dot(gram_weights)

    if len(u_weights) == 0:
        assert loop_closed
        return L@L.T, None
    elif len(u_weights) == 1:
        return L@L.T, u_weights[0]
    else:
        return L@L.T, np.linalg.multi_dot(u_weights)


def get_V_model(model_dir, tag):
    file_name = model_dir + '/V_model' + tag + '.h5'
    with CustomObjectScope({'Divide': Divide, 'DotKernel': DotKernel, 'Power':
                            Power, 'max_pred': max_pred, 'mean_pred': mean_pred,
                            'neg_percent': neg_percent, 'mean_pos': mean_pos,
                            'mean_neg': mean_neg, 'reg_u': reg_u, 'reg_L': reg_L}):
        model = load_model(file_name)
    print('Loaded model ' + file_name)
    model.summary()
    print(model.loss)
    return model


def load_features_dataset(file_path, loop_closed):
    l = np.load(file_path)
    if loop_closed:
        features = [l['phi'], l['eta']]
    else:
        if len(l.files) ==5:
            features = [l['g'], l['B'],l['phi'], l['dphidx'], l['ubasis']]
        else:
            features = [l['g'], l['phi'], l['dphidx'], l['ubasis']]
    return features


def eval_model(model, system, x, features=None):
    if features is None:
        features = system.features_at_x(x)
    predicted = model.predict(features)
    bad_samples = x[predicted.ravel() > 0]
    bad_predictions = predicted[predicted.ravel() > 0]
    [scatterSamples(bad_samples, system, slice_idx=i)
     for i in system.all_slices]
    return bad_samples, bad_predictions


def test_model(model, system, V, Vdot, x=None):
    if x is None:
        n_tests = 3
        test_x = np.random.randn(n_tests, system.num_states)
    else:
        test_x = x
    test_features = system.features_at_x(test_x)
    test_prediction = model.predict(test_features)
    test_V = system.get_v_values(test_x, V=V)
    test_Vdot = system.get_v_values(test_x, V=Vdot)
    return [test_prediction, test_V, test_Vdot]


class reg_u(regularizers.Regularizer):
    """Regularizer for scaled L1 and L2 regularization.
    """

    def __init__(self, scaling=None, l1=0.01, l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.scaling = scaling

    def __call__(self, x):
        regularization = 0.
        scaled = K.dot(K.constant(self.scaling), x)
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(scaled))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(scaled))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2),
                'scaling': self.scaling}


class reg_L(regularizers.Regularizer):
    """Regularizer for scaled L1 and L2 regularization.
    """

    def __init__(self, dim1, dim2, l2=0.01):
        self.l2 = K.cast_to_floatx(l2)
        scaling = np.zeros((dim1, dim2))
        scaling[-1] = np.ones((dim2,))
        self.scaling = scaling

    def __call__(self, x):
        regularization = 0.
        scaled = K.constant(self.scaling) * x
        regularization += K.sum(self.l2 * K.square(scaled))
        return regularization

    def get_config(self):
        return {'l2': float(self.l2),
                'scaling': self.scaling}


def get_dim(num_var, deg, rm_one):
    if rm_one:
        return fact(num_var + deg) // fact(num_var) // fact(deg) - 1
    else:
        return fact(num_var + deg) // fact(num_var) // fact(deg)


# def gram_decomp_model_for_levelsetpoly(sys_dim, sigma_deg, psi_deg):
#     psi_dim = get_dim(sys_dim, psi_deg)
#     psi = Input(shape=(psi_dim,), name='psi')
#     layers = [
#         Dense(psi_dim, use_bias=False),
#         # Dense(math.floor(psi_dim / 2), use_bias=False),
#         # Dense(10, use_bias=False),
#         # Dense(4, use_bias=False),
#     ]
#     layers = layers + [TransLayer(i) for i in layers[::-1]]
#     psiLL = Sequential(layers, name='Gram')(psi)  # (None,
#     # monomial_dim)
#     candidateSOS = Dot(1)([psiLL, psi])  # (None,1)

#     xxd = Input(shape=(1,), name='xxd')
#     V = Input(shape=(1,), name='V')
#     Vdot = Input(shape=(1,), name='Vdot')

#     xxdV = Dot(-1)([xxd, V])

#     sigma_dim = get_dim(sys_dim + sigma_deg)
#     sigma = Input(shape=(sigma_dim,), name='sigma')

#     multiplierLayers = [
#         # Dense(sigma_dim, use_bias=False),
#         # Dense(4, use_bias=False),
#         Dense(1, use_bias=False),
#     ]
#     L1 = Sequential(multiplierLayers, name='multiplier')(sigma)
#     L1Vdot = Dot(-1, name='L1Vdot')([L1, Vdot])

#     rholayer = [Dense(1, use_bias=False, kernel_regularizer=rho_reg)]
#     # kernel_regularizer=rho_reg
#     rholayer = rholayer + [TransLayer(i) for i in rholayer[::-1]]

#     # residual = sos - (xxdV-xxdrho+L1Vdot)
#     # if use xxd(V-rho)
#     # xxdrho = Sequential(rholayer, name='rho')(xxd)
#     # vminusrho = Subtract()([xxdV, xxdrho])
#     # if use xxd(rho*V - 1)
#     xxdVrho = Sequential(rholayer, name='rho')(xxdV)
#     vminusrho = Subtract()([xxdVrho, xxd])
#     vminusrhoplusvdot = Add()([vminusrho, L1Vdot])
#     ratio = Divide()([vminusrhoplusvdot, candidateSOS])
#     # outputs = Dot(-1)([ratio, xxdrho])
#     model = Model(inputs=[V, Vdot, xxd, psi, sigma], outputs=ratio)
#     model.compile(loss='mse', metrics=[mean_pred, 'mse'],
#                   optimizer='adam')
#     model.summary())
#     return model


# def linear_model_for_V(sys_dim, A):
#     x = Input(shape=(sys_dim,))  # (None, sys_dim)
#     layers = [
#         Dense(5, input_shape=(sys_dim,), use_bias=False,
#               kernel_regularizer=regularizers.l2(0.)),
#         Dense(2, use_bias=False, kernel_regularizer=regularizers.l2(0.)),
#         Dense(2, use_bias=False, kernel_regularizer=regularizers.l2(0.))
#     ]
#     layers = layers + [TransLayer(i) for i in layers[::-1]]
#     xLL = Sequential(layers)(x)  # (None, sys_dim)
#     # need to avoid 0 in the denominator (by adding a strictly positive scalar
#     # to V)
#     V = Dot(1)([xLL, x])  # (None, 1)
#     xLLA = DotKernel(A)(xLL)  # A: (sys_dim,sys_dim), xLLA: (None, sys_dim)
#     Vdot = Dot(1)([xLLA, x])  # Vdot: (None,1)
#     rate = Divide()([Vdot, V])
#     model = Model(inputs=x, outputs=rate)
#     model.compile(loss=max_pred, optimizer='adam')
#     model.summary())
#     return model


# def test_idx(y_true,y_pred):
    # return K.gather(y_pred,1)

# def hinge_and_max(y_true, y_pred):
#     return K.max(y_pred) + 10 * K.mean(K.maximum(1. - y_true * y_pred, 0.),
#                                        axis=-1)

# def guided_MSE(y_true, y_pred):
#     # want slighty greater than one
#     if K.sign(y_pred - y_true) is 1:
#         return 0.1 * K.square(y_pred - y_true)
#     else:
#         return K.square(y_pred - y_true)

# def max_and_sign(y_true, y_pred):
#     # return K.cast(K.equal(y_true, K.sign(y_pred)), K.floatx()) +
#     # .001*K.max(y_pred)
#     return K.sign(y_pred)

# def get_gram_trans_for_levelset_poly(model):
#     names = [weight.name for layer in model.layers for weight in layer.weights]
#     weights = model.get_weights()
#     gram_weights = []
#     rho_weights = []
#     L_weights = []
#     for name, weight in zip(names, weights):
#         if 'Gram' in name:
#             gram_weights = gram_weights + [weight]
#         elif 'rho' in name:
#             rho_weights = rho_weights + [weight]
#         elif 'multiplier' in name:
#             L_weights = L_weights + [weight]
#         else:
#             print('should not have other weights')
#     if len(gram_weights) == 1:
#         g = gram_weights[0]
#     else:
#         g = np.linalg.multi_dot(gram_weights)
#     gram = g@g.T
#     print('cond # of the candidate gram: %s' %np.linalg.cond(gram))
#     rho = rho_weights[0]**2
#     if len(L_weights) == 1:
#         L = L_weights[0]
#     else:
#         L = np.linalg.multi_dot(L_weights)
#     return [gram, g, rho, L]
