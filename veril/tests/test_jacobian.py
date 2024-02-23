
from keras import backend as K

import numpy as np
from tensorflow.python.ops.parallel_for.gradients import jacobian
np_x=np.reshape(np.array([1,2,3]),(1, 3))
xx = (K.variable(np_x))
scale = K.reshape(K.variable(np.array([1,2,3,2,3,4,5,6,8])), (3, 3))
print(K.eval(scale))

yy = K.dot(xx,scale)

# print(K.eval(yy))

J = jacobian(yy, xx)
print(K.eval(J))
J_reshape = K.transpose(K.reshape(J, (3,3)))
npJ=K.eval(J_reshape)
print(npJ)

# print((np_x@npJ))



