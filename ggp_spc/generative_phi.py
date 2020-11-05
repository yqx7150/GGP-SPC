import numpy as np
import scipy.io as io
from scipy.linalg import norm,orth
Phi = np.zeros((4915,49152))
Phi[:,:] = orth(np.random.randn(4915,49152).T).T
io.savemat('./Phi.mat',{'Phi':Phi})Phi = np.zeros((19660,49152))
#Phi[:,:] = orth(np.random.randn(19660,49152).T).T
#io.savemat('./Phi.mat',{'Phi':Phi})
