#%%
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax import jacfwd, jacrev
from jax.numpy import linalg

from numpy import nanargmin, nanargmax

key = random.PRNGKey(42)
#%%
# Grad - take automatic derivative of a function
def f(x): return 3 * x[0] ** 2
gradf = grad(f)
gradf(np.array([2.0]))
#%%
# Jacobian - automatic derivative of a function with vector input
def circle(x): return x[0] ** 2 + x[1] ** 2
J= jacfwd(circle)
J(np.array([1.0, 2.0])) 
#%%
# Compute the hessian of a function, by computing jacobian twice
def hessian(f): return jacfwd(jacrev(f))
H = hessian(circle)
H(np.array([1.0, 2.0]))
# Gradients are computed using automatic differentation, more accurate/efficient than finite distances

