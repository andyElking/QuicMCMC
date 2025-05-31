import jax
print("JAX version:   ", jax.__version__)
print("jaxlib version:", jax.lib.__version__)
print("CUDA visible to JAX:", jax.devices())      # or jax.local_devices()

import jax.numpy as jnp

# Force everything onto GPU explicitly:
a_gpu = jnp.asarray(jnp.eye(5) * 2.0 + 0.1 * jnp.arange(25).reshape(5, 5))
# Make it symmetric positive‐definite:
A = (a_gpu + a_gpu.T) / 2.0

try:
    L = jnp.linalg.cholesky(A)   # this invokes cuSolverDnPotrf (dense linear solve)
    print("Cholesky completed. L =\n", L)
except Exception as e:
    print("Cholesky on GPU failed with:\n", e)