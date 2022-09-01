Jax
----
All JAX operations implemented in terms of XLA operations (Accelerated Linear Algebra Compiler)

``Jax.numpy vs jax.lax``
^^^^^^^^^^^^^^^^^^^^^^^^
    - jax.numpy: high level wrapper, familiar interface
    - jax.lax: lower api, stricter, more powerful

Implicitly promotes args to allow operations between mixed data type :code:`lax.add(1, 1.0)` throws an error. :code:`lax.add(jnp.float32(1), 1.0)` won't
e.g. Numpy convolution is under-the-hood translated to more general convolution, designed to be efficient for convolutions, like those found in neural nets
.. code:: python
    from jax import lax, numpy as jnp

    x = jnp.array([1, 2, 1])
    y = jnp.ones(10)
    jnp.convolve(x, y)

    result = lax.conv_general_dilated(
        x.reshape(1, 1, 3).astype(float),  # note: explicit promotion
        y.reshape(1, 1, 10),
        window_strides=(1,),
        padding=[(len(y) - 1, len(y) - 1)])  # equivalent of padding='full' in NumPy

    DeviceArray([1., 3., 4., 4., 4., 4., 4., 4., 4., 4., 3., 1.], dtype=float32)


``Jax.numpy vs Numpy``
^^^^^^^^^^^^^^^^^^^^^^
Jax.numpy closesly mirrors the numpy API.
- Jax arrays are always immutable
They are implemented as different Python types :code:`numpy.ndarray` vs :code:`jax.interpreters.xla._DeviceArray` but duck-typing prevents this being problematic
Jax does provide indexed update syntax
.. code:: python
    y = x.at[0].set(10)


``JIT Just in Time Compiler``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default JAX runs operations one-at-a-time, using JIT compilation decorator, seq of operations optimized to run at once. 
Requires array shapes to be static and known at compile time
.. code:: python
    from jax import numpy as jnp, jit
    def norm(X): return (X - X.mean(0)) / X.std(0)
    norm_compiled = jit(norm)

    X = jnp.array(np.random.rand(10000, 10))

    # Return same values
    np.allclose(norm(X), norm_compiled(X), atol=1E-6)

    # Compilation (fuse operations, avoid allocating temporary arrays) improves speed order of magnitudes
    %timeit norm(X).block_until_ready()
    %timeit norm_compiled(X).block_until_ready()

``JIT: tracing and static vars``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
JIT and JAX transforms, trace a function to determine its effects on specifc shape/type inputs
static marked variables are not traced.
Basic tracers are stand-ins that encode type/shape, but are agnostic to values
Recorded seq of computations can by efficiently applied within XLA to new inputs without re-executing python code
When a compiled function is recalled with new inputs, nothing is printed (if prints exist in function), as result computed in compiled XLA not python.
Seq of ops, encoded in JAX expression (jaxpr)

As JIT compilation is done without information on content of array, control flow statemnets in the function cannot depend on traced values.

``Sources``
^^^^^^^^^^^^^^^^^
https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html
