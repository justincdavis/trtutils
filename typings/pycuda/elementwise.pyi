"""
This type stub file was generated by pyright.
"""

from pycuda.tools import context_dependent_memoize
from pytools import memoize_method

"""Elementwise functionality."""
__copyright__ = ...
__license__ = ...
def get_elwise_module(arguments, operation, name=..., keep=..., options=..., preamble=..., loop_prep=..., after_loop=...): # -> SourceModule:
    ...

def get_elwise_range_module(arguments, operation, name=..., keep=..., options=..., preamble=..., loop_prep=..., after_loop=...): # -> SourceModule:
    ...

def get_elwise_kernel_and_types(arguments, operation, name=..., keep=..., options=..., use_range=..., **kwargs): # -> tuple[SourceModule, Any, list[ScalarArg | VectorArg] | Any]:
    ...

def get_elwise_kernel(arguments, operation, name=..., keep=..., options=..., **kwargs):
    """Return a L{pycuda.driver.Function} that performs the same scalar operation
    on one or several vectors.
    """
    ...

class ElementwiseKernel:
    def __init__(self, arguments, operation, name=..., keep=..., options=..., **kwargs) -> None:
        ...
    
    def get_texref(self, name, use_range=...):
        ...
    
    @memoize_method
    def generate_stride_kernel_and_types(self, use_range): # -> tuple[SourceModule, Any, list[ScalarArg | VectorArg] | Any]:
        ...
    
    def __call__(self, *args, **kwargs): # -> None:
        ...
    


@context_dependent_memoize
def get_take_kernel(dtype, idx_dtype, vec_count=...): # -> tuple[Any, list[Any]]:
    ...

@context_dependent_memoize
def get_take_put_kernel(dtype, idx_dtype, with_offsets, vec_count=...): # -> tuple[Any, list[Any]]:
    ...

@context_dependent_memoize
def get_put_kernel(dtype, idx_dtype, vec_count=...):
    ...

@context_dependent_memoize
def get_copy_kernel(dtype_dest, dtype_src):
    ...

@context_dependent_memoize
def get_linear_combination_kernel(summand_descriptors, dtype_z): # -> tuple[Any, list[Any]]:
    ...

@context_dependent_memoize
def get_axpbyz_kernel(dtype_x, dtype_y, dtype_z, x_is_scalar=..., y_is_scalar=...):
    """
    Returns a kernel corresponding to ``z = ax + by``.

    :arg x_is_scalar: A :class:`bool` which is *True* only if `x` is a scalar :class:`gpuarray`.
    :arg y_is_scalar: A :class:`bool` which is *True* only if `y` is a scalar :class:`gpuarray`.
    """
    ...

@context_dependent_memoize
def get_axpbz_kernel(dtype_x, dtype_z):
    ...

@context_dependent_memoize
def get_binary_op_kernel(dtype_x, dtype_y, dtype_z, operator, x_is_scalar=..., y_is_scalar=...):
    """
    Returns a kernel corresponding to ``z = x (operator) y``.

    :arg x_is_scalar: A :class:`bool` which is *True* only if `x` is a scalar :class:`gpuarray`.
    :arg y_is_scalar: A :class:`bool` which is *True* only if `y` is a scalar :class:`gpuarray`.
    """
    ...

@context_dependent_memoize
def get_rdivide_elwise_kernel(dtype_x, dtype_z):
    ...

@context_dependent_memoize
def get_binary_func_kernel(func, dtype_x, dtype_y, dtype_z):
    ...

@context_dependent_memoize
def get_binary_func_scalar_kernel(func, dtype_x, dtype_y, dtype_z):
    ...

def get_binary_minmax_kernel(func, dtype_x, dtype_y, dtype_z, use_scalar):
    ...

@context_dependent_memoize
def get_fill_kernel(dtype):
    ...

@context_dependent_memoize
def get_reverse_kernel(dtype):
    ...

@context_dependent_memoize
def get_real_kernel(dtype, real_dtype):
    ...

@context_dependent_memoize
def get_imag_kernel(dtype, real_dtype):
    ...

@context_dependent_memoize
def get_conj_kernel(dtype, conj_dtype):
    ...

@context_dependent_memoize
def get_arange_kernel(dtype):
    ...

@context_dependent_memoize
def get_pow_array_kernel(dtype_x, dtype_y, dtype_z, is_base_array, is_exp_array):
    """
    Returns the kernel for the operation: ``z = x ** y``
    """
    ...

@context_dependent_memoize
def get_fmod_kernel():
    ...

@context_dependent_memoize
def get_modf_kernel():
    ...

@context_dependent_memoize
def get_frexp_kernel():
    ...

@context_dependent_memoize
def get_ldexp_kernel():
    ...

@context_dependent_memoize
def get_unary_func_kernel(func_name, in_dtype, out_dtype=...):
    ...

@context_dependent_memoize
def get_if_positive_kernel(crit_dtype, dtype):
    ...

@context_dependent_memoize
def get_where_kernel(crit_dtype, dtype):
    ...

@context_dependent_memoize
def get_scalar_op_kernel(dtype_x, dtype_a, dtype_y, operator):
    ...

@context_dependent_memoize
def get_logical_not_kernel(dtype_x, dtype_out):
    ...
