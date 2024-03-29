"""
This type stub file was generated by pyright.
"""

from pytools import memoize_method

__copyright__ = ...
class vec:
    ...


def splay(n, dev=...): # -> tuple[tuple[Any | Literal[1], Literal[1]], tuple[Any | Literal[128], Literal[1], Literal[1]]]:
    ...

class GPUArray:
    """A GPUArray is used to do array-based calculation on the GPU.

    This is mostly supposed to be a numpy-workalike. Operators
    work on an element-by-element basis, just like numpy.ndarray.
    """
    __array_priority__ = ...
    def __init__(self, shape, dtype, allocator=..., base=..., gpudata=..., strides=..., order=...) -> None:
        ...
    
    @property
    def __cuda_array_interface__(self): # -> dict[str, Any]:
        """Returns a CUDA Array Interface dictionary describing this array's
        data."""
        ...
    
    @property
    def ndim(self): # -> int:
        ...
    
    @property
    @memoize_method
    def flags(self): # -> ArrayFlags:
        ...
    
    def set(self, ary, async_=..., stream=..., **kwargs): # -> None:
        ...
    
    def set_async(self, ary, stream=...): # -> None:
        ...
    
    def get(self, ary=..., pagelocked=..., async_=..., stream=..., **kwargs):
        ...
    
    def get_async(self, stream=..., ary=...):
        ...
    
    def copy(self): # -> GPUArray:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def __hash__(self) -> int:
        ...
    
    def __bool__(self): # -> bool:
        ...
    
    @property
    def ptr(self):
        ...
    
    def mul_add(self, selffac, other, otherfac, add_timer=..., stream=...):
        """Return `selffac * self + otherfac*other`."""
        ...
    
    def __add__(self, other): # -> GPUArray | _NotImplementedType:
        """Add an array with an array or an array with a scalar."""
        ...
    
    __radd__ = ...
    def __sub__(self, other): # -> GPUArray | _NotImplementedType:
        """Substract an array from an array or a scalar from an array."""
        ...
    
    def __rsub__(self, other):
        """Substracts an array by a scalar or an array::

        x = n - self
        """
        ...
    
    def __iadd__(self, other):
        ...
    
    def __isub__(self, other):
        ...
    
    def __pos__(self): # -> Self:
        ...
    
    def __neg__(self):
        ...
    
    def __mul__(self, other): # -> _NotImplementedType:
        ...
    
    def __rmul__(self, scalar):
        ...
    
    def __imul__(self, other):
        ...
    
    def __div__(self, other): # -> GPUArray | _NotImplementedType:
        """Divides an array by an array or a scalar::

        x = self / n
        """
        ...
    
    __truediv__ = ...
    def __rdiv__(self, other):
        """Divides an array by a scalar or an array::

        x = n / self
        """
        ...
    
    __rtruediv__ = ...
    def __idiv__(self, other): # -> Self:
        """Divides an array by an array or a scalar::

        x /= n
        """
        ...
    
    __itruediv__ = ...
    def fill(self, value, stream=...): # -> Self:
        """fills the array with the specified value"""
        ...
    
    def bind_to_texref(self, texref, allow_offset=...):
        ...
    
    def bind_to_texref_ext(self, texref, channels=..., allow_double_hack=..., allow_complex_hack=..., allow_offset=...):
        ...
    
    def __len__(self): # -> Integral:
        """Return the size of the leading dimension of self."""
        ...
    
    def __abs__(self): # -> Self:
        """Return a `GPUArray` of the absolute values of the elements
        of `self`.
        """
        ...
    
    def __pow__(self, other): # -> Self:
        """pow function::

        example:
                array = pow(array)
                array = pow(array,4)
                array = pow(array,array)

        """
        ...
    
    def __ipow__(self, other): # -> Self:
        """ipow function::

        example:
                array **= 4
                array **= array

        """
        ...
    
    def __rpow__(self, other): # -> Self:
        ...
    
    def reverse(self, stream=...): # -> Self:
        """Return this array in reversed order. The array is treated
        as one-dimensional.
        """
        ...
    
    def astype(self, dtype, stream=...): # -> GPUArray | Self:
        ...
    
    def any(self, stream=..., allocator=...): # -> empty:
        ...
    
    def all(self, stream=..., allocator=...): # -> empty:
        ...
    
    def reshape(self, *shape, **kwargs): # -> Self | GPUArray:
        """Gives a new shape to an array without changing its data."""
        ...
    
    def ravel(self): # -> Self | GPUArray:
        ...
    
    def view(self, dtype=...): # -> GPUArray:
        ...
    
    def squeeze(self): # -> GPUArray:
        """
        Returns a view of the array with dimensions of
        length 1 removed.
        """
        ...
    
    def transpose(self, axes=...): # -> GPUArray:
        """Permute the dimensions of an array.

        :arg axes: list of ints, optional.
            By default, reverse the dimensions, otherwise permute the axes
            according to the values given.

        :returns: :class:`GPUArray` A view of the array with its axes permuted.

        .. versionadded:: 2015.2
        """
        ...
    
    @property
    def T(self): # -> GPUArray:
        """
        .. versionadded:: 2015.2
        """
        ...
    
    def __getitem__(self, index): # -> GPUArray:
        """
        .. versionadded:: 2013.1
        """
        ...
    
    def __setitem__(self, index, value): # -> None:
        ...
    
    @property
    def real(self): # -> Self:
        ...
    
    @property
    def imag(self): # -> Self | GPUArray:
        ...
    
    def conj(self, out=...): # -> Self:
        ...
    
    conjugate = ...
    __eq__ = ...
    __ne__ = ...
    __le__ = ...
    __ge__ = ...
    __lt__ = ...
    __gt__ = ...


def to_gpu(ary, allocator=...): # -> GPUArray:
    """converts a numpy array to a GPUArray"""
    ...

def to_gpu_async(ary, allocator=..., stream=...): # -> GPUArray:
    """converts a numpy array to a GPUArray"""
    ...

empty = GPUArray
def zeros(shape, dtype=..., allocator=..., order=...): # -> GPUArray:
    """Returns an array of the given shape and dtype filled with 0's."""
    ...

def ones(shape, dtype=..., allocator=..., order=...): # -> GPUArray:
    """Returns an array of the given shape and dtype filled with 1's."""
    ...

def empty_like(other_ary, dtype=..., order=...): # -> GPUArray:
    ...

def zeros_like(other_ary, dtype=..., order=...): # -> GPUArray:
    ...

def ones_like(other_ary, dtype=..., order=...): # -> GPUArray:
    ...

def arange(*args, **kwargs): # -> GPUArray:
    """Create an array filled with numbers spaced `step` apart,
    starting from `start` and ending at `stop`.

    For floating point arguments, the length of the result is
    `ceil((stop - start)/step)`.  This rule may result in the last
    element of the result being greater than stop.
    """
    class Info(Record):
        ...
    
    

def take(a, indices, out=..., stream=...): # -> GPUArray:
    ...

def multi_take(arrays, indices, out=..., stream=...): # -> list[Any] | list[GPUArray]:
    ...

def multi_take_put(arrays, dest_indices, src_indices, dest_shape=..., out=..., stream=..., src_offsets=...): # -> list[Any] | list[GPUArray]:
    ...

def multi_put(arrays, dest_indices, dest_shape=..., out=..., stream=...): # -> list[Any] | list[GPUArray]:
    ...

def concatenate(arrays, axis=..., allocator=...): # -> empty:
    """
    Join a sequence of arrays along an existing axis.
    :arg arrays: A sequnce of :class:`GPUArray`.
    :arg axis: Index of the dimension of the new axis in the result array.
        Can be -1, for the new axis to be last dimension.
    :returns: :class:`GPUArray`
    """
    ...

def stack(arrays, axis=..., allocator=...): # -> empty:
    """
    Join a sequence of arrays along a new axis.
    :arg arrays: A sequnce of :class:`GPUArray`.
    :arg axis: Index of the dimension of the new axis in the result array.
        Can be -1, for the new axis to be last dimension.
    :returns: :class:`GPUArray`
    """
    ...

def transpose(a, axes=...):
    """Permute the dimensions of an array.

    :arg a: :class:`GPUArray`
    :arg axes: list of ints, optional.
        By default, reverse the dimensions, otherwise permute the axes
        according to the values given.

    :returns: :class:`GPUArray` A view of the array with its axes permuted.

    .. versionadded:: 2015.2
    """
    ...

def reshape(a, *shape, **kwargs):
    """Gives a new shape to an array without changing its data.

    .. versionadded:: 2015.2
    """
    ...

def if_positive(criterion, then_, else_, out=..., stream=...): # -> GPUArray:
    ...

def where(criterion, then_, else_, out=..., stream=...): # -> GPUArray:
    ...

minimum = ...
maximum = ...
def sum(a, dtype=..., stream=..., allocator=...): # -> empty:
    ...

def any(a, stream=..., allocator=...): # -> empty:
    ...

def all(a, stream=..., allocator=...): # -> empty:
    ...

def subset_sum(subset, a, dtype=..., stream=..., allocator=...): # -> empty:
    ...

def dot(a, b, dtype=..., stream=..., allocator=...): # -> empty:
    ...

def subset_dot(subset, a, b, dtype=..., stream=..., allocator=...): # -> empty:
    ...

_builtin_min = ...
_builtin_max = ...
min = ...
max = ...
subset_min = ...
subset_max = ...
def logical_and(x1, x2, /, out=..., *, allocator=...): # -> GPUArray:
    ...

def logical_or(x1, x2, /, out=..., *, allocator=...): # -> GPUArray:
    ...

def logical_not(x, /, out=..., *, allocator=...): # -> empty:
    ...

