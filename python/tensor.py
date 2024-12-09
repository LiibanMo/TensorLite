import os
import ctypes

class C_Tensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('strides', ctypes.POINTER(ctypes.c_int)),
        ('ndim', ctypes.c_int),
        ('size', ctypes.c_int),
    ]

class Tensor:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, 'libtensor.so')
    _C = ctypes.CDLL(lib_path)

    def __init__(self, data: list, shape:list):

        self._c_data = (len(data) * ctypes.c_float)(*data)
        self._c_shape = (len(shape) * ctypes.c_int)(*shape)
        self._c_ndim = ctypes.c_int(len(shape))

        self.shape = shape
        self.ndim = len(shape)

        Tensor._C.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        Tensor._C.create_tensor.restype = ctypes.POINTER(C_Tensor)

        self.tensor = Tensor._C.create_tensor(
            self._c_data,
            self._c_shape,
            self._c_ndim,
        )
        
