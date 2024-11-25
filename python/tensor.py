import ctypes
import os
from typing import Union
from pprint import pformat

class C_Tensor(ctypes.Structure):
    _fields_ = [
        ('ndim', ctypes.c_int),
        ('size', ctypes.c_int),
        ('data', ctypes.POINTER(ctypes.c_double)),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('strides', ctypes.POINTER(ctypes.c_int)),
    ]
    
class Tensor:
    script_dir = os.path.dirname(__file__)
    lib_path = os.path.join(script_dir, "libtensor.so")
    _C = ctypes.CDLL(lib_path)

    def __init__(self, data:list):

        self._input_data = data
        self._flattened_data = self._flatten(self._input_data)
        c_data = (ctypes.c_float * len(self._flattened_data))(*self._flattened_data.copy())
        
        def get_shape(nested_list:list):
            shape = []
            if isinstance(nested_list[0], list):
                shape.append(len(nested_list))
                shape += get_shape(nested_list[0])
            else:
                shape.append(len(nested_list))
            return shape
        
        
        self.shape = get_shape(self._input_data)
        c_shape = (ctypes.c_int * len(self.shape))(*self.shape.copy())

        self.ndim = len(self.shape)
        c_ndim = ctypes.c_int(self.ndim)

        Tensor._C.create_tensor.restype = ctypes.POINTER(C_Tensor)
        Tensor._C.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.tensor = Tensor._C.create_tensor(c_data, c_shape, c_ndim)

    def __del__(self) -> None:
        if hasattr(self, 'tensor'):
            Tensor._C.delete_tensor.restype = None
            Tensor._C.delete_tensor.argtypes = [ctypes.POINTER(C_Tensor)]
            Tensor._C.delete_tensor(self.tensor)

    def _flatten(self, data:list) -> list:
        if isinstance(data[0], Union[int, float]):
            return data
        flattened_data = []
        for element in data:
            if isinstance(element[0], list):
                flattened_data += self._flatten(element)
            else:
                flattened_data += element
        return flattened_data
    
    def get_strides(self) -> list:
        return self.tensor.contents.strides[:self.ndim]
    
    def __str__(self) -> str:
        return pformat(self._input_data, indent=1, width=10 * self.ndim)