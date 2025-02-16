import torch

from tensorlite import Tensor

# Function(s) ------------------------------------------------------------------------


def compare_tensors(tensor1, tensor2, epsilon=1e-6) -> bool:
    diff = torch.abs(tensor1 - tensor2)
    return torch.all(diff < epsilon)


# Data -------------------------------------------------------------------------------

without_broadcasting_data = [
    # 1D tensors
    ([0, -1, 2], [3, -2, 1]),  # (3,)
    ([10, 20, 30, 40], [5, 10, 15, 20]),  # (4,)

    # 2D tensors
    ([[0, -1], [2, -3]], [[3, -2], [1, 4]]),  # (2,2)
    ([[10, 20, 30], [40, 50, 60]], [[5, 10, 15], [20, 25, 30]]),  # (2,3)

    # 3D tensors
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]),  # (2,2,2)
    ([[[0, -1], [2, -3], [0, 0]], [[4, -5], [6, -7], [0, 0]]], [[[8, -9], [10, -11], [1, 1]], [[12, -13], [14, -15], [1, 1]]]),  # (2,3,2)

    # 4D tensors
    ([[[[1, 2], [3, 4]]]], [[[[5, 6], [7, 8]]]]),  # (1,1,2,2)
    ([[[[[10, 20], [30, 40]]], [[[50, 60], [70, 80]]]]], [[[[5, 10], [15, 20]]], [[[25, 30], [35, 40]]]]),  # (2,1,2,2)
]

broadcasting_data = [
    # 2D, 1D broadcasting
    ([[1, 2, 3], [4, 5, 6]], [1, 2, 3]),  # (2,3) + (3,)
    ([1, 2, 3], [[1], [2], [3]]),  # (3,) + (3,1)

    # Higher dimensions
    ([[[1, 2, 3]], [[4, 5, 6]]], [1, 2, 3]),  # (2,1,3) + (3,)
    ([[[1]], [[2]]], [1, 2, 3]),  # (2,1,1) + (3,)
]


# IDs --------------------------------------------------------------------------------

without_broadcasting_ids = [
    "1D: (3,)",
    "1D: (4,)",
    "2D: (2,2)",
    "2D: (2,3)",
    "3D: (2,2,2)",
    "3D: (2,3,2)",
    "4D: (1,1,2,2)",
    "4D: (2,1,2,2)",
]

# addition ---------------------------------------------------------------------------

add_broadcasting_ids = [
    "2D + 1D",
    "1D + 2D",
    "3D + 1D: Normal Case",
    "3D + 1D: Special Case",
]

# subtraction ------------------------------------------------------------------------

sub_broadcasting_ids = [
    "2D - 1D",
    "1D - 2D",
    "3D - 1D: Normal Case",
    "3D - 1D: Special Case",
]

# multiplication ---------------------------------------------------------------------

mul_broadcasting_ids = [
    "2D * 1D",
    "1D * 2D",
    "3D * 1D: Normal Case",
    "3D * 1D: Special Case",
]

# division ---------------------------------------------------------------------------

div_broadcasting_ids = [
    "2D / 1D",
    "1D / 2D",
    "3D / 1D: Normal Case",
    "3D / 1D: Special Case",
]

# matmul -----------------------------------------------------------------------------

matmul_data = [
    # 1D @ 1D
    ([1, 2, 3], [4, 5, 6]),
    ([1], [2]),
    # 1D @ 2D
    ([1, 2, 3], [[4, 5], [6, 7], [8, 9]]),
    ([1, 1], [[1], [1]]),
    # 1D @ 3D
    ([1, 2], [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]),
    ([1], [[[1]]]),
    # 2D @ 1D
    ([[1, 2], [3, 4]], [5, 6]),
    ([[1, 2], [3, 4]], [1, 1]),
    # 2D @ 2D
    ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
    ([[1, 0], [0, 1]], [[5, 6], [7, 8]]),
    # 2D @ 3D
    ([[1, 2], [3, 4]], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    ([[1, 2]], [[[1, 2], [3, 4]]]),
    # 3D @ 1D
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [1, 1]),
    ([[[1, 2]]], [3, 4]),
    # 3D @ 2D
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], [[1, 2], [3, 4]]),
    ([[[1, 0], [0, 1]], [[1, 1], [1, 1]]], [[2, 3], [4, 5]]),
    # 3D @ 3D
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    ([[[1, 0], [0, 1]], [[1, 1], [1, 1]]], [[[1, 0], [0, 1]], [[1, 1], [1, 1]]]),
]

matmul_ids = [
    "1D @ 1D: Normal Case",
    "1D @ 1D: Edge Case",
    "1D @ 2D: Normal Case",
    "1D @ 2D: Edge Case",
    "1D @ 3D: Normal Case",
    "1D @ 3D: Edge Case",
    "2D @ 1D: Normal Case",
    "2D @ 1D: Edge Case",
    "2D @ 2D: Normal Case",
    "2D @ 2D: Edge Case",
    "2D @ 3D: Normal Case",
    "2D @ 3D: Edge Case",
    "3D @ 1D: Normal Case",
    "3D @ 1D: Edge Case",
    "3D @ 2D: Normal Case",
    "3D @ 2D: Edge Case",
    "3D @ 3D: Normal Case",
    "3D @ 3D: Edge Case",
]
