import tifffile
import numpy as np
import torch

def load_tif_images_as_tensor(file_path):
    # Load TIF images using tifffile
    tif_stack = tifffile.imread(file_path)

    # Convert the NumPy array to a PyTorch tensor
    torch_tensor = torch.from_numpy(tif_stack)

    return torch_tensor

# Specify the path to your TIF file
tif_file_path = '/home/sergios/dataset/S2A_MSIL1C_20190319T095031_N0207_R079_T33SUB_20190319T133450_7936_10752.tif'

# Load TIF images as a PyTorch tensor
image_tensor = load_tif_images_as_tensor(tif_file_path)

# Print the shape of the resulting tensor
print("Tensor shape:", image_tensor.shape)

image_tensor = np.transpose(image_tensor, (2, 0, 1))
image_tensor = image_tensor[:, :120, :120]

print("Tensor shape:", image_tensor.shape)

pad_width = ((0, 10 - image_tensor.shape[0]), (0, 0), (0, 0))
image_tensor = np.pad(image_tensor, pad_width, mode='constant', constant_values=0)

print("Tensor shape:", image_tensor.shape)

print(image_tensor)
