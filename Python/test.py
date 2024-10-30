import torch

string = ""

for i in str(torch.utils.cmake_prefix_path):
    if i == "\\":
        string += "\\\\"
    else:
        string += i
print(string)