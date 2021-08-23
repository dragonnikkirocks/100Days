import torch
a = torch.ones(5)
print(a)
points = torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]])
print(points.storage())
print(points)
second_point= points[1]
print(second_point.storage_offset()) #storage offset is 2 as we need to skip two elements to reach points[1]
print(points.stride())