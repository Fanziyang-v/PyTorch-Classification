from models.googlenet import GoogLeNet
import torch

model = GoogLeNet(3, 1000, use_aux=True)

data = torch.rand(2, 3, 224, 224)
out, out_aux1, out_aux2 = model(data)
print(out.shape, out_aux1.shape, out_aux2.shape)
