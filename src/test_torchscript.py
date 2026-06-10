import torch

model = torch.jit.load("best_model_scripted.pt")

model.eval()

dummy = torch.randn(
    1,
    3,
    224,
    224,
)

with torch.no_grad():
    output = model(dummy)

print(output.shape)
