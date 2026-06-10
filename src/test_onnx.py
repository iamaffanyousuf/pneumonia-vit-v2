import onnx

model = onnx.load("best_model.onnx")

onnx.checker.check_model(model)

print("ONNX model is valid")
