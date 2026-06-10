import torch

from utils import load_config
from model import get_model


def main():
    config = load_config("configs/config.yaml")

    device = torch.device("cpu")

    model = get_model(
        config,
        load_v1=False,
    )

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=device,
        )
    )

    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        224,
        224,
    )

    torch.onnx.export(
        model,
        dummy_input,
        "best_model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print("ONNX export successful!")


if __name__ == "__main__":
    main()
