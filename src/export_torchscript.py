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

    scripted_model = torch.jit.trace(
        model,
        dummy_input,
    )

    scripted_model.save("best_model_scripted.pt")

    print("TorchScript export successful!")


if __name__ == "__main__":
    main()
