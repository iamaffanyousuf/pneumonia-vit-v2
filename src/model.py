import torch
import timm


def get_model(config):
    model = timm.create_model(
        config["model"]["name"],
        pretrained=config["model"]["pretrained"],
        num_classes=config["model"]["num_classes"],
    )

    checkpoint_path = config["model"].get("v1_checkpoint")

    if checkpoint_path:
        print(f"Loading V1 checkpoint: {checkpoint_path}")

        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        # remove incompatible classifier head
        keys_to_remove = [k for k in state_dict.keys() if "head" in k]

        for k in keys_to_remove:
            del state_dict[k]

        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False,
        )

        print("\nLoaded pretrained backbone")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    return model
