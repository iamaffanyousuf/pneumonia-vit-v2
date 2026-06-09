from utils import load_config
from model import get_model

config = load_config("configs/config.yaml")

model = get_model(config)

print(model.head)
