import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr

from src.model import get_model
from src.utils import load_config

from huggingface_hub import hf_hub_download


# Load custom CSS
def load_css():
    css_path = os.path.join("frontend", "customCSS.css")

    with open(css_path, "r") as f:
        return f.read()


# Load model
model_path = hf_hub_download(
    repo_id="iamaffanyousuf/pneumonia-vit-model",
    filename="best_model.pth",
    cache_dir="models",
)

# Load config
config = load_config("configs/config.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_model(config)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Transform (same as training)
transform = transforms.Compose(
    [
        transforms.Resize((config["data"]["img_size"], config["data"]["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Prediction function
def predict(image: Image.Image) -> dict:
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

    pneumonia_prob = probs[0][1].item()
    normal_prob = probs[0][0].item()

    return {
        "PNEUMONIA": round(pneumonia_prob, 4),
        "NORMAL": round(normal_prob, 4),
    }


# Gradio UI — modern Blocks-based layout
with gr.Blocks(title="Pneumonia Detection (ViT)", css=load_css()) as app:
    gr.Markdown("""
    <h1>🫁 Pneumonia Detection AI</h1>
    <p>Vision Transformer-powered medical imaging assistant</p>
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Chest X-Ray")
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column():
            label_output = gr.Label(num_top_classes=2, label="Prediction")
            gr.Markdown("""
            <div class="disclaimer">
            ⚠️ <b>Medical Disclaimer:</b><br>
            This AI tool is for educational purposes only and should not be used for diagnosis.
            Consult a certified medical professional.
            </div>
            """)

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=label_output,
    )

app.launch()
