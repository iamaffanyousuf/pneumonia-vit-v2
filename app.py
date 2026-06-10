import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr
from src.model import get_model
from src.utils import load_config

CLASS_NAMES = [
    "Normal",
    "Pneumonia",
    "Effusion",
    "Nodule",
    "Mass",
    "Cardiomegaly",
    "Fibrosis",
    "Edema",
]

config = load_config("configs/config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "best_model.pth"
model = get_model(config, load_v1=False)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((config["data"]["img_size"], config["data"]["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict(image: Image.Image):
    if image is None:
        return gr.update(visible=False), gr.update(
            visible=True, value="⚠️ Please upload an image first."
        )

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)

    probs = probs.squeeze().cpu().numpy()
    results = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    return gr.update(visible=True, value=results), gr.update(visible=False)


def clear_all():
    return None, gr.update(visible=False), gr.update(visible=False)


custom_css = """
/* Center the main content block */
.main-block {
    max-width: 560px;
    margin: 0 auto !important;
    width: 100%;
    padding-top: 0 !important;
}

/* Upload area */
.upload-box .wrap {
    border: 2px dashed #334155 !important;
    width: 100% !important;
    border-radius: 16px !important;
    background: #1e2535 !important;
    transition: border-color 0.2s;
}
.upload-box .wrap:hover {
    border-color: #60a5fa !important;
}

/* Predict button */
#predict-btn {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.45) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
#predict-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.6) !important;
}
#predict-btn:active {
    transform: translateY(0px) !important;
}

/* Clear button */
#clear-btn {
    background: #1e2535 !important;
    color: #94a3b8 !important;
    border: 1.5px solid #334155 !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    transition: background 0.15s, color 0.15s !important;
}
#clear-btn:hover {
    background: #263046 !important;
    color: #cbd5e1 !important;
    border-color: #475569 !important;
}

/* Result card */
.result-card {
    background: #1e2535;
    border: 1px solid #334155;
    border-radius: 16px;
    margin-top: 0.5rem;
}

/* Notes */
.notes-section {
    background: #1a2030;
    border: 1px solid #2d3748;
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    margin-top: 1.5rem;
    color: #94a3b8;
    font-size: 0.92rem;
    line-height: 1.7;
}
.notes-section h3 {
    color: #cbd5e1;
    font-size: 1rem;
    margin-bottom: 0.5rem;
}
.notes-section ul { padding-left: 1.2rem; margin: 0; }
.notes-section li { margin-bottom: 0.25rem; }
.notes-section strong { color: #93c5fd; }
"""

with gr.Blocks(title="Chest X-Ray Disease Classification", css=custom_css) as app:

    # Replace the gr.HTML header block with this:
    gr.HTML("""
        <div style="text-align:center; padding: 1rem 1rem 0.8rem;">
            <h1 style="
                font-size: 1.75rem;
                font-weight: 600;
                color: #cbd5e1;
                letter-spacing: -0.2px;
                margin: 0 0 0.4rem;
                filter: grayscale(1) brightness(1.4);
            ">
                🫁 Chest X-Ray Detection AI
            </h1>
            <p style="color:#64748b; font-size:0.93rem; margin:0;">
                Upload a chest X-ray to detect possible findings using a Vision Transformer model.
            </p>
        </div>
    """)

    with gr.Column(elem_classes=["main-block"]):

        image_input = gr.Image(
            type="pil",
            label="Upload Chest X-ray",
            height=340,
            elem_classes=["upload-box"],
        )

        with gr.Row():
            submit_btn = gr.Button("Analyze", elem_id="predict-btn", variant="primary")
            clear_btn = gr.Button("Clear", elem_id="clear-btn")

        label_output = gr.Label(
            num_top_classes=5,
            label="Top Predictions",
            visible=False,
            elem_classes=["result-card"],
        )

        warn_text = gr.HTML(visible=False)

        gr.HTML("""
            <div class="notes-section">
                <h3>⚠️ Important Notes</h3>
                <ul>
                    <li>Model trained on <strong>8 classes</strong>: Normal, Pneumonia, Effusion, Nodule, Mass, Cardiomegaly, Fibrosis, Edema.</li>
                    <li>Current test accuracy is approximately <strong>76%</strong>.</li>
                    <li>Predictions are probability estimates and <strong>may be incorrect</strong>.</li>
                    <li>For <strong>educational and research purposes only</strong> — not for clinical diagnosis.</li>
                </ul>
            </div>
        """)

    submit_btn.click(fn=predict, inputs=image_input, outputs=[label_output, warn_text])
    clear_btn.click(
        fn=clear_all, inputs=[], outputs=[image_input, label_output, warn_text]
    )

app.launch()
