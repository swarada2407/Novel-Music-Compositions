
import os
import uuid

import numpy as np
import soundfile as sf
import torch

from flask import Flask, render_template, request, url_for
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel

app = Flask(__name__)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "generated")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model settings
BASE_MODEL = "facebook/musicgen-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print("Using device:", DEVICE)
print("Loading MusicGen processor...")

processor = AutoProcessor.from_pretrained(MODEL_DIR)

print("Loading pretrained base model...")
base_model = MusicgenForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

print("Loading saved LoRA adapter...")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)

model = model.to(DEVICE)
model.eval()

SAMPLE_RATE = processor.feature_extractor.sampling_rate

print("MusicGen model loaded successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_music():
    prompt = request.form.get("prompt", "").strip()

    try:
        duration = int(request.form.get("duration", "5"))
    except ValueError:
        duration = 5

    if not prompt:
        return render_template(
            "index.html",
            error="Please enter a music description."
        )

    if duration not in (5, 10, 15):
        duration = 5

    try:
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)

        # Keep generation responsive: CPU inference is slow with long token counts.
        # Using a smaller cap makes audio generation finish much faster.
        max_new_tokens = min(duration * 4, 40)

        with torch.inference_mode():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=2,
                max_new_tokens=max_new_tokens
            )

        audio = audio_values[0, 0].detach().cpu().numpy()
        audio = np.nan_to_num(audio)

        filename = f"music_{uuid.uuid4().hex[:10]}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)

        sf.write(filepath, audio, SAMPLE_RATE)

        audio_url = url_for("static", filename=f"generated/{filename}")

        return render_template(
            "index.html",
            audio_url=audio_url,
            prompt=prompt,
            duration=duration,
            message="Music generated successfully!"
        )

    except Exception as error:
        print("Generation error:", repr(error))

        return render_template(
            "index.html",
            error="Music generation failed. Please check the terminal for details."
        )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
