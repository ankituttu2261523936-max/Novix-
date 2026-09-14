import os
from pathlib import Path

import gradio as gr
from gradio_client import Client, handle_file


# ============================================================
# NOVIX CONFIG
# ============================================================

# बाद में यहाँ असली Hugging Face Space का नाम आएगा.
# अभी इसे खाली ही रहने दो.
MODEL_SPACE = os.getenv("MODEL_SPACE", "")

# बाद में "Use via API" से मिलने वाला endpoint यहाँ आएगा.
MODEL_API = os.getenv("MODEL_API", "")

# अगर model को Hugging Face token चाहिए तो बाद में
# environment variable के रूप में लगाया जा सकता है.
HF_TOKEN = os.getenv("HF_TOKEN", "")


# ============================================================
# POSE FILES
# ============================================================

# GitHub project में poses folder:
#
# Novix/
# ├── poses/
# │   ├── pose1.png
# │   ├── pose2.jpg
# │   └── ...
# └── backend/
#     └── app.py

POSES_DIR = Path(__file__).resolve().parent.parent / "poses"


def find_pose(pose_number):
    """Find the selected pose image."""

    pose_number = int(pose_number)

    if pose_number < 1 or pose_number > 50:
        raise ValueError("Pose must be between 1 and 50.")

    # PNG, JPG और JPEG सभी support होंगे.
    for extension in ["png", "jpg", "jpeg", "webp"]:
        pose_file = POSES_DIR / f"pose{pose_number}.{extension}"

        if pose_file.exists():
            return pose_file

    raise FileNotFoundError(
        f"Pose {pose_number} was not found in the poses folder."
    )


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(reference_image, pose_number):

    if reference_image is None:
        raise gr.Error("Please upload your character image.")

    if pose_number is None:
        raise gr.Error("Please select a pose.")

    # अभी API configured नहीं है.
    # सही two-image model मिलने के बाद ये values भरेंगे.
    if not MODEL_SPACE or not MODEL_API:
        raise gr.Error(
            "The Novix image-generation API is not connected yet."
        )

    try:
        pose_file = find_pose(pose_number)

    except Exception as error:
        raise gr.Error(str(error))

    # Hugging Face Space से connect करें.
    try:
        if HF_TOKEN:
            client = Client(
                MODEL_SPACE,
                token=HF_TOKEN
            )
        else:
            client = Client(MODEL_SPACE)

    except Exception as error:
        raise gr.Error(
            f"Could not connect to the AI model: {error}"
        )

    # Character image + selected pose image
    # model को भेजे जाएंगे.
    try:
        result = client.predict(
            reference_image=handle_file(reference_image),
            pose_image=handle_file(str(pose_file)),
            api_name=MODEL_API,
        )

    except Exception as error:
        raise gr.Error(
            f"Image generation failed: {error}"
        )

    # अगर API multiple outputs लौटाती है,
    # तो पहला output generated image माना जाएगा.
    if isinstance(result, (list, tuple)):
        if len(result) == 0:
            raise gr.Error("The AI returned no image.")

        return result[0]

    return result


# ============================================================
# NOVIX BACKEND INTERFACE
# ============================================================

with gr.Blocks(title="Novix AI Anime Pose Generator") as demo:

    gr.Markdown(
        "# Novix AI Anime Pose Generator"
    )

    # Character/reference image
    reference_image = gr.Image(
        label="Character Image",
        type="filepath",
        sources=["upload"],
    )

    # Pose 1-50
    pose_number = gr.Number(
        label="Pose Number",
        minimum=1,
        maximum=50,
        precision=0,
        value=1,
    )

    # Generate button
    generate_button = gr.Button(
        "Generate"
    )

    # Generated result
    generated_image = gr.Image(
        label="Generated Image",
        type="filepath",
    )

    generate_button.click(
        fn=generate_image,
        inputs=[
            reference_image,
            pose_number,
        ],
        outputs=generated_image,
        api_name="generate",
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    demo.launch()
