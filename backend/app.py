import os
from pathlib import Path

import gradio as gr
from gradio_client import Client, handle_file


# ============================================================
# NOVIX - QWEN IMAGE EDIT ANYPOSE
# ============================================================

MODEL_SPACE = "abidlabs/Qwen-Image-Edit-2511-AnyPose"

MODEL_API = "/infer"

HF_TOKEN = os.getenv("HF_TOKEN", "")


# ============================================================
# FIXED AI INSTRUCTION
# ============================================================
# User को यह prompt दिखाई नहीं देगा.
# Backend इसे automatically भेजेगा.

FIXED_PROMPT = """
Make the person in image 1 do the exact same pose of the person in image 2.
Changing the style and background of the image of the person in image 1 is undesirable, so don't do it.
The new pose should be pixel accurate to the pose we are trying to copy.
The position of the arms and head and legs should be the same as the pose we are trying to copy.
Change the field of view and angle to match exactly image 2.
Head tilt and eye gaze pose should match the person in image 2.
Remove the background of image 2, and replace it with the background of image 1.
Don't change the identity of the person in image 1, keep their appearance the same.
Don't change their facial features or hair style.
"""


# ============================================================
# POSES FOLDER
# ============================================================

POSES_DIR = Path(__file__).resolve().parent.parent / "poses"


def find_pose(pose_number):
    """Find Pose 1-50 from the GitHub poses folder."""

    pose_number = int(pose_number)

    if pose_number < 1 or pose_number > 50:
        raise ValueError("Pose number must be between 1 and 50.")

    # Supports PNG, JPG, JPEG and WEBP
    for extension in ["png", "jpg", "jpeg", "webp"]:

        pose_file = POSES_DIR / f"pose{pose_number}.{extension}"

        if pose_file.exists():
            return pose_file

    raise FileNotFoundError(
        f"Pose {pose_number} was not found."
    )


# ============================================================
# GENERATE IMAGE
# ============================================================

def generate_image(reference_image, pose_number):

    if reference_image is None:
        raise gr.Error(
            "Please upload your character image."
        )

    if pose_number is None:
        raise gr.Error(
            "Please select a pose."
        )

    # Find selected pose
    try:
        pose_file = find_pose(pose_number)

    except Exception as error:
        raise gr.Error(str(error))

    # Connect to Hugging Face Space
    try:

        if HF_TOKEN:

            client = Client(
                MODEL_SPACE,
                token=HF_TOKEN
            )

        else:

            client = Client(
                MODEL_SPACE
            )

    except Exception as error:

        raise gr.Error(
            f"Could not connect to AI model: {error}"
        )

    # ========================================================
    # SEND CHARACTER + POSE TO AI
    # ========================================================

    try:

        result = client.predict(
            reference_image=handle_file(
                reference_image
            ),

            pose_image=handle_file(
                str(pose_file)
            ),

            # Hidden fixed prompt.
            # User never sees this.
            prompt=FIXED_PROMPT,

            seed=0,

            randomize_seed=True,

            true_guidance_scale=1,

            num_inference_steps=4,

            height=1024,

            width=1024,

            rewrite_prompt=False,

            api_name=MODEL_API,
        )

    except Exception as error:

        raise gr.Error(
            f"Image generation failed: {error}"
        )

    # ========================================================
    # GET GENERATED IMAGE
    # ========================================================

    if not result:
        raise gr.Error(
            "The AI returned no image."
        )

    # API returns:
    #
    # [0] = Result Gallery
    # [1] = Seed
    #
    generated_result = result[0]

    # Result Gallery may contain multiple images.
    if isinstance(generated_result, list):

        if len(generated_result) == 0:
            raise gr.Error(
                "No generated image was returned."
            )

        return generated_result[0]

    return generated_result


# ============================================================
# BACKEND INTERFACE
# ============================================================

with gr.Blocks(
    title="Novix AI Anime Pose Generator"
) as demo:

    gr.Markdown(
        "# Novix AI Anime Pose Generator"
    )

    # Character image
    reference_image = gr.Image(
        label="Character Image",
        type="filepath",
        sources=["upload"],
    )

    # Pose number
    pose_number = gr.Number(
        label="Pose Number",
        minimum=1,
        maximum=50,
        precision=0,
        value=1,
    )

    # Generate
    generate_button = gr.Button(
        "Generate"
    )

    # Result
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
# START
# ============================================================

if __name__ == "__main__":

    demo.launch()
