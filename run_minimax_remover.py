import os
import cv2
import math
import time
import random
import argparse
import tempfile

import numpy as np
import torch

from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler

from transformer_minimax_remover import Transformer3DModel
from pipeline_minimax_remover import Minimax_Remover_Pipeline

from moviepy.editor import ImageSequenceClip


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The original application was designed around this number of frames.
# Keep this as a safety limit unless your model supports longer sequences.
MAX_FRAMES = os.getenv("MAX_FRAMES", 201)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

def get_pipe():
    """
    Load the MiniMax-Remover pipeline.
    """
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    vae = AutoencoderKLWan.from_pretrained(
        "weights/vae",
        torch_dtype=dtype,
    )

    transformer = Transformer3DModel.from_pretrained(
        "weights/transformer",
        torch_dtype=dtype,
    )

    scheduler = UniPCMultistepScheduler.from_pretrained(
        "weights/scheduler"
    )

    pipe = Minimax_Remover_Pipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )

    pipe.to(DEVICE)

    return pipe


# -----------------------------------------------------------------------------
# Video loading
# -----------------------------------------------------------------------------

def load_video(video_path):
    """
    Load a video into memory as RGB numpy frames.

    Returns:
        frames: np.ndarray of shape [T, H, W, 3]
        fps: input video FPS
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 15.0

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # OpenCV gives BGR; MiniMax expects RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames found in video: {video_path}")

    frames = np.stack(frames, axis=0)

    return frames, fps


# -----------------------------------------------------------------------------
# Mask loading
# -----------------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
}


def binarize_mask(mask):
    """
    Convert a mask to uint8 binary format:

        0   -> background
        255 -> object to remove
    """
    return ((mask > 0).astype(np.uint8) * 255)


def load_single_mask(mask_path):
    """
    Load one mask image.

    Returns:
        mask: np.ndarray of shape [H, W]
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise RuntimeError(f"Could not read mask image: {mask_path}")

    return binarize_mask(mask)


def load_mask_video(mask_path):
    """
    Load a mask video.

    Every frame is converted into a binary [H, W] mask.

    Returns:
        masks: np.ndarray of shape [T, H, W]
    """
    cap = cv2.VideoCapture(mask_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open mask video: {mask_path}")

    masks = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert to grayscale if the mask video has 3 channels.
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = binarize_mask(frame)
        masks.append(frame)

    cap.release()

    if len(masks) == 0:
        raise RuntimeError(f"No mask frames found in: {mask_path}")

    masks = np.stack(masks, axis=0)

    return masks


def load_masks(mask_path, num_video_frames, video_height, video_width):
    """
    Load a ready mask.

    Supported cases:

    1. A single image:
       The same mask is applied to every video frame.

    2. A mask video:
       Must have the same number of frames and the same H/W
       as the input video.
    """
    extension = os.path.splitext(mask_path)[1].lower()

    if extension in IMAGE_EXTENSIONS:
        single_mask = load_single_mask(mask_path)

        mask_h, mask_w = single_mask.shape

        if (mask_h, mask_w) != (video_height, video_width):
            raise ValueError(
                "Mask size does not match video size.\n"
                f"Video size: {video_width}x{video_height}\n"
                f"Mask size:  {mask_w}x{mask_h}"
            )

        # Reuse the same mask for every frame.
        masks = np.repeat(
            single_mask[None, :, :],
            num_video_frames,
            axis=0,
        )

        return masks

    elif extension in VIDEO_EXTENSIONS:
        masks = load_mask_video(mask_path)

        mask_frames, mask_h, mask_w = masks.shape

        if mask_frames != num_video_frames:
            raise ValueError(
                "Mask video and input video have different numbers of frames.\n"
                f"Video frames: {num_video_frames}\n"
                f"Mask frames:  {mask_frames}"
            )

        if (mask_h, mask_w) != (video_height, video_width):
            raise ValueError(
                "Mask video dimensions do not match input video dimensions.\n"
                f"Video size: {video_width}x{video_height}\n"
                f"Mask size:  {mask_w}x{mask_h}"
            )

        return masks

    else:
        raise ValueError(
            f"Unsupported mask format: {extension}\n"
            "Use a mask image (.png/.jpg/...) or a mask video "
            "(.mp4/.avi/.mov/.mkv/...)."
        )


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------

def preprocess_for_removal(images, masks):
    """
    Resize images and masks to the resolution expected by MiniMax-Remover.

    Input:
        images: [T, H, W, 3], RGB uint8
        masks:  [T, H, W], binary uint8

    Output:
        image_tensor: [T, H', W', 3]
        mask_tensor:  [T, H', W', 1]
    """
    out_images = []
    out_masks = []

    for img, msk in zip(images, masks):

        # Landscape
        if img.shape[0] < img.shape[1]:
            target_width = 832
            target_height = 480

        # Portrait
        else:
            target_width = 480
            target_height = 832

        img_resized = cv2.resize(
            img,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )

        img_resized = img_resized.astype(np.float32) / 127.5 - 1.0
        out_images.append(img_resized)

        msk_resized = cv2.resize(
            msk,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

        # Binary mask: 0 = keep, 1 = remove.
        msk_resized = (msk_resized > 0).astype(np.float32)

        out_masks.append(msk_resized)

    images_np = np.stack(out_images, axis=0)
    masks_np = np.stack(out_masks, axis=0)

    image_tensor = torch.from_numpy(images_np)

    mask_tensor = torch.from_numpy(masks_np).unsqueeze(-1)

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    image_tensor = image_tensor.to(
        device=DEVICE,
        dtype=dtype,
    )

    mask_tensor = mask_tensor.to(
        device=DEVICE,
        dtype=dtype,
    )

    return image_tensor, mask_tensor


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

def remove_object(
    pipe,
    video_path,
    mask_path,
    output_path,
    dilation_iterations=6,
    num_inference_steps=6,
    seed=RANDOM_SEED,
):
    """
    Run MiniMax-Remover using a precomputed mask.
    """

    print(f"Loading video: {video_path}")
    images, fps = load_video(video_path)

    num_frames, video_height, video_width, _ = images.shape

    print(f"Video frames : {num_frames}")
    print(f"Video size   : {video_width} x {video_height}")
    print(f"Video FPS    : {fps:.3f}")

    # if num_frames > MAX_FRAMES:
    #     raise ValueError(
    #         f"Video contains {num_frames} frames, but this script is "
    #         f"configured for at most {MAX_FRAMES} frames.\n"
    #         "Increase MAX_FRAMES only if your MiniMax-Remover model "
    #         "supports longer sequences."
    #     )

    print(f"Loading mask: {mask_path}")

    masks = load_masks(
        mask_path=mask_path,
        num_video_frames=num_frames,
        video_height=video_height,
        video_width=video_width,
    )

    print(f"Mask shape   : {masks.shape}")

    # -------------------------------------------------------------------------
    # Prepare tensors
    # -------------------------------------------------------------------------

    image_tensor, mask_tensor = preprocess_for_removal(
        images,
        masks,
    )

    # Determine the model resolution.
    _, height, width, _ = image_tensor.shape

    print(f"Model input  : {width} x {height}")
    print(f"Device       : {DEVICE}")
    print(f"Dilation     : {dilation_iterations}")
    print(f"Inference steps: {num_inference_steps}")

    # -------------------------------------------------------------------------
    # Run MiniMax-Remover
    # -------------------------------------------------------------------------

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    print("Running MiniMax-Remover...")

    with torch.no_grad():
        result = pipe(
            images=image_tensor,
            masks=mask_tensor,
            num_frames=mask_tensor.shape[0],
            height=height,
            width=width,
            num_inference_steps=int(num_inference_steps),
            generator=generator,
            iterations=int(dilation_iterations),
        )

    output = result.frames[0]

    # Convert from [0, 1] float to uint8.
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------------
    # Save output video
    # -------------------------------------------------------------------------

    output_frames = [frame for frame in output]

    output_dir = os.path.dirname(os.path.abspath(output_path))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Writing output: {output_path}")

    clip = ImageSequenceClip(
        output_frames,
        fps=fps,
    )

    clip.write_videofile(
        output_path,
        codec="libx264",
        audio=False,
        verbose=False,
        logger=None,
    )

    clip.close()

    print("Done.")
    print(f"Output saved to: {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "MiniMax-Remover using a precomputed mask. "
            "No Gradio or SAM2 is required."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video.",
    )

    parser.add_argument(
        "--mask",
        required=True,
        help=(
            "Path to the ready mask. "
            "Can be a single mask image or a mask video."
        ),
    )

    parser.add_argument(
        "--save_path",
        default="./results",
        help="Save path.",
    )

    parser.add_argument(
        "--dilation-iterations",
        type=int,
        default=6,
        help="Mask dilation iterations. Default: 6",
    )

    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=6,
        help="Number of MiniMax inference steps. Default: 6",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed. Default: 42",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(
            f"Input video does not exist: {args.video}"
        )

    if not os.path.isfile(args.mask):
        raise FileNotFoundError(
            f"Mask does not exist: {args.mask}"
        )

    # Load model once.
    pipe = get_pipe()

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_path = os.path.join(args.save_path, video_name + "_MiniMax-Remover.mp4")

    remove_object(
        pipe=pipe,
        video_path=args.video,
        mask_path=args.mask,
        output_path=output_path,
        dilation_iterations=args.dilation_iterations,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()