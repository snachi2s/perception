import argparse
import os
import re

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

#================
# Configuration
#================
PALIGEMMA_MODEL          = "google/paligemma2-3b-mix-224"
PALIGEMMA_PROMPT         = "<image> detect camera;"
SAM_CHECKPOINT_PATH      = "../checkpoints/sam2.1_hiera_large.pt"
SAM_CONFIG_PATH          = "configs/sam2.1/sam2.1_hiera_l.yaml"
OUTPUT_BOX_PATH          = "detected_boxes.jpg"
DEVICE                   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#=============
# PaliGemma
#=============

def load_paligemma(model_id: str):
    """Load PaliGemma model + processor onto DEVICE."""
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    return model, processor

def detect_with_paligemma(image: Image.Image, model, processor, prompt: str) -> str:
    """Run PaliGemma and return its full decoded response."""
    inputs = processor(
        text=prompt,
        images=image,
        padding="longest",
        do_convert_rgb=True,
        return_tensors="pt",
    ).to(device=DEVICE, dtype=model.dtype)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_length=512)
    decoded = processor.decode(out_ids[0], skip_special_tokens=True)
    return decoded[len(prompt):].lstrip("\n")

def parse_boxes(response: str, img_size: tuple[int, int]) -> list[dict]:
    """
    Convert PaliGemma’s <loc> tags into pixel boxes.
    Returns [{'box':[x1,y1,x2,y2], 'label':str}, …].
    """
    w, h = img_size
    detections = []

    for part in response.split(" ; "):
        text = part.strip()
        if not text:
            continue

        coords = re.findall(r"<loc0*([0-9]+)>", text)
        if len(coords) == 3:  # sometimes the first <loc> tag is missing
            m = re.match(r"^(\d+)>", text)
            if m:
                coords.insert(0, m.group(1))

        if len(coords) != 4:
            print(f"Skipping malformed detection: '{text}'")
            continue

        y1, x1, y2, x2 = map(int, coords)
        # normalize from 0–1024 → paligemma token size
        x1 = int(x1 * w / 1024)
        y1 = int(y1 * h / 1024)
        x2 = int(x2 * w / 1024)
        y2 = int(y2 * h / 1024)
        label = text.split(">")[-1].strip()
        detections.append({"box": [x1, y1, x2, y2], "label": label})

    return detections

def draw_boxes(image_np: np.ndarray, detections: list[dict], out_path: str):
    canvas = image_np.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        lbl = det["label"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw, y1), (0, 0, 255), -1)
        cv2.putText(canvas, lbl, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Save with BGR conversion
    #cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    #print("Saved overlay →", out_path)


#================
# SAM2 Inference    
#================

def load_sam(checkpoint: str, config: str) -> SAM2ImagePredictor:
    """Load the SAM2 model and wrap it in a predictor."""
    model = build_sam2(config, checkpoint, device=DEVICE)
    return SAM2ImagePredictor(model)

def segment_with_sam(image_np: np.ndarray, predictor: SAM2ImagePredictor, detections: list[dict]) -> list[dict]:
    """
    For each box, predict a mask + score.
    Returns [{'box':…, 'label':…, 'mask':…, 'score':…}, …].
    """
    predictor.set_image(image_np)
    results = []

    for det in detections:
        box_arr = np.array(det["box"])[None, :]
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_arr,
            multimask_output=False
        )
        results.append({
            "box": det["box"],
            "label": det["label"],
            "mask": masks[0],
            "score": float(scores[0])
        })
    return results

def show_mask(mask: np.ndarray, ax, alpha=0.6):
    """Overlay a single mask on an axis, with contours."""
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[..., 2] = 1.0
    rgba[..., 3] = mask * alpha
    ax.imshow(rgba)

    cnts, _ = cv2.findContours(mask.astype(np.uint8),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        c = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        ax.plot(c[:, 0, 0], c[:, 0, 1], color="white", linewidth=2)

def display_segmentations(image_np: np.ndarray, segs: list[dict]):
    for seg in segs:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_np)
        show_mask(seg["mask"], ax)
        x1, y1, x2, y2 = seg["box"]
        ax.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            edgecolor="green", facecolor="none", lw=2
        ))
        ax.set_title(f"{seg['label']} (score {seg['score']:.3f})")
        ax.axis("off")
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Detect & segment objects with PaliGemma & SAM2"
    )
    parser.add_argument("image", help="Path to input image")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Usage: python {os.path.basename(__file__)} <image_path>")
        return

    input_image = Image.open(args.image).convert("RGB")
    image_numpy = np.array(input_image) # to use it for visualization
    w, h = input_image.size

    # paligemma2
    pg_model, pg_proc = load_paligemma(PALIGEMMA_MODEL)
    resp = detect_with_paligemma(input_image, pg_model, pg_proc, PALIGEMMA_PROMPT)
    print("PaliGemma says:", resp)

    boxes = parse_boxes(resp, (w, h))
    print(f"Found {len(boxes)} boxes:", boxes)

    del pg_model, pg_proc
    torch.cuda.empty_cache()

    draw_boxes(image_numpy, boxes, OUTPUT_BOX_PATH)

    # SAM2
    sam_pred = load_sam(SAM_CHECKPOINT_PATH, SAM_CONFIG_PATH)
    segs = segment_with_sam(image_numpy, sam_pred, boxes)

    display_segmentations(image_numpy, segs)


if __name__ == "__main__":
    main()
