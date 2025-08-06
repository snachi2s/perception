from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import torch
from PIL import Image
import cv2
import numpy as np
import re

#model inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "google/paligemma2-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
processor = PaliGemmaProcessor.from_pretrained(model_id)
model.to(device)

input_image = "yolo/dog-2.jpeg" 
input_img = Image.open(input_image)
output_img = np.array(input_img)

prompt = "<image> detect dog; person"
inputs = processor(text=prompt, images=input_img, padding="longest", do_convert_rgb=True, return_tensors="pt").to("cuda")
inputs = inputs.to(dtype=model.dtype)

with torch.no_grad():
    output = model.generate(**inputs, max_length=496)

paligemma_response = processor.decode(output[0], skip_special_tokens=True)[len(prompt):].lstrip("\n")
print("PaliGemma Response:", repr(paligemma_response))

# parse detections - SIMPLIFIED
detections = paligemma_response.split(" ; ")
print("Detections:", detections)

for detection in detections:
    #print(f"Processing: {detection}")
    
    # handle malformed detection (missing <loc at start)
    if detection and detection[0].isdigit():
        first_num = detection.split('>')[0]
        rest = detection[len(first_num)+1:]
        detection = f"<loc{first_num.zfill(4)}>" + rest
        #print(f"Fixed to: {detection}")
    
    #extract coordinates using simple regex
    coords = re.findall(r'<loc(\d+)>', detection)
    
    if len(coords) == 4:
        # Get label (everything after the last >)
        label = detection.split('>')[-1].strip()
        
        # Convert PaliGemma coords (0-1024) to pixel coordinates
        y1, x1, y2, x2 = [int(c) for c in coords]
        
        # normalizing from paligemma coordinates (0-1024) to pixel coordinates to image coordinates
        height, width = output_img.shape[:2]
        x1_px = int(x1 * width / 1024) #1024 --> max token size of paligemma
        y1_px = int(y1 * height / 1024)  
        x2_px = int(x2 * width / 1024)
        y2_px = int(y2 * height / 1024)
        
        print(f"Drawing {label}: ({x1_px},{y1_px}) to ({x2_px},{y2_px})")
        
        #box with label
        cv2.rectangle(output_img, (x1_px, y1_px), (x2_px, y2_px), (0, 0, 255), 5)        
        cv2.rectangle(output_img, (x1_px, y1_px-30), (x1_px+100, y1_px), (0, 0, 255), -1)
        cv2.putText(output_img, label, (x1_px+5, y1_px-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

output_img = Image.fromarray(output_img)
output_img.show()
