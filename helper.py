import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

KITTI_COLORMAP = {
    0: (0, 0, 255),        
    1: (128, 64, 128),    
    2: (244, 35, 232),    
    3: (70, 70, 70),      
    4: (102, 102, 156),   
    5: (190, 153, 153),   
    6: (153, 153, 153),   
    7: (250, 170, 30), 
    8: (220, 220, 0),
    9: (107, 142, 35),
    10: (152, 251, 152),
    11: (70, 130, 180),
    12: (220, 20, 60),
    13: (255, 0, 0),
    14: (0, 0, 142),
    15: (0, 0, 70),
    16: (0, 60, 100),
    17: (0, 80, 100),
    18: (0, 0, 230),
    19: (119, 11, 32)
}

def separate_stereo_images(source_folder, left_folder_name="left_cam", right_folder_name="right_cam"):
    """
    Separates stereo images into 'left' and 'right' folders based on filename endings

    Args:
        source_folder (str): Path to the folder containing stereo images
        left_folder_name (str): destination to store left cam images (default: 'left_cam')
        right_folder_name (str): destination to store right cam images (default: 'right_cam')
    """
    left_folder = os.path.join(source_folder, left_folder_name)
    right_folder = os.path.join(source_folder, right_folder_name)

    os.makedirs(left_folder, exist_ok=True)
    os.makedirs(right_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if os.path.isfile(file_path):
            if filename.endswith("L.jpg"):
                shutil.move(file_path, os.path.join(left_folder, filename))
            elif filename.endswith("R.jpg"):
                shutil.move(file_path, os.path.join(right_folder, filename))

    # print(f"Left images moved to: {left_folder}")
    # print(f"Right images moved to: {right_folder}")

def extract_frames_from_video(input_video_path, output_folder='frames'):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    #print(f"Extracted {frame_count} frames and saved in '{output_folder}'")

def convert_grayscale_segmentation_masks_to_rgb(input_mask_path, out_path="rgb_masks"):
    """Takes in grayscale masks (pixel values based on classes) and converts into RGB masks

    Args:
        input_mask_path: either one file or folder of images (if single image --> plot visualized)
    """

    if not os.path.exists(input_path): #sanity check
        raise FileNotFoundError(f"Error: '{input_path}' does not exist.")

    os.makedirs(output_folder, exist_ok=True)

    if os.path.isfile(input_path): 
        paths, show = [input_path], True #single image --> visualization set to true
    else:
        paths = [
            os.path.join(input_path, f) for f in os.listdir(input_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        show = False

    for p in paths:
        mask = cv2.imread(p, cv2.IMREAD_UNCHANGED)

        if mask is None: continue

        #color mapping
        rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for label, color in KITTI_COLORMAP.items():
            rgb[mask == label] = color

        out_file = os.path.join(out_path, f"rgb_{os.path.basename(p)}")
        cv2.imwrite(out_file, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if show:
            plt.imshow(rgb)
            plt.axis('off')
            plt.title("RGB Mask")
            plt.show()



def main():
    #=======
    # separate stereo images
    #========
    source_folder = "mastr_dataset/framesRectified"
    separate_stereo_images(source_folder)

    #=======
    # frames from video
    #========
    input_video_path = 'videoplayback.mp4'
    extract_frames_from_video(input_video_path)

    #=========================
    # grayscale masks --> RGB
    #=========================
    convert_grayscale_segmentation_masks_to_rgb('image.png')

if __name__ == "__main__":
    main()