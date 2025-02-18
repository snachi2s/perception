import os
import shutil
import cv2

def separate_stereo_images(source_folder, left_folder_name="left_cam", right_folder_name="right_cam"):
    """
    Separates stereo images into 'left' and 'right' folders based on filename endings

    Args:
        source_folder (str): Path to the folder containing stereo images
        left_folder_name (str): Name for the left image folder (default: 'left_cam')
        right_folder_name (str): Name for the right image folder (default: 'right_cam')
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

if __name__ == "__main__":
    main()