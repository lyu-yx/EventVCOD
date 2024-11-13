import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

def generate_event_data(prev_frame, current_frame, prev_mask, current_mask, threshold=15):
    """ Generate event data based on increases and decreases in pixel intensity, excluding padding areas. """
    increase = cv2.subtract(current_frame, prev_frame)
    decrease = cv2.subtract(prev_frame, current_frame)
    
    _, increase_thresh = cv2.threshold(increase, threshold, 255, cv2.THRESH_BINARY)
    _, decrease_thresh = cv2.threshold(decrease, threshold, 255, cv2.THRESH_BINARY)
    
    mask = cv2.bitwise_and(prev_mask, current_mask)
    positive_events = cv2.bitwise_and(increase_thresh, increase_thresh, mask=mask[:,:,0])
    negative_events = cv2.bitwise_and(decrease_thresh, decrease_thresh, mask=mask[:,:,0])
    
    return positive_events, negative_events

def process_frames(directory, save_path):
    files = sorted(os.listdir(directory))
    prev_frame = cv2.imread(os.path.join(directory, files[0]))

    if prev_frame is None:
        print("Error loading the first frame.")
        return

    prev_mask = np.ones_like(prev_frame, dtype=np.uint8) * 255  # Set initial mask to full (no padding)

    for file in files[1:]:
        current_frame = cv2.imread(os.path.join(directory, file))
        if current_frame is None:
            continue

        # Directly compute the event data from pixel differences
        positive_events, negative_events = generate_event_data(prev_frame, current_frame, prev_mask, prev_mask, threshold=50)

        # Save the event data images
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, 'positive_new'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'negative_new'), exist_ok=True)
        
        cv2.imwrite(os.path.join(save_path, 'positive_new', file), positive_events)
        cv2.imwrite(os.path.join(save_path, 'negative_new', file), negative_events)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = current_frame  # Update previous frame
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    directory = 'D:/Dateset/MoCA-Mask-Pseudo/MoCA-Video-Train'  # Adjust to your path
    moca_list = os.listdir(directory)
    
    for curr_dir in tqdm(moca_list):
        os.makedirs(os.path.join(directory, curr_dir, 'Eventflow'), exist_ok=True)
        frame_directory = os.path.join(directory, curr_dir, 'Frame')
        save_path = os.path.join(directory, curr_dir, 'Eventflow')
        process_frames(frame_directory, save_path)
