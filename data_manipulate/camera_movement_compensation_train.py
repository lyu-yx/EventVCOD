import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

def estimate_global_motion(prev_frame, next_frame):
    """ Estimate global motion between two frames using optical flow and RANSAC to find a homography matrix. """
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow (Farneback's method)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Prepare data for homography estimation
    h, w = flow.shape[:2]
    points_orig = np.float32([ [x, y] for y in range(h) for x in range(w) ])
    points_flow = points_orig + np.reshape(flow, (h*w, 2))

    # Estimate homography using RANSAC
    H, status = cv2.findHomography(points_orig, points_flow, cv2.RANSAC, 20)  # 5.0 is the RANSAC reprojection threshold
    return H

def compensate_motion(frame, homography_mat):
    # Compensate for camera motion
    h, w = frame.shape[:2]
    compensated_frame = cv2.warpPerspective(frame, homography_mat, (w, h), flags=cv2.INTER_LINEAR)
    # Create a mask for non-padding areas
    mask = cv2.warpPerspective(np.ones_like(frame, dtype=np.uint8) * 255, homography_mat, (w, h), flags=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return compensated_frame, mask

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

    prev_compensated, prev_mask = compensate_motion(prev_frame, np.eye(3))  # Identity matrix for the first frame
    
    for file in files[1:]:
        current_frame = cv2.imread(os.path.join(directory, file))
        if current_frame is None:
            continue

        homography_mat = estimate_global_motion(prev_frame, current_frame)
        if homography_mat is not None:
            current_compensated, current_mask = compensate_motion(current_frame, homography_mat)

            # cv2.imshow('com', current_compensated)
            
            positive_events, negative_events = generate_event_data(prev_compensated, current_compensated, prev_mask, current_mask, threshold=50)


            # Save the event data image
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(os.path.join(save_path, 'positive'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'negative'), exist_ok=True)
            
            cv2.imwrite(os.path.join(save_path, 'positive', file), positive_events)
            cv2.imwrite(os.path.join(save_path, 'negative', file), negative_events)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prev_compensated = current_compensated
            prev_mask = current_mask
            # cv2.imshow('prev_compensated Frame', prev_compensated)
            # cv2.imshow('Compensated Frame', current_compensated)
            # cv2.imshow('Mask', current_mask)
            # cv2.imshow('Masked Event Data', event_data)

        prev_frame = current_frame
    
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    directory = 'D:\Dateset\MoCA-Mask\MoCA_Video\TrainDataset_per_sq'
    moca_list = os.listdir(directory)
    for curr_dir in tqdm(moca_list):
        os.makedirs(os.path.join(directory, curr_dir, 'Eventflow'), exist_ok=True)
        frame_directory = os.path.join(directory, curr_dir, 'Imgs')
        save_path = os.path.join(directory, curr_dir, 'Eventflow')
        process_frames(frame_directory, save_path)

    
    
    # frame_directory = 'D:\Dateset\MoCA-Mask-Pseudo\MoCA-Video-Train\crab_1\Frame'  # Update this with the actual path
    # # frame_directory = 'D:\Dateset\MoCA-Mask\MoCA_Video\TrainDataset_per_sq\crab\Imgs'  # Update this with the actual path
    # save_path = './figs/MoCA-crab_1-psudo-compensate'  # Define your save path
    # process_frames(frame_directory, save_path)
