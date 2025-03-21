import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import logging
from pathlib import Path
import concurrent.futures
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# def detect_and_compensate_camera_motion(prev_frame, curr_frame, max_features=1000):
#     """
#     Detect and compensate for camera motion using feature matching.
    
#     Args:
#         prev_frame: Previous frame
#         curr_frame: Current frame
#         max_features: Maximum number of features to detect
    
#     Returns:
#         compensated_frame: Motion-compensated current frame
#         mask: Valid regions after compensation
#         is_moving: Boolean indicating if significant camera motion was detected
#     """
#     # Convert frames to grayscale
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#     curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
#     # Detect features
#     orb = cv2.ORB_create(max_features)
#     kp1, des1 = orb.detectAndCompute(prev_gray, None)
#     kp2, des2 = orb.detectAndCompute(curr_gray, None)
    
#     if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
#         return curr_frame, np.ones_like(curr_frame), False
    
#     # Match features
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
    
#     # Sort matches by distance
#     matches = sorted(matches, key=lambda x: x.distance)
    
#     # Take only good matches
#     good_matches = matches[:min(50, len(matches))]
    
#     # Extract matched keypoints
#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
#     # Estimate rigid transformation (translation + rotation)
#     transform_matrix, inliers = cv2.estimateAffinePartial2D(
#         src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
#     )
    
#     if transform_matrix is None:
#         return curr_frame, np.ones_like(curr_frame), False
    
#     # Convert to full 3x3 homography matrix
#     transform_matrix_h = np.vstack([transform_matrix, [0, 0, 1]])
    
#     # Calculate movement magnitude
#     translation = np.linalg.norm(transform_matrix[:, 2])
#     rotation = np.arccos(transform_matrix[0, 0]) * 180 / np.pi
#     is_moving = translation > 5 or abs(rotation) > 2
    
#     # Apply compensation
#     h, w = curr_frame.shape[:2]
#     compensated_frame = cv2.warpAffine(
#         curr_frame, 
#         transform_matrix,
#         (w, h),
#         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
#     )
    
#     # Create validity mask
#     mask = cv2.warpAffine(
#         np.ones_like(curr_frame),
#         transform_matrix,
#         (w, h),
#         flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP
#     )
    
#     return compensated_frame, mask, is_moving

def detect_and_compensate_camera_motion(prev_frame, curr_frame, max_features=1000):
    """
    Detect and compensate for camera motion using feature matching with homography.
    
    Args:
        prev_frame: Previous frame
        curr_frame: Current frame
        max_features: Maximum number of features to detect
    
    Returns:
        compensated_frame: Motion-compensated current frame
        mask: Valid regions after compensation
        is_moving: Boolean indicating if significant camera motion was detected
    """
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect features
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return curr_frame, np.ones_like(curr_frame), False
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Take only good matches
    good_matches = matches[:min(50, len(matches))]
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Estimate homography transformation
    homography_matrix, inliers = cv2.findHomography(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    
    if homography_matrix is None:
        return curr_frame, np.ones_like(curr_frame), False
    
    # Calculate movement magnitude
    translation = np.linalg.norm(homography_matrix[:2, 2])
    rotation = np.arccos(homography_matrix[0, 0]) * 180 / np.pi
    is_moving = translation > 5 or abs(rotation) > 2
    
    # Apply compensation using homography
    h, w = curr_frame.shape[:2]
    compensated_frame = cv2.warpPerspective(
        curr_frame, 
        homography_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )
    
    # Create validity mask
    mask = cv2.warpPerspective(
        np.ones_like(curr_frame),
        homography_matrix,
        (w, h),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP
    )
    
    return compensated_frame, mask, is_moving

def generate_clean_event_data(prev_frame, curr_frame, threshold=15):
    """
    Generate event data with additional noise reduction.
    """
    # Compensate for camera motion
    compensated_curr, mask, is_moving = detect_and_compensate_camera_motion(prev_frame, curr_frame)
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(compensated_curr, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    prev_filtered = cv2.bilateralFilter(prev_gray, 5, 75, 75)
    curr_filtered = cv2.bilateralFilter(curr_gray, 5, 75, 75)
    
    # Generate events
    increase = cv2.subtract(curr_filtered, prev_filtered)
    decrease = cv2.subtract(prev_filtered, curr_filtered)
    
    # Apply adaptive thresholding
    increase_thresh = cv2.adaptiveThreshold(
        increase, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -threshold
    )
    decrease_thresh = cv2.adaptiveThreshold(
        decrease, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -threshold
    )
    
    # Apply mask
    positive_events = cv2.bitwise_and(increase_thresh, increase_thresh, mask=mask[:,:,0])
    negative_events = cv2.bitwise_and(decrease_thresh, decrease_thresh, mask=mask[:,:,0])
    
    # Remove small noise blobs
    kernel = np.ones((3,3), np.uint8)
    positive_events = cv2.morphologyEx(positive_events, cv2.MORPH_OPEN, kernel)
    negative_events = cv2.morphologyEx(negative_events, cv2.MORPH_OPEN, kernel)
    
    return positive_events, negative_events, is_moving

def setup_directories(base_dir: str) -> Tuple[Path, Path, Path]:
    """
    Create necessary directories for output files.
    """
    base_path = Path(base_dir)
    positive_path = base_path / 'positive'
    negative_path = base_path / 'negative'
    debug_path = base_path / 'debug'
    
    for path in [positive_path, negative_path, debug_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    return positive_path, negative_path, debug_path

def process_single_sequence(
    sequence_path: str,
    output_base: str,
    debug: bool = False
) -> Optional[Tuple[int, int]]:
    """
    Process a single sequence of frames and generate event data.
    
    Args:
        sequence_path: Path to the input sequence directory
        output_base: Base path for output files
        debug: Whether to save debug visualizations
    
    Returns:
        Tuple of (processed_frames, error_frames) or None if failed
    """
    try:
        # Setup output directories
        pos_path, neg_path, debug_path = setup_directories(output_base)
        
        # Get sorted list of frame files
        frame_files = sorted([f for f in os.listdir(sequence_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not frame_files:
            logger.warning(f"No valid image files found in {sequence_path}")
            return None
        
        # Initialize counters
        processed_frames = 0
        error_frames = 0
        
        # Get the first frame
        prev_frame = cv2.imread(os.path.join(sequence_path, frame_files[0]))
        if prev_frame is None:
            logger.error(f"Could not read first frame: {frame_files[0]}")
            return None
            
        # Process frame pairs
        for frame_file in tqdm(frame_files[1:], desc=f"Processing {Path(sequence_path).name}"):
            curr_frame = cv2.imread(os.path.join(sequence_path, frame_file))
            if curr_frame is None:
                logger.warning(f"Could not read frame: {frame_file}")
                error_frames += 1
                continue
                
            try:
                # Generate event data with camera motion compensation
                positive_events, negative_events, is_moving = generate_clean_event_data(
                    prev_frame, curr_frame, threshold=15
                )
                
                # Save event data
                cv2.imwrite(str(pos_path / frame_file), positive_events)
                cv2.imwrite(str(neg_path / frame_file), negative_events)
                
                # Save debug visualizations if requested
                if debug and is_moving:
                    debug_vis = np.hstack([
                        cv2.cvtColor(positive_events, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(negative_events, cv2.COLOR_GRAY2BGR)
                    ])
                    cv2.imwrite(str(debug_path / f"debug_{frame_file}"), debug_vis)
                
                processed_frames += 1
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_file}: {str(e)}")
                error_frames += 1
                
            prev_frame = curr_frame
            
        return processed_frames, error_frames
        
    except Exception as e:
        logger.error(f"Error processing sequence {sequence_path}: {str(e)}")
        return None

def process_dataset(
    root_dir: str,
    output_dir: str,
    num_workers: int = 4,
    debug: bool = False
) -> None:
    """
    Process entire dataset with multiple sequences.
    
    Args:
        root_dir: Root directory containing sequence directories
        output_dir: Base directory for output
        num_workers: Number of parallel workers
        debug: Whether to save debug visualizations
    """
    try:
        # Get all sequence directories
        sequence_dirs = [d for d in Path(root_dir).iterdir() if d.is_dir()]
        
        if not sequence_dirs:
            logger.error(f"No valid sequence directories found in {root_dir}")
            return
            
        total_processed = 0
        total_errors = 0
        
        # Process sequences in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_seq = {
                executor.submit(
                    process_single_sequence,
                    str(seq_dir) + '/Frame',
                    str(Path(output_dir) / seq_dir.name),
                    debug
                ): seq_dir.name
                for seq_dir in sequence_dirs
            }
            
            for future in tqdm(
                concurrent.futures.as_completed(future_to_seq),
                total=len(future_to_seq),
                desc="Processing sequences"
            ):
                seq_name = future_to_seq[future]
                try:
                    result = future.result()
                    if result is not None:
                        processed, errors = result
                        total_processed += processed
                        total_errors += errors
                        logger.info(f"Completed sequence {seq_name}: {processed} frames processed, {errors} errors")
                except Exception as e:
                    logger.error(f"Sequence {seq_name} failed: {str(e)}")
                    
        # Log final statistics
        logger.info(f"""
        Processing completed:
        - Total frames processed: {total_processed}
        - Total errors: {total_errors}
        - Success rate: {(total_processed / (total_processed + total_errors) * 100):.2f}%
        """)
        
    except Exception as e:
        logger.error(f"Dataset processing failed: {str(e)}")


def create_redblue_visualization(positive_event: np.ndarray, negative_event: np.ndarray) -> np.ndarray:
    """
    Create a red-blue visualization from positive and negative events.
    
    Args:
        positive_event: Binary image of positive events
        negative_event: Binary image of negative events
    
    Returns:
        RGB image with red for positive events and blue for negative events
    """
    # Create empty RGB image
    h, w = positive_event.shape
    vis_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Set red channel (positive events)
    vis_image[positive_event > 1, 2] = 255  # Red channel
    
    # Set blue channel (negative events)
    vis_image[negative_event > 1, 0] = 255  # Blue channel
    
    return vis_image

def save_images_with_padding(sequence_path: str, output_base: str, file_extension: str = '.png') -> None:
    """
    Process and save images with renaming (starting from 00000, 00005, 00010...) and padding with a blank frame.

    Args:
        sequence_path: Path to input sequence directory
        output_base: Base path for output files
        file_extension: File extension for saved images
    """
    try:
        # Setup output directories
        output_base = Path(output_base)
        pos_path = output_base / 'positive'
        neg_path = output_base / 'negative'
        redblue_path = output_base / 'Eventflow_new'
        
        # Create directories if they don't exist
        redblue_path.mkdir(parents=True, exist_ok=True)
        
        # Get sorted list of frame files (assuming they match in positive and negative directories)
        pos_files = sorted(list(pos_path.glob(f'*{file_extension}')))
        neg_files = sorted(list(neg_path.glob(f'*{file_extension}')))
        
        if len(pos_files) != len(neg_files):
            raise ValueError("Number of positive and negative event files don't match")
        
        # Process each pair of files
        for pos_file, neg_file in tqdm(zip(pos_files, neg_files), total=len(pos_files)):
            # Read event images
            pos_event = cv2.imread(str(pos_file), cv2.IMREAD_GRAYSCALE)
            neg_event = cv2.imread(str(neg_file), cv2.IMREAD_GRAYSCALE)
            
            if pos_event is None or neg_event is None:
                logging.warning(f"Could not read files: {pos_file} or {neg_file}")
                continue
            
            # Create visualization
            redblue_img = create_redblue_visualization(pos_event, neg_event)
            
            # Use the original name from the positive folder, but shift naming to start from 00000
            original_name = pos_file.name
            new_name = f"{int(original_name.split('.')[0]) - int(pos_files[0].stem) + 1:05d}{file_extension}"
            output_path = redblue_path / new_name
            cv2.imwrite(str(output_path), redblue_img)
        
        # Add blank frame using the last filename from positive directory
        if pos_files:
            last_filename = pos_files[-1].name
            blank_frame = np.zeros((pos_event.shape[0], pos_event.shape[1], 3), dtype=np.uint8)
            cv2.imwrite(str(redblue_path / last_filename), blank_frame)
        
        logging.info(f"Saved {len(pos_files) + 1} frames (including blank frame) to {redblue_path}")
        
    except Exception as e:
        logging.error(f"Error in save_images_with_padding: {str(e)}")

def process_dataset_with_visualization(
    root_dir: str,
    output_dir: str,
    num_workers: int = 4,
    debug: bool = False
) -> None:
    """
    Enhanced dataset processing function that includes red-blue visualization.
    """
    try:
        sequence_dirs = [d for d in Path(root_dir).iterdir() if d.is_dir()]
        
        for seq_dir in tqdm(sequence_dirs, desc="Processing sequences"):
            # First process the sequence to generate positive and negative events
            output_base = Path(output_dir) / seq_dir.name
            result = process_single_sequence(str(seq_dir), str(output_base), debug)
            
            if result is not None:
                # Then create the red-blue visualizations
                save_images_with_padding(str(seq_dir), str(output_base))
                
        logging.info("Dataset processing completed successfully")
        
    except Exception as e:
        logging.error(f"Dataset processing failed: {str(e)}")


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "D:\Dateset\CAD2016_Processed"  # Update this
    OUTPUT_DIR = "D:\Dateset\CAD2016_Processed"       # Update this
    NUM_WORKERS = 12                    # Adjust based on your CPU
    DEBUG_MODE = True                   # Set to True to save debug visualizations
    
    # Process the dataset
    process_dataset(
        root_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        num_workers=NUM_WORKERS,
        debug=DEBUG_MODE
    )
    
    
    sequence_dirs = [d for d in Path(OUTPUT_DIR).iterdir() if d.is_dir()]
    for seq_dir in sequence_dirs:
        save_images_with_padding(
        sequence_path=INPUT_DIR / seq_dir,
        output_base=  seq_dir,
        file_extension = '.png'
        )
        
    # Or use the complete dataset processing
    # process_dataset_with_visualization(
    #     root_dir="datasets\MoCA-Video-Train_event",
    #     output_dir="datasets\MoCA-Video-Train_event"
    # )