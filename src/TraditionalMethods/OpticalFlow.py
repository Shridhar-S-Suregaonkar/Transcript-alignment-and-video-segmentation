import cv2
import numpy as np

def optical_flow_segmentation(video_path, threshold=5.0, resize_dim=(320, 240), stride=4):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    scene_boundaries = []

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read the video.")
        return []
    prev_gray = cv2.cvtColor(cv2.resize(prev_frame, resize_dim), cv2.COLOR_BGR2GRAY)

    while True:
        for _ in range(stride - 1):
            cap.read()
            frame_num += 1

        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(cv2.resize(frame, resize_dim), cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion_magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_motion = np.mean(motion_magnitude)
        
        if mean_motion > threshold:
            time_in_seconds = frame_num / fps
            scene_boundaries.append(time_in_seconds)
            print(f"Scene boundary detected at time {time_in_seconds:.2f} seconds")
        
        prev_gray = gray
        frame_num += 1

    cap.release()
    return scene_boundaries
