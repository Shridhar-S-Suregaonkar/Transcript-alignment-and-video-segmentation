import cv2
from datetime import timedelta

def detect_scene_timestamps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_hist = None
    scene_timestamps = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if diff < 0.4:
                seconds = frame_count / fps
                timestamp = str(timedelta(seconds=seconds))
                scene_timestamps.append(timestamp)

        prev_hist = hist
        frame_count += 1

    cap.release()
    return scene_timestamps