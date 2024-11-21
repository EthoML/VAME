import cv2


def get_video_frame_rate(video_path):
    video = cv2.VideoCapture(video_path)
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    video.release()
    return frame_rate
