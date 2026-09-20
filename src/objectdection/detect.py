import cv2
import sys

from ultralytics import YOLO


def detect_faces(video_path, weights="yolov8n.pt"):
    """Runs YOLOv8 inference on a video stream."""
    print(f"Loading model {weights} for inference on {video_path}...")
    model = YOLO(weights)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video stream ended or cannot be read.")
            break

        results = model(frame, verbose=False)

        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Face Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
