import argparse
import cv2
import sys
from ultralytics import YOLO


def train_model(data_yaml, epochs=50, weights="yolov8n.pt"):
    """Trains a YOLOv8 model on a custom dataset."""
    print(
        f"Starting training using {weights} on dataset {data_yaml} for {
            epochs
        } epochs..."
    )
    model = YOLO(weights)

    model.train(data=data_yaml, epochs=epochs, imgsz=640)
    print(
        "Training complete! Your weights are saved in 'runs/detect/train/weights/best.pt'"
    )


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO Face Detection and Training Script"
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Run in training mode")
    mode_group.add_argument(
        "--detect", action="store_true", help="Run in detection (inference) mode"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO weights (.pt file)",
    )

    parser.add_argument(
        "--data", type=str, help="Path to dataset YAML file (required for training)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train (default: 50)"
    )

    parser.add_argument(
        "--video", type=str, help="Path to input video file (required for detection)"
    )

    args = parser.parse_args()

    if args.train:
        if not args.data:
            parser.error(
                "The --train mode requires the --data argument (path to your dataset.yaml)."
            )
        train_model(args.data, args.epochs, args.weights)

    elif args.detect:
        if not args.video:
            parser.error(
                "The --detect mode requires the --video argument (path to your input video)."
            )
        detect_faces(args.video, args.weights)
