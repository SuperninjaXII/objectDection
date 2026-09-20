import argparse
from detect import detect_faces
from tarin import train_model


def main():
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


if __name__ == "__main__":
    main()
