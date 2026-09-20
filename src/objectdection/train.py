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
