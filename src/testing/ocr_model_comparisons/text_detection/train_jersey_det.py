from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.yaml")
model = YOLO("yolov8m.pt")

# Use the model
data_path = '/playpen-storage/levlevi/player-re-id/src/testing/ocr_model_comparisons/text_detection/Jersey-Number-detection-2/data.yaml'
model.train(data=data_path, epochs=10)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
path = model.export(format="onnx")  # export the model to ONNX format