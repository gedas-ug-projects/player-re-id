from ultralytics import YOLO

# load a model
model = YOLO("yolov8x.yaml")
model = YOLO("yolov8x.pt")

# use the model
data_path = '/playpen-storage/levlevi/player-re-id/src/testing/ocr_model_comparisons/text_detection/Jersey-Number-detection-2/data.yaml'
model.train(data=data_path, epochs=100)  # train the model
path = model.export(format="onnx")  # export the model to ONNX format
metrics = model.val()  # evaluate model performance on the validation set