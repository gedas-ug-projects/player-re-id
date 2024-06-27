from roboflow import Roboflow
rf = Roboflow(api_key="XCN9TB9TRsGQ2YsvvcE6")
project = rf.workspace("volleyai-actions").project("jersey-number-detection-s01j4")
version = project.version(2)
dataset = version.download("yolov8")
