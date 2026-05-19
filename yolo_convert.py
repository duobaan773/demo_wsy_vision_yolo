from ultralytics import YOLO
#加载训练好的模型
model = YOLO(r"D:\deeplearning\runs\detect\train3\weights\best.pt")
#导出模型为ONNX格式
model.export(format="onnx")