from ultralytics import YOLO

# 加载预训练的YOLO11n模型（轻量版，适合新手）
model = YOLO(r"D:\deeplearning\ultralytics-8.3.163\yolo11n.pt")

# 开始训练，修改data路径为你的yaml文件路径
results = model.train(
    # --- 基础参数 ---
    data="D:/deeplearning/dataset/data.yaml",
    epochs=200,
    imgsz=1024,        # 💡 核心提分点：提高输入分辨率，保护小目标（如果显存不够报错，再改回800或640）
    device=0,
    workers=0,
    batch=8,           # 💡 因为分辨率提高了，batch减半以防显存溢出（如果显卡好，可以保持16）
    patience=50,
    cache=True,

    # --- 数据增强（对小目标极其关键） ---
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,        # 装甲板允许轻微倾斜，10度合理
    translate=0.1,
    scale=0.1,         # 💡 修改：降低缩放幅度，防止小装甲板被缩得太小而消失
    fliplr=0.5,        # 左右翻转没问题，装甲板是对称的
    mosaic=True,
    close_mosaic=15,   # 💡 新增：在最后15个epoch关闭马赛克增强，让模型在真实图片比例下微调

    # --- 优化器与学习率 ---
    optimizer='AdamW', # 💡 新增：明确使用 AdamW 优化器，对小目标和小数据集更友好
    lr0=0.001,         # 💡 新增：设定合适的初始学习率
    lrf=0.01,
    weight_decay=0.0005,
)
