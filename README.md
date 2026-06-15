这是一个有Java编写的使用 FFmpeg 硬件加速捕获屏幕画面，YOLO 实时目标检测，坐标映射 + 平滑算法驱动底层鼠标自动跟随画面目标的项目;
核心技术栈
视频捕获：FFmpeg 硬件加速屏幕录制、管道帧输出
目标检测：YOLOv8/v10 / ONNX Runtime CUDA 推理
图像处理：OpenCV、帧格式转换、ROI 区域裁剪
系统交互：Windows User32 API、JNA /pywin32 底层鼠标模拟
