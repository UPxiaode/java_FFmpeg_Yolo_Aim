---


---

<h1 id="java_ffmpeg_yolo_aim">java_FFmpeg_Yolo_Aim</h1>
<p>顾名思义 , 这是由java代码编写的 , 使用 FFmpeg 硬件加速捕获屏幕画面 , YOLO 实时目标检测 , 坐标映射 + 平滑算法驱动底层鼠标自动跟随画面目标的项目;</p>
<h1 id="完整工作链路">完整工作链路</h1>
<ul>
<li><strong>画面采集层</strong>：FFmpeg 捕获显示器画面原始帧 , 相比逐帧截图大幅降低延迟；</li>
<li><strong>视觉推理层</strong>：ONNX Runtime CPU 实时检测画面目标 , 输出目标框坐标、置信度；</li>
<li><strong>坐标计算层</strong>：画面裁剪区域映射、目标中心点偏移计算、平滑插值算法消除鼠标瞬移；</li>
<li><strong>输入控制层</strong>：调用 Windows 原生 API（User32/JNA）底层控制鼠标移动，避开上层库延迟；</li>
</ul>
<h2 id="特点">特点</h2>
<p>嗯~ , 可能就只有 “用java写的” 这一个特点了;  (因为,  为了速度 , 一般的这种项目是用c++或python)</p>
<p>项目采用的是单线程 , 从获取帧信息到yolo推理 , 再到控制鼠标.一条路走到黑!<br>
(延迟会很高 , 如果想要更快的速度可以改为"经典的多线程，生产者与消费者模式")</p>
<p>关于YOLO的推理是用的ONNX Runtime CPU. 注意是CPU不是CUDA. 使用CPU的好处是你可以较方便的运行该项目.免去了CUDA的配置! (当然代价是跟高的延迟)</p>
<p>因为项目的ONNX Runtime 推理用的java包太老了 , 因此将训练好的YOLO模型转".ONNX"时要注意IR版本要&lt;=9!</p>

