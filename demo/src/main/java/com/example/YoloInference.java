package com.example;


// ...existing code...
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
// ...existing code...

// 这个类就是专门做YOLO推理的，对接你之前写的YoloPreProcess类
public class YoloInference {
    // ====================== 你可以自己改的参数 ======================
    // 置信度阈值：低于这个把握的检测直接扔掉，0.5就是只留50%以上把握的结果
    private static final float CONFIDENCE_THRESHOLD = 0.3f;
    // 重复框过滤阈值：两个框重叠超过这个比例，就删掉重复的
    private static final float NMS_THRESHOLD = 0.45f;

    // ====================== COCO数据集的类别标签 ======================
    // 如果你用的是自己训练的模型，把这里换成你自己的类别名字就行！
    private static final String[] COCO_LABELS = {
         "Head", 
    
    };

    // ====================== 检测框结果类：给你最终能用的框信息 ======================
    // 你拿到这个对象，直接就能用里面的坐标、类别、置信度！
    public static class DetectBox {
        public float x1;    // 框的左上角X坐标（原图上的！）
        public float y1;    // 框的左上角Y坐标（原图上的！）
        public float x2;    // 框的右下角X坐标（原图上的！）
        public float y2;    // 框的右下角Y坐标（原图上的！）
        public String label;// 检测到的东西名字，比如"person"、"car"
        public float confidence; // 模型对这个检测的把握度，0~1之间，越高越准

        public DetectBox(float x1, float y1, float x2, float y2, String label, float confidence) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.label = label;
            this.confidence = confidence;
        }
    }

    // ====================== 核心推理方法！你直接调用这个就行！ ======================
    /**
     * YOLO模型推理的核心方法，对接你之前写的预处理结果
     * @param preResult 你之前YoloPreProcess.preprocess()返回的结果
     * @param ortSession 你提前初始化好的模型会话（整个程序只初始化一次！）
     * @return 检测到的所有框的列表，直接用就行
     */
    public static List<DetectBox> runYoloInference(
            YoloPreProcess.PreProcessResult preResult,
            OrtSession ortSession
    ) {
        // 先拿到ONNX的运行环境（全局的，不用管）
        OrtEnvironment ortEnv = OrtEnvironment.getEnvironment();

        try {
            // -------------------------- 1. 把预处理好的数据包装成模型能认的张量 --------------------------
            // 张量你可以理解成：把一堆数字按模型要求的顺序排好，打包给模型
            long[] inputShape = {1, 3, YoloPreProcess.MODEL_SIZE, YoloPreProcess.MODEL_SIZE};
            // 把你预处理好的输入数组，包装成ONNX能识别的张量对象
            OnnxTensor inputTensor = OnnxTensor.createTensor(
                    ortEnv,
                    FloatBuffer.wrap(preResult.inputData),
                    inputShape
            );

            // 告诉模型：输入的名字是"images"，数据是我们刚打包好的张量
            // （99%的YOLO模型输入节点都叫这个名字，不用改）
            Map<String, OnnxTensor> inputMap = new HashMap<>();
            inputMap.put("images", inputTensor);

            // -------------------------- 2. 执行推理！模型真正开始算的地方 --------------------------
            // 这一步就是把数据丢给模型，等它算完给我们结果
            // try-with-resources语法：用完自动帮你释放内存，不用你手动关，小白不用管
            try (
                OnnxTensor t = inputTensor;
                OrtSession.Result result = ortSession.run(inputMap)
            ) {
                // -------------------------- 3. 解析模型输出的原始数据 --------------------------
                // 模型输出的原始数据是一大串数字，我们要把它拆成一个个检测框
                OnnxTensor outputTensor = (OnnxTensor) result.get(0);
                float[][][] outputData = (float[][][]) outputTensor.getValue();

                // YOLOv8的输出格式是：[1, 84, 8400]，意思是：
                // 1张图，84个特征，8400个候选框（模型会扫8400个位置找东西）
                List<DetectBox> boxes = new ArrayList<>();
                int numAnchors = outputData[0][0].length; // 8400个候选框

                // 遍历所有候选框，挑出靠谱的
                for (int i = 0; i < numAnchors; i++) {
                    // 取出这个框的中心坐标、宽高
                    float x = outputData[0][0][i]; // 框的中心X
                    float y = outputData[0][1][i]; // 框的中心Y
                    float w = outputData[0][2][i]; // 框的宽度
                    float h = outputData[0][3][i]; // 框的高度

                    // 找出这个框最可能是什么东西，以及对应的置信度
                    int classId = -1;
                    float maxConf = 0;
                    // 遍历80个类别的得分，找最高的那个
                    for (int j = 0; j < COCO_LABELS.length; j++) {
                        float conf = outputData[0][j + 4][i];
                        if (conf > maxConf) {
                            maxConf = conf;
                            classId = j;
                        }
                    }

                    // 过滤掉置信度太低的框，比如模型自己都没把握的，我们直接扔掉
                    if (maxConf < CONFIDENCE_THRESHOLD) continue;

                    // 把中心坐标+宽高，转成我们常用的「左上角、右下角」坐标（方便画框）
                    float x1 = x - w / 2;
                    float y1 = y - h / 2;
                    float x2 = x + w / 2;
                    float y2 = y + h / 2;

                    // 把这个候选框先存起来
                    boxes.add(new DetectBox(x1, y1, x2, y2, COCO_LABELS[classId], maxConf));
                }

                // -------------------------- 4. NMS去重：删掉重叠的重复框 --------------------------
                // 比如同一个人，模型可能检测出好几个重叠的框，我们只留最准的那个
                List<DetectBox> finalBoxes = nonMaxSuppression(boxes, NMS_THRESHOLD);

                // -------------------------- 5. 坐标还原：把1280图上的坐标，转成你屏幕原图的坐标！ --------------------------
                // 这一步就是用你预处理返回的缩放、补边参数，把坐标映射回你真实的屏幕尺寸
                for (DetectBox box : finalBoxes) {
                    box.x1 = (box.x1 - preResult.padLeft) / preResult.scaleRatio;
                    box.y1 = (box.y1 - preResult.padTop) / preResult.scaleRatio;
                    box.x2 = (box.x2 - preResult.padLeft) / preResult.scaleRatio;
                    box.y2 = (box.y2 - preResult.padTop) / preResult.scaleRatio;
                }

                // 最终给你所有检测到的框！直接用就行！
                return finalBoxes;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    // ====================== 内部用的NMS去重方法 ======================
    private static List<DetectBox> nonMaxSuppression(List<DetectBox> boxes, float iouThreshold) {
        if (boxes.isEmpty()) return new ArrayList<>();

        // 按置信度从高到低排序
        boxes.sort((a, b) -> Float.compare(b.confidence, a.confidence));
        List<DetectBox> result = new ArrayList<>();

        boolean[] suppressed = new boolean[boxes.size()];
        for (int i = 0; i < boxes.size(); i++) {
            if (suppressed[i]) continue;
            DetectBox boxI = boxes.get(i);
            result.add(boxI);

            for (int j = i + 1; j < boxes.size(); j++) {
                if (suppressed[j]) continue;
                DetectBox boxJ = boxes.get(j);
                float iou = calculateIoU(boxI, boxJ);
                if (iou > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        return result;
    }

    // 计算两个框的重叠比例
    private static float calculateIoU(DetectBox a, DetectBox b) {
        float interX1 = Math.max(a.x1, b.x1);
        float interY1 = Math.max(a.y1, b.y1);
        float interX2 = Math.min(a.x2, b.x2);
        float interY2 = Math.min(a.y2, b.y2);

        float interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
        float aArea = (a.x2 - a.x1) * (a.y2 - a.y1);
        float bArea = (b.x2 - b.x1) * (b.y2 - b.y1);
        return interArea / (aArea + bArea - interArea);
    }
}