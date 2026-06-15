package com.example;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
/*注意:
    ai.onnxruntime 是 ONNX Runtime 早期 Java 封装.旧版 ONNX Runtime 仅支持低版本的IR
*/ 
import java.util.List;
import org.bytedeco.opencv.global.opencv_highgui; 
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_core.CV_MAKETYPE;
import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import org.bytedeco.javacv.Frame;


    
    /*
        我们使用双线程实现目标检测与跟抢
        1.New_Window_Draw_Box_Mode() 新建窗口绘框模式
        2.Crosshair_Mode()           准心模式



    */
public class run_mode {


    private static OrtEnvironment ortEnv;
    private static OrtSession ortSession;
    // 新加的：红点的颜色，不用改
    private static final Scalar POINT_COLOR = new Scalar(255, 0, 0, 10);
    
    public static void  New_Window_Draw_Box_Mode()throws Exception {
      // -------------------------- 1. 你原来的加载模型代码，完全不用动！ --------------------------
        System.out.println("正在加载YOLO模型...");
        ortEnv = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        
        // 👉 如果你有NVIDIA显卡，想加速，把下面这行注释打开！
        //options.addCUDA(0);
        
        String modelPath = Config_Constants.modelPath;
        ortSession = ortEnv.createSession(modelPath, options);
        System.out.println("模型加载完成！");

        // 新加的：创建窗口，只做一次
        opencv_highgui.namedWindow("Result", opencv_highgui.WINDOW_NORMAL);

        Mat srcMat = null;
        try {
            // -------------------------- 2. 你原来的抓帧循环，完全不用动！ --------------------------
            while (true) {
                Frame frame = ScreenCapture.getOneFrame();
                if (frame == null) break;


                // -------------------------- 3. 你原来的预处理、推理，完全不用动！ --------------------------
                YoloPreProcess.PreProcessResult preResult = YoloPreProcess.preprocess(frame);
                List<YoloInference.DetectBox> detectResult = YoloInference.runYoloInference(preResult, ortSession);



                // -------------------------- 新加的：画框！ --------------------------
                // 把当前帧转成OpenCV能画的Mat
                srcMat = frameToRawMat(frame);
                // 给每个检测目标画框
                
                drawTargetBox(srcMat, detectResult);
                // 把画面展示出来
                opencv_highgui.imshow("Result", srcMat);

                // 释放Mat
                if (srcMat != null) srcMat.close();


                // -------------------------- 新加的：按q键退出，和原来的逻辑兼容 --------------------------
                /*有点问题,不能做到全局捕获,会失灵
                 */
                int key = opencv_highgui.waitKey(1);
                if (key == 27) { // 按ESC退出
                    System.out.println("收到退出指令");
                    break;
                }
            }
        } finally {
            // 新加的：安全释放所有资源，防止内存泄漏
            if (ortSession != null) ortSession.close();
            opencv_highgui.destroyAllWindows();
            System.out.println("程序安全退出");
        }
    }



    // 新加的：把你的Frame转成原始屏幕的Mat，用来绘制框体
    private static Mat frameToRawMat(Frame frame) {
        org.bytedeco.javacpp.BytePointer bytePtr = new org.bytedeco.javacpp.BytePointer((java.nio.ByteBuffer) frame.image[0]);
        int matType = CV_MAKETYPE(CV_8U, frame.imageChannels);
        Mat mat = new Mat(frame.imageHeight, frame.imageWidth, matType, bytePtr);
        bytePtr.close();
        return mat;
    }

    
        // 新加的：给每个目标绘制检测框，标记识别目标
        private static void drawTargetBox(Mat img, List<YoloInference.DetectBox> boxes) {
            // 无检测结果直接返回，不执行绘制逻辑
            if (boxes.isEmpty()) return;
            // 绘制框的线条宽度
            //int lineWidth = 2;
            // 遍历每一个检测框结果
            for (YoloInference.DetectBox box : boxes) {
                // 取出当前框的四个坐标：左上角(x1,y1)、右下角(x2,y2)
                int x1 = Math.round(box.x1);
                int y1 = Math.round(box.y1);
                int x2 = Math.round(box.x2);
                int y2 = Math.round(box.y2);

                // 构造左上角、右下角坐标点
                Point pt1 = new Point(x1, y1);
                Point pt2 = new Point(x2, y2);

                // OpenCV 绘制矩形框：原图、左上点、右下点、颜色、线条宽度
                rectangle(img, pt1, pt2, POINT_COLOR);

                // 释放原生Point资源，避免内存泄漏
                pt1.close();
                pt2.close();
            }
        }
//------------------------------------------------------------------------------------------------------
        
        public static void Crosshair_Mode() throws Exception {
            ortEnv = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            String modelPath = Config_Constants.modelPath;
            ortSession = ortEnv.createSession(modelPath, options);
            //opencv_highgui.namedWindow("检测结果", opencv_highgui.WINDOW_NORMAL);
            try {
                while (true) {
                    Frame frame = ScreenCapture.getOneFrame();
                    YoloPreProcess.PreProcessResult preResult = YoloPreProcess.preprocess(frame);
                    List<YoloInference.DetectBox> detectResult = YoloInference.runYoloInference(preResult, ortSession);
                    
                    for (YoloInference.DetectBox box : detectResult) {
                    float centerX = (box.x1 + box.x2) / 2f;
                    float centerY = (box.y1 + box.y2) / 2f;
                    int cx = Math.round(centerX);
                    int cy = Math.round(centerY);
                    RealTimeAim.aimTrack(cx, cy);
                    Thread.sleep(8);
                    }
                    // -------------------------- 新加的：按q键退出，和原来的逻辑兼容 --------------------------
                    /*有点问题,不能做到全局捕获,会失灵
                    */
                    int key = opencv_highgui.waitKey(1);
                    if (key == 'q' || key == 27) { // 按q或者ESC退出
                        System.out.println("收到退出指令");
                        break;
                    }
                }
            } finally {
                // 新加的：安全释放所有资源，防止内存泄漏
                if (ortSession != null) ortSession.close();
                opencv_highgui.destroyAllWindows();
                System.out.println("程序安全退出");
            }
    }
    //---------------------------------------------------------------------------------------------

    


//------------------------------------------------------------------------------------------
    public static void main(String[] args) {
        try {
            run_mode.New_Window_Draw_Box_Mode();
        } catch (Exception e) {
            // TODO: handle exception
        }
        
    }
}