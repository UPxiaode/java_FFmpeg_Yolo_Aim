package com.example;
import org.bytedeco.javacv.Frame;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import org.bytedeco.javacpp.BytePointer;

/**
 * YOLO模型专用预处理工具类
 * 作用：把FFmpeg抓取的原始Frame帧，加工成YOLO ONNX模型能直接用的输入张量
 * 同时配套处理坐标还原需要的参数，方便后处理把模型输出的框还原回原图位置
 */
public class YoloPreProcess {
    // YOLO模型要求的固定输入尺寸
    // 模型训练时全程使用1280x1280的正方形输入，输入必须严格匹配，不然识别会出错
    public static final int MODEL_SIZE = Config_Constants.MODEL_SIZE;

    // 预处理结果包装类：把模型输入张量 + 后处理需要的参数打包在一起
    public static class PreProcessResult {
        public final float[] inputData;    // 模型输入张量：一维float数组，NCHW格式
        public final float scaleRatio;     // 原图到模型的缩放比例，后处理还原坐标用
        public final int padTop;           // 顶部补边像素数，后处理还原坐标用
        public final int padLeft;          // 左侧补边像素数，后处理还原坐标用
        // 构造方法：把传入的参数赋值给成员变量，完成打包
        public PreProcessResult(float[] inputData, float scaleRatio, int padTop, int padLeft) {
            this.inputData = inputData;
            this.scaleRatio = scaleRatio;
            this.padTop = padTop;
            this.padLeft = padLeft;
        }
    }
    

    /**
     * 核心预处理主方法
     * @param frame FFmpeg抓取的原始帧：可以是屏幕截图、摄像头画面、视频帧
     * @return 预处理完的结果，包含输入张量和坐标还原参数
     */
    public static PreProcessResult preprocess(Frame frame) {
        // 提前声明所有中间处理容器，最后统一释放，避免内存泄漏
        Mat srcMat = null;
        Mat paddedMat = null;
        Mat floatMat = null;
        BytePointer bytePtr = null;
        Scalar padColor = null;
        try {
            // 第一步：校验输入通道数
            /*
            为什么要校验通道数？
                我们这套预处理逻辑，只处理3通道的彩色图像，灰度图、带透明通道的图都适配不了。
                如果通道数不对，直接抛出异常提醒，避免后面处理出错。
             */
            // 这行：判断frame的通道数是不是3，不是就抛异常
            if (frame.imageChannels != 3) {
                throw new RuntimeException("仅支持3通道图像，当前通道数：" + frame.imageChannels);
            }

            // 1. Frame 转 OpenCV Mat
            /*前提:我们拿到的帧信息是Frame的对象,这里的frame是对象名字与类名相似,不要被误导了.
                    Frame 是一帧的对象，对象里的各种变量来描述画面的信息:
                    image 数组：像素颜色信息
                    imageWidth、imageHeight、imageChannels：画面尺寸、通道数；
                    时间戳、帧序号、帧类型等附属信息；它的定位只是存储原始画面数据 + 画面说明参数，
                    在Frame类中有数组成员变量public ByteBuffer[] image;可以直接用"对象."的方式获取.
            frame.image[0]等价ByteBuffer[0],那么ByteBuffer[0]是什么?
                ByteBuffer[] 里面只有一帧画面的每个像素的颜色数值.
                Frame存的为 电脑桌面的帧信息时,ByteBuffer[]里下标0位置是一帧画面的每个像素B、G、R三个颜色数值,其他下标是空的!.
            只有0下标有信息时,为什么不直接用ByteBuffer变量呢?
                因为Frame不仅可以获取电脑桌面的帧信息,还可以获取摄像头等的帧信息,当Frame获取的是摄像头帧信息时,画面里的每个像素颜色数值就分成三个
                ByteBuffer存放了所以,虽然电脑桌面的一帧画面的像素颜色信息用一个ByteBuffer就放完了,但为了适配,通用,我们统一成了一个名为image的
                ByteBuffer数组.
            那现在frame.image[0]明白了,是拿取帧对象里像素颜色信息.但为什么要把这个信息当成BytePointer对象参数呢?
                是为了下面这一行做准备"srcMat = frameToMat(frame, bytePtr);".
                推理模型要的是一帧画面的(NCHW 是一种yolo推理模型的张量结构)NCHW张量格式信息 "单纯一维 float [] 浮点数组"，
                我们需要把Frame里的各种变量表示的帧信息转变为NCHW张量格式信息float [] 浮点数组.在转化过程中有缩放、变色道、
                像素转换这些复杂的,对图像信息的操作.但OpenCV有一套工具库,利用OpenCV工具转换虽然省去了对于信息转换的低层逻辑
                的编写,但OpenCV是C++编写的,我们说Frame里的一些信息如ByteBuffer[]里的像素颜色信息,说的是(引用地址) 底层存的java能看懂的 指向
                计算机内存的真实地址.如ByteBuffer[0]---->代表0x0001,指向的是内存的模块区域,这个区域里存放的才是帧信息里的像素颜色信息.
                C++ 写的程序OpenCV是看不懂这个引用地址0x0001指向内存的哪个区域,找不到内存区域就找不到信息.BytePointer是 Java封装
                出来的对接工具，作用只有一个：把 Java 这边 ByteBuffer 里的像素信息，映射成 C++ 能识别的内存地址，打通两边。
                没有它，OpenCV 根本找不到你那一帧画面的像素数据
            frameToMat方法是 Java 的工具方法，那么我们给它传参的时候,它不是能够看懂 ByteBuffer里面的数据引用吗？
            那应该是在这一步可以直接给它传 ByteBuffer呀，为什么要先给它转成 C++ 能看懂的引用呢？
                frameToMat 看着像 Java 方法，但它内部底层还是要跳进 C++ OpenCV 代码干活，它只是一层 Java 壳子，真正实现他功能的全靠
                C++ 原生逻辑.所以我们要给他c++能看懂的数据引用.
            frameToMat功能是什么?为什么?
                读取传入的 frame 对象，取出画面宽、高、通道数、像素格式这些规格参数;
                接收已经包装好 C++ 指针的 bytePtr，拿到像素内存的原生访问地址;
                在 C++ 内存中创建标准的 cv::Mat 图像结构体，把像素指针、尺寸、通道全部绑定在一起;
                给 Java 返回一个 Mat 包装对象，让你在 Java 代码里可以调用 OpenCV 所有图像操作函数（缩放、换色、类型转换等）。
            
             */
            // 这行：把Java的ByteBuffer包装成C++能识别的BytePointer指针，打通跨语言访问
            bytePtr = new BytePointer((ByteBuffer) frame.image[0]);
            // 这行：调用frameToMat方法，把Frame转成OpenCV的Mat对象
            srcMat = frameToMat(frame, bytePtr);
            // 这行：校验生成的Mat是否为空，避免空图导致后续处理崩溃
            if (srcMat.empty()) {
                throw new RuntimeException("当前帧图像为空");
            }

            // 填充色：用YOLO训练时默认的114灰色
            /*
            什么是填充色?
                我们原始画面的尺寸大多不是正方形，而 YOLO 模型要求输入图像必须是固定尺寸的正方形。处理时会先对原图做等比例缩放
                保证画面里的物体不会变形。缩放完成后，画面无法完全占满标准正方形画布，四周会多出空白区域。填充色，就是专门用来
                填满这些空白区域的颜色。本代码中选用了灰色 (114,114,114) 作为填充色。
            为什么用114这个值？
                用 114 这个值是和我们训练 YOLO 模型的参数是绑定的，在训练 YOLO 模型的时候，如果要是改变了这个参数，那么这里也需要改变!
                YOLO模型训练的时候,填充边框默认用的就是这个灰色,不然输入的边框颜色和训练时不一样，会影响模型的识别精度。
             */
            // 这行：创建Scalar对象，定义填充的灰色，三个通道都是114
            padColor = new Scalar(114, 114, 114, 0);

            // 2. 等比例缩放+填充到1280x1280，同时带回缩放/补边参数
            /*
            为什么要做缩放补边？
                YOLO模型只认固定1280x1280的正方形输入，原图尺寸五花八门，必须调整到这个尺寸。
                不能直接拉伸原图！如果硬把长方形的图拉成正方形，画面里的物体就会变形，模型从来没见过变形的物体，就认不出来了。
                所以我们用等比例缩放，保证物体比例不变，剩下的空地方填充灰色边框，凑成正方形。
                同时我们要记录缩放比例和补边的大小，后面后处理要把模型输出的框，还原回原图的坐标。
             */
            // 这行：调用scaleAndPad方法，完成缩放补边，拿到处理后的结果和参数
            ScalePadResult scalePadResult = scaleAndPad(srcMat, padColor);
            // 这行：从结果里取出处理完的标准尺寸Mat
            paddedMat = scalePadResult.paddedMat;
            // 这行：从结果里取出缩放比例，后处理用
            float scaleRatio = scalePadResult.scaleRatio;
            // 这行：从结果里取出顶部补边数，后处理用
            int padTop = scalePadResult.padTop;
            // 这行：从结果里取出左侧补边数，后处理用
            int padLeft = scalePadResult.padLeft;

            // 3. BGR 转 RGB
            /*
            为什么要转通道顺序？
                FFmpeg抓取的屏幕帧，原生的像素顺序是B、G、R（蓝、绿、红），这是FFmpeg默认的输出格式。
                但YOLO模型训练的时候，用的是R、G、B（红、绿、蓝）的顺序。
                如果不转换，颜色就会完全错位，模型把蓝色当成红色，红色当成蓝色，根本认不出画面里的物体。
             */
            // 这行：调用cvtColor方法，把paddedMat的通道从BGR转成RGB，原地修改
            /**
                第一个参数：待转换的原图
                第二个参数：转换后保存结果的图像
                这里写 paddedMat, paddedMat，代表转换后的结果直接覆盖原图像，也就是原地修改
            */
            cvtColor(paddedMat, paddedMat, COLOR_BGR2RGB);

            // 4. 转float32格式，准备归一化
            /*
            为什么要转float32？
                原图像素是0~255的8位整数，而我们后面要做归一化，把整数转成0~1的小数，
                必须先把图像的类型转成32位浮点型，才能存储小数，不然整数存不了小数会出错。
             */
            // 这行：创建临时的floatMat，用来存转成浮点型的图像
            floatMat = new Mat();
            // 这行：调用convertTo方法，把paddedMat转成CV_32F浮点型，存到floatMat里
            paddedMat.convertTo(floatMat, CV_32F);

            // 把C++ Mat里的浮点数据读到Java数组里，减少拷贝
            /*
            为什么要读到Java数组里？
                后面的归一化、转NCHW，我们直接在Java层做，不用把处理完的数据再写回C++的Mat里，
                这样可以减少一次内存拷贝，提升处理速度，适合实时截屏的高帧率场景。
             */
            // 这行：创建FloatBuffer，把floatMat里的数据包装成Buffer，方便读取
            FloatBuffer buf = floatMat.createBuffer();
            // 这行：计算总像素数，高*宽*通道，就是所有像素的总个数
            int pixelCount = floatMat.rows() * floatMat.cols() * floatMat.channels();
            // 这行：创建Java的float数组，用来存读取出来的像素数据
            float[] hwcData = new float[pixelCount];
            // 这行：把Buffer里的C++端的浮点数据，读到Java的hwcData数组里
            buf.get(hwcData);

            // 全部像素归一化 /255
            /*
            什么是归一化？
                就是把原本0~255的像素整数，全部压缩到0.0~1.0的小数范围。
            为什么要做归一化？
                神经网络这个计算工具，最擅长处理0~1之间的小数，太大的数值会导致计算精度下降、梯度爆炸。
                而且YOLO模型训练的时候，就是用的0~1范围的输入，我们必须和训练时的输入范围完全对齐，不然模型识别会出错。
            怎么做的？
                把每个像素值都除以255，255是8位像素的最大值，这样最大的255就变成1.0，最小的0就变成0.0，刚好落到0~1区间。
             */
            // 这行：遍历所有像素，每个像素值除以255，完成归一化
            for (int i = 0; i < pixelCount; i++) {
                hwcData[i] /= 255.0f;
            }

            // 直接用归一化后的hwcData转NCHW，省去put回Mat的开销
            /*
            什么是HWC？
                是OpenCV默认的像素排布格式：每个像素的R、G、B三个值挨在一起，按行依次排列所有像素。
                比如第一个像素R1、G1、B1，第二个R2、G2、B2，数组就是[R1,G1,B1,R2,G2,B2...]
            什么是NCHW？
                是YOLO推理模型要求的张量排布格式：先把所有R通道的像素排完，再排所有G通道，最后排所有B通道。
                比如上面的例子，就会排成[R1,R2,G1,G2,B1,B2...]
            为什么要转这个格式？
                GPU做卷积运算的时候，按通道整块读取数据效率更高，速度更快，这是深度学习框架的标准输入格式。
                而且模型训练的时候就是用的NCHW格式，我们必须和它的输入排布完全一致，不然模型读数据会错位，识别完全乱掉。
             */
            // 这行：调用hwcToNchw方法，把HWC数组转成NCHW数组
            float[] inputData = hwcToNchw(hwcData, MODEL_SIZE, MODEL_SIZE, 3);

            // 这行：把处理完的输入张量和参数打包，返回给上层
            return new PreProcessResult(inputData, scaleRatio, padTop, padLeft);
        } finally {
            // 统一释放所有原生资源
            /*
            为什么要手动close这些对象？
                这些Mat、BytePointer、Scalar都是C++内存里的对象，Java的垃圾回收机制管不到C++的内存。
                如果不手动释放，每处理一帧就会泄漏一点内存，时间长了程序越跑越卡，最后内存耗尽崩溃。
                所以我们必须用完一个释放一个，把内存还给系统。
             */
            // 这行：释放padColor对象，回收C++内存
            if (padColor != null) padColor.close();
            // 这行：释放bytePtr对象，回收C++内存
            if (bytePtr != null) bytePtr.close();
            // 这行：释放srcMat对象，回收C++内存
            if (srcMat != null) srcMat.close();
            // 这行：释放paddedMat对象，回收C++内存
            if (paddedMat != null) paddedMat.close();
            // 这行：释放floatMat对象，回收C++内存
            if (floatMat != null) floatMat.close();
        }
    }

    /** 缩放补边的内部结果包装类：把处理后的图像和配套参数打包返回 */
    private static class ScalePadResult {
        final Mat paddedMat;     // 缩放补边后的1280x1280标准图像
        final float scaleRatio;  // 原图的缩放比例
        final int padTop;        // 顶部补边像素数
        final int padLeft;       // 左侧补边像素数
        // 构造方法：把参数赋值给成员变量，完成打包
        ScalePadResult(Mat paddedMat, float scaleRatio, int padTop, int padLeft) {
            this.paddedMat = paddedMat;
            this.scaleRatio = scaleRatio;
            this.padTop = padTop;
            this.padLeft = padLeft;
        }
    } 

    /** FFmpeg Frame 转为 Mat，外部传入BytePointer方便释放 */
    private static Mat frameToMat(Frame frame, BytePointer data) {
        /*
        把FFmpeg的Frame和包装好的指针，转成OpenCV的Mat对象
         */
        // 这行：创建图像类型，CV_MAKETYPE是把深度和通道数组合成OpenCV的类型
        // CV_8U是8位无符号整数，frame.imageChannels是3通道，所以组合起来就是3通道8位整数，对应BGR彩色图
        int type = CV_MAKETYPE(CV_8U, frame.imageChannels);
        // 这行：调用OpenCV的Mat构造函数，创建Mat对象
        // 参数依次是：高、宽、图像类型、数据指针
        // 这里是零拷贝，像素数据没有复制，只是共享内存地址，速度很快
        return new Mat(frame.imageHeight, frame.imageWidth, type, data);
    }

    /** 等比例缩放，短边填充灰色至正方形MODEL_SIZE，带回缩放补边参数 */
    private static ScalePadResult scaleAndPad(Mat src, Scalar padColor) {
        /*
        为什么要做缩放补边？
            YOLO模型只认固定1280x1280的正方形输入，原图尺寸五花八门，必须调整到这个尺寸。
            不能直接拉伸原图！如果硬把长方形的图拉成正方形，画面里的物体就会变形，模型从来没见过变形的物体，就认不出来了。
            所以我们用等比例缩放，保证物体比例不变，剩下的空地方填充灰色边框，凑成正方形。
            同时我们要记录缩放比例和补边的大小，后面后处理要把模型输出的框，还原回原图的坐标。
         */
        // 这行：拿到原图的宽度，Mat的cols就是列数，对应图像的宽度
        int w = src.cols();
        // 这行：拿到原图的高度，Mat的rows就是行数，对应图像的高度
        int h = src.rows();

        // 这行：计算缩放比例，取宽和高各自缩放比例的最小值
        // 比如宽要缩到1280需要乘0.666，高要缩到1280需要乘1.185，取小的0.666
        // 这样能保证缩完之后宽高都不超过1280，而且画面不会变形
        float scaleRatio = Math.min((float) MODEL_SIZE / w, (float) MODEL_SIZE / h);
        // 这行：算出缩放之后的新宽度，用原图宽乘缩放比例
        int newW = Math.round(w * scaleRatio);
        // 这行：算出缩放之后的新高度，用原图高乘缩放比例
        int newH = Math.round(h * scaleRatio);
        // 这行：创建缩放用的Size对象，告诉OpenCV我们要缩到多大
        Size resizeSize = new Size(newW, newH);

        // 这行：创建临时的Mat，用来存缩放后的图像
        Mat resizeImg = new Mat();
        // 这行：调用OpenCV的resize方法执行缩放
        // INTER_LINEAR是双线性插值，保证缩放后画面清晰，没有锯齿
        resize(src, resizeImg, resizeSize, 0, 0, INTER_LINEAR);
        // 这行：用完的Size临时对象，立刻释放，避免内存泄漏
        resizeSize.close();

        // 这行：计算左边要补多少边，把缩放后的图居中，左边补一半的空
        int padLeft = (MODEL_SIZE - newW) / 2;
        // 这行：计算右边要补多少边，剩下的空补到右边
        int padRight = MODEL_SIZE - newW - padLeft;
        // 这行：计算上边要补多少边，上边补一半的空
        int padTop = (MODEL_SIZE - newH) / 2;
        // 这行：计算下边要补多少边，剩下的空补到下边
        int padBottom = MODEL_SIZE - newH - padTop;

        // 这行：创建最终的输出Mat，存补完边的图像
        Mat out = new Mat();
        // 这行：调用OpenCV的copyMakeBorder方法执行补边
        // 参数依次是：输入图、输出图、上补边、下补边、左补边、右补边、填充类型、填充颜色
        copyMakeBorder(
                resizeImg,
                out,
                padTop, padBottom,
                padLeft, padRight,
                BORDER_CONSTANT,
                padColor
        );
        // 这行：用完的缩放临时图，立刻释放，避免内存泄漏
        resizeImg.close();

        // 这行：把处理完的图和配套参数打包，返回给上层
        return new ScalePadResult(out, scaleRatio, padTop, padLeft);
    }

    /**
     * HWC浮点数组 快速转为 NCHW 输出数组
     * @param hwcData 归一化后的HWC数组
     * @param h 高
     * @param w 宽
     * @param c 通道
     * @return NCHW一维float数组
     */
    private static float[] hwcToNchw(float[] hwcData, int h, int w, int c) {
        /*
        什么是HWC？
            是OpenCV默认的像素排布格式：每个像素的R、G、B三个值挨在一起，按行依次排列所有像素。
            比如第一个像素R1、G1、B1，第二个R2、G2、B2，数组就是[R1,G1,B1,R2,G2,B2...]
        什么是NCHW？
            是YOLO推理模型要求的张量排布格式：先把所有R通道的像素排完，再排所有G通道，最后排所有B通道。
            比如上面的例子，就会排成[R1,R2,G1,G2,B1,B2...]
        为什么要转这个格式？
            GPU做卷积运算的时候，按通道整块读取数据效率更高，速度更快，这是深度学习框架的标准输入格式。
            而且模型训练的时候就是用的NCHW格式，我们必须和它的输入排布完全一致，不然模型读数据会错位，识别完全乱掉。
         */
        // 这行：计算总数据量，N=1（单张图），所以总大小是1*通道数*高*宽
        int totalSize = 1 * c * h * w;
        // 这行：创建新的数组，用来存NCHW格式的输出数据
        float[] output = new float[totalSize];
        // 这行：输出数组的下标指针，用来记录当前写到哪个位置了
        int idx = 0;
        // 第一层循环：遍历通道，先处理R，再G，再B
        for (int ch = 0; ch < c; ch++) {
            // 第二层循环：遍历图像的行，一行一行处理
            for (int y = 0; y < h; y++) {
                // 第三层循环：遍历图像的列，一行里的每个像素
                for (int x = 0; x < w; x++) {
                    // 这行：计算当前像素在HWC数组里的位置
                    // 公式是：(行*宽+列)*通道数 + 当前通道，就能找到这个通道对应的像素值的位置
                    int hwcPos = (y * w + x) * c + ch;
                    // 这行：把HWC里的这个值，写到NCHW数组的当前位置，然后下标指针加1
                    output[idx++] = hwcData[hwcPos];
                }
            }
        }
        // 这行：返回排好序的NCHW数组
        return output;
    }
    //-------------------------------------------------------------------------------------
}