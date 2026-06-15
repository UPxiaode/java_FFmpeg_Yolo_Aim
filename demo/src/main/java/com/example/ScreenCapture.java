package com.example;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.FFmpegLogCallback;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.*;


/**screencapture/ˈskriːnkæptʃə(r)/ 屏幕捕获
 * 这个类的作用是 .提供一个方法,每调用一次方法就可以获取桌面一帧的信息
 *  case:         // 调用方法，抓取第一帧
        Frame frame = ScreenCapture.getOneFrame();
    为我们后续不断地获取桌面信息给到yolo模型进行目标检测铺垫.
    还有一件事,当我们不需要获取桌面信息时 记得调用ScreenCapture.stop();关闭这个开关
 */
public class ScreenCapture {
    // FFmpeg 帧抓取器实例 FFmpegFrameGrabber：类型，就是前面用来截屏的采集器类
    //grabber：变量名，用来存放采集器对象的引用  下面的代码就把grabber看作采集器本身!
    private static FFmpegFrameGrabber grabber;

    public static String img="C:\\Users\\KL-A\\Desktop\\PROJECT";

    /*静态代码块: 类加载阶段,优先执行静态代码块，全局只运行1 次. 优先级：静态代码块 > 静态方法 
      这里的作用: 初始化FFmpeg抓取器
     */
    static {
        
        FFmpegLogCallback.set();// 外部包的方法,用于打印ffmpeg详细日志 import org.bytedeco.javacv."FFmpegLogCallback";
        try {
            
            /*
            "new FFmpegFrameGrabber(...)"-->新建一个FFmpeg 画面抓取工具的对象，括号里 "desktop" 是告诉工具：采集源选电脑桌面屏幕
            "grabber ="--> 把刚创建好的抓取工具实例，赋值给变量 grabber，之后全靠这个变量调用抓帧、启停功能
             */
            grabber = new FFmpegFrameGrabber("desktop");
            /*
            前面你传了采集源的地址 "desktop"是对桌面进行采集，但 FFmpeg 不知道用哪种方式抓屏幕!!；setFormat("gdigrab") 明确告诉它：使用
            Windows 原生 GDI 图形接口捕获桌面画面
            */
            grabber.setFormat("gdigrab");
            /*
            前面只是创建对象、设置来源、设置格式，只是配置阶段.执行 start () 后：FFmpeg 内部初始化 gdigrab 抓屏驱动
            建立和桌面画面的数据流通道分配缓冲内存、准备持续输出画面帧,此时状态变为就绪可抓取。
            简单理解：之前是准备工作 .start()是让他从 "准备状态" 转成 "就绪状态!" 转成就绪状态后我们只要给他个信号他就抓取画面信息
            而这个信号就是你调用 grabFrame(). 调用 grabFrame()就能拿到画面帧了
            */
            grabber.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 抓取单帧画面，同时打印本次抓取耗时(毫秒)
     * * @param No_printTime 参数本身没意义,但有参数代表这个方法是 "打印耗时版本",没有参数的重载版本则是 "不打印耗时版本"
     * @return 帧对象，异常返回null
     */
    public static Frame getOneFrame(boolean printTime) {
        // 记录开始时间
        long start = System.currentTimeMillis();//返回当前时间,单位为毫秒
        Frame frame = null;
        try {
            /*采集器grabber调用grabFrame()方法，发出抓取信号，拿到一帧画面数据，存到frame对象里
            抓取失败会返回空值null 所以下面的要用到frame对象的地方都要先判断一下它是不是null，避免程序崩溃
             */
            frame = grabber.grabFrame();
        } catch (Exception e) {
            e.printStackTrace();
        }
        // 计算消耗时间
        long cost = System.currentTimeMillis() - start;
        System.out.println("单帧抓取耗时：" + cost + " ms");
        return frame;
    }
    /**
    * 抓取单帧画面，控制是否打印耗时日志
    
    * @return 帧对象，异常返回null
    * 
    是上面方法的重载版本，
    我认为不打印时间信息可以让程序运行的更快一点.当我们想要根块的运行速度时可以启用! 而打印时间的版本则适合我们调试阶段，观察每帧抓取的耗时
    情况.
    */
        public static Frame getOneFrame() {
        Frame frame = null;
        try {
            frame = grabber.grabFrame();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return frame;
    }


    /**
     * 停止抓取 释放资源，程序结束调用
     */
    public static void stop() {
        try {
            if (grabber != null) {
                grabber.stop();
                grabber.close();
            }
        } catch (Exception ignored) {}
    }
//---------------------------------------------------------------------------------
 // 
    /**
     * 转换工具方法，输入FFmpeg Frame，输出BufferedImage，后续我们就可以用这个BufferedImage对象进行显示、保存等操作了。
     * FFmpeg帧对象转BufferedImage  BufferedImage是图像对象,还不是图片!!
     * @param frame 代表屏幕抓取的帧信息
     * @return 图像对象，帧非法返回null
     */
    /*
    我们拿到的 Frame，里面存放的是屏幕单帧的全部画面信息：包含画面宽高尺寸、总像素数量、每一个像素对应的
    色彩数值、明暗数值，底层全部都是二进制数字数据。这些纯数字本身没法直接展示成肉眼可见的画面，必须依靠绘
    图工具解析、渲染才能显示图像。但是Frame里存放的帧信息的方式并不通用,虽然都是二进制01的组合但是他某个区域的二进制信息可能代表
    的是画面宽度,而通用的格式在这块可能代表的是像素色彩,这就导致我们无法直接用绘图工具把Frame里的数据渲染成图像.所以我们需要先把
    Frame转换成Java通用的图像对象BufferedImage,这个对象里存放的也是二进制数据,但是它的数据格式是被Java的绘图工具认可的,所以我们
    就能用Java的绘图工具把BufferedImage渲染成肉眼可见的画面了.
     * 
     * 这里我们使用JavaCV提供的Java2DFrameConverter这个转换器类来完成Frame到BufferedImage的转换
     * Java2DFrameConverter是专门设计来处理FFmpeg Frame和Java BufferedImage之间转换的工具类，它内部封装了复杂的数据解析、格式
     * 转换逻辑，我们只需要调用它提供的方法就能轻松完成转换，无需关心底层细节。
     *
    */
    public static BufferedImage frameToBufferedImage(Frame frame) {
        // 基础合法性校验
        if (frame == null || frame.image == null) {
            return null;
        }
        // 单次独立转换器，无缓存干扰
        Java2DFrameConverter converter = new Java2DFrameConverter();
        return converter.convert(frame);
    }
//---------------------------------------------------------------------------------
/**
     * 弹出窗口展示图片   将BufferedImage变为图片弹窗展示
     * @param image 转换好的BufferedImage
     * 这个方法一点毛病可能是因为这个"frame.setSize(image.getWidth(), image.getHeight())"
     * 这行代码的作用是根据传入的图像对象的宽度和高度来设置弹窗窗口的尺寸。实际效果是它并不能确
     * 保弹窗的大小正好适合显示图像，只能展示图片一部分!
     */

    public static void showImage(BufferedImage image) {
        // 判空，空图直接不执行，避免空指针报错
        if(image == null) return;

        // 创建一个Swing弹窗窗口，标题叫截图预览
        JFrame frame = new JFrame("截图预览");
        // 点关闭按钮时只销毁这个预览窗口，不把整个程序关掉
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        // 设置窗口整体宽高等于图片宽高
        frame.setSize(image.getWidth(), image.getHeight());
        // JLabel用来承载图片，ImageIcon把BufferedImage包装成Swing能识别的图像图标
        JLabel label = new JLabel(new ImageIcon(image));
        // 把装着图片的标签放进窗口里
        frame.add(label);
        // 把窗口显示出来
        frame.setVisible(true);
    }
//---------------------------------------------------------------------------------
/**
     * 将BufferedImage保存为png图片到指定路径 用于排查FFmpeg帧抓取是否正常
     * @param image 内存图像对象
     * @param filePath 提供要将图片对象转图片后的存放路径文件夹，例如 "D:/screen"
     * @return 保存成功返回true，失败false
     * showImage方法只是将图片通过弹窗展示出来，并没有把图片文件保存到磁盘上，如果我们想要把
     * 抓取的帧保存成图片文件，方便我们查看、分析，就可以用这个方法。
     * 产生这方法的原因是showImage()方法用窗口展示帧对应的图像时并不是完整的桌面,导致我需要验证
     * 我抓的帧信息是不是整个屏幕的画面
     */
    public static boolean saveImageToPath(BufferedImage image, String filePath) {
        String filePath_1=filePath+"\\test.png";
        // 空值判断
        if (image == null || filePath_1 == null || filePath_1.isBlank()) {
            return false;
        }
        try {
            // 创建文件对象并写入png格式
            File saveFile = new File(filePath_1);
            // 自动创建不存在的父文件夹
            if (!saveFile.getParentFile().exists()) {
                saveFile.getParentFile().mkdirs();
            }
            ImageIO.write(image, "png", saveFile);
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }
//------------------------------------------------------------------------------------
// 
/* 测试的遗留,冷知识"可以用来考古!"
    public static void main(String[] args) throws InterruptedException {
        // 调用方法，抓取第一帧
        Frame frame = ScreenCapture.getOneFrame();
        BufferedImage image = ScreenCapture.frameToBufferedImage(frame);
        ScreenCapture.showImage(image);
        ScreenCapture.saveImageToPath(image, img);
  

        Thread.sleep(200);
        // 再次调用，抓取第二帧
        Frame frame2 = ScreenCapture.getOneFrame();
        Frame frame3 = ScreenCapture.getOneFrame();
        System.out.println("画面宽："+frame.imageWidth+" 高："+frame.imageHeight);
        

        // 程序结束，释放资源
        ScreenCapture.stop();
    }
        */
}