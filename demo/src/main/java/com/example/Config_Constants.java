package com.example;
/*
    因为我们自己的yolo模型参数会有不同.因为一些功能要拿运行平台的参数.这些变量不能写死,所以这个类把所有的
    要配置的变量都列举了出来,方便我们配置环境.
*/

public class Config_Constants {
    static String modelPath ="demo\\src\\main\\java\\com\\example\\game_aim.onnx";//你训练的yolo模型
    /*
            1: yolo模型要的是.onnx后缀的!
            2:将.pt后缀的模型转.onnx时有一个关键参数 "IR版本" 因为这个项目用的推理模型库版本太低了!-->"import ai.onnxruntime.xxxxx"
        导致最高只能支持IR版本<=9!. 
    */
    static final int MODEL_SIZE=1280;//训练模型时的"imgsz=x?"这个参数
    /*屏幕分辨率
     */
    static  int SCREEN_WIDTH = 2560;
    static  int SCREEN_HEIGHT = 1600;
    
}
