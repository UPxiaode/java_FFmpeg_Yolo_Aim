package com.example;
import com.sun.jna.platform.win32.User32;
import com.sun.jna.platform.win32.WinDef;
import com.sun.jna.platform.win32.WinUser;
import com.sun.jna.platform.win32.BaseTSD;

public class RealTimeAim {
    // ========== 已适配 2560×1600 分辨率 ==========
    private static final int SCREEN_WIDTH = Config_Constants.SCREEN_WIDTH;
    private static final int SCREEN_HEIGHT =Config_Constants.SCREEN_HEIGHT;
    private static final int CENTER_X = SCREEN_WIDTH / 2;  // 1280
    private static final int CENTER_Y = SCREEN_HEIGHT / 2; // 800

    // ========== 跟枪参数（已按2.5K屏比例优化）==========
    private static final float SMOOTH_FACTOR = 0.5f;  // 跟随强度，越小越柔和
    private static final int DEAD_ZONE = 3;             // 死区，偏差小于5像素不修正
    private static final int MAX_STEP = 45;            // 单帧最大移动步长，防止瞬移
    // 有效跟随范围：超出这个范围自动停止跟枪，避免锁死视角
    private static final int AIM_RANGE_X = 1000;        // 中心左右各1000像素
    private static final int AIM_RANGE_Y = 1000;        // 中心上下各1000像素

    // Windows 常量
    private static final int INPUT_MOUSE = 0;
    private static final int MOUSEEVENTF_MOVE = 0x0001;

    /**
     * 实时跟枪入口（常驻开启，无按键触发）
     * @param currentTargetX 目标屏幕X坐标
     * @param currentTargetY 目标屏幕Y坐标
     */
    public static void aimTrack(int currentTargetX, int currentTargetY) {
        // 1. 范围判定：目标超出有效跟随范围，停止跟枪
        int deltaX = currentTargetX - CENTER_X;
        int deltaY = currentTargetY - CENTER_Y;
        if (Math.abs(deltaX) > AIM_RANGE_X || Math.abs(deltaY) > AIM_RANGE_Y) {
            return;
        }

        // 2. 死区判定：微小偏差不修正，防抖动、防磁吸锁死
        if (Math.abs(deltaX) <= DEAD_ZONE && Math.abs(deltaY) <= DEAD_ZONE) {
            return;
        }

        // 3. 计算步长 + 限幅
        int stepX = (int) (deltaX * SMOOTH_FACTOR);
        int stepY = (int) (deltaY * SMOOTH_FACTOR);
        stepX = clamp(stepX, -MAX_STEP, MAX_STEP);
        stepY = clamp(stepY, -MAX_STEP, MAX_STEP);

        // 4. 发送相对鼠标移动（游戏识别原始输入）
        sendRelativeMouse(stepX, stepY);
    }

    // 数值限制工具
    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    // 发送相对鼠标移动
    private static void sendRelativeMouse(int dx, int dy) {
        WinUser.INPUT input = new WinUser.INPUT();
        input.type = new WinDef.DWORD(INPUT_MOUSE);
        input.input.setType("mi");

        WinUser.MOUSEINPUT mi = new WinUser.MOUSEINPUT();
        mi.dx = new WinDef.LONG(dx);
        mi.dy = new WinDef.LONG(dy);
        mi.dwFlags = new WinDef.DWORD(MOUSEEVENTF_MOVE);
        mi.mouseData = new WinDef.DWORD(0);
        mi.dwExtraInfo = new BaseTSD.ULONG_PTR(0);

        input.input.mi = mi;
        WinUser.INPUT[] inputs = {input};

        User32.INSTANCE.SendInput(
                new WinDef.DWORD(1),
                inputs,
                input.size()
        );
    }
}