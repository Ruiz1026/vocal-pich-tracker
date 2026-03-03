# 🎵 Vocal Pitch Tracker

一个基于 `PyQt5` 的桌面端实时音高检测工具，面向唱歌练习、跟唱分析和基础音准观察。

## ✨ 功能特性

- 🎧 **实时采集桌面播放音频**（系统输出回采，而不是麦克风直录）
- 🎼 **实时音高检测**（YIN + HPS 组合算法）
- 🔢 显示当前频率（Hz）、音名（如 `A4`）、偏差（cents）和置信度
- 🎹 钢琴键盘高亮当前音符
- 📈 频率曲线实时滚动显示
- ▶️/⏹️ 支持一键开始与停止检测

## 🧱 环境要求

- Python `3.10+`（建议）
- Windows（当前默认配置下仅完整支持 Windows 桌面回采）

> 说明：当前 `main.py` 默认是 `allow_microphone_fallback=False`，即优先且仅使用桌面回采模式。

## 🚀 安装步骤

```bash
# 1) 克隆仓库
git clone https://github.com/Ruiz1026/vocal-pich-tracker.git
cd tune_test

# 2) 创建并激活虚拟环境（可选但推荐）
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (CMD)
.venv\Scripts\activate.bat

# 3) 安装依赖
pip install -r requirements.txt
```

## ▶️ 运行方式

```bash
python main.py
```

运行后：

1. 点击 `Start` 开始检测
2. 播放音乐或伴奏
3. 观察频率、音名、偏差与曲线变化
4. 点击 `Stop` 停止检测

## ⚙️ 依赖说明

`requirements.txt` 主要依赖：

- `numpy`
- `sounddevice`
- `soundcard`
- `PyQt5`
- `pyqtgraph`

## 🛠️ 常见问题

- 无法开始采集 / 提示设备错误
  检查系统音频设备是否正常、采样率是否可用，以及是否启用了可用的回采设备（WASAPI/Stereo Mix）。
- 非 Windows 系统运行失败
  当前项目实现以 Windows 桌面回采为主，非 Windows 默认不启用麦克风回退。
- 看不到曲线
  确认 `pyqtgraph` 已正确安装。

---
