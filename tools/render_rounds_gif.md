# render_rounds_gif.py

将 VIGA 每轮生成的 `state.blend` 文件渲染为 360° 旋转 GIF 动画。

## 功能

- 自动检测 rounds 目录下的 `state.blend` 文件
- 使用 **ray-casting** 从原始相机位置确定旋转锚点
- 从原始相机视角开始平滑旋转 360°
- 保留原始相机焦距（不覆盖 GPT 设定的 lens）
- 渲染动画第 1 帧（物体在初始位置）
- 输出 PNG 帧序列 + 合成 GIF + RESULTS.md 汇总文档

## 用法

```bash
python tools/render_rounds_gif.py \
    --renders-dir <rounds目录> \
    --output-dir <输出目录> \
    --blender-command <Blender路径> \
    [--target-image <目标图片>] \
    [--num-frames 36] \
    [--resolution 512] \
    [--gif-size 384]
```

## 参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--renders-dir` | 是 | - | 包含轮次子目录的路径，每个子目录下需有 `state.blend` |
| `--output-dir` | 是 | - | GIF 和帧 PNG 的输出目录 |
| `--blender-command` | 是 | - | Blender 可执行文件路径 |
| `--target-image` | 否 | 无 | 目标图片路径，会嵌入 RESULTS.md |
| `--num-frames` | 否 | 36 | 旋转帧数（36 = 每帧 10°） |
| `--resolution` | 否 | 512 | 渲染分辨率（像素） |
| `--gif-size` | 否 | 384 | GIF 输出尺寸（像素） |

## 示例

### VIGA Dynamic Scene 输出

```bash
python tools/render_rounds_gif.py \
    --renders-dir output/dynamic_scene/20260206_011742/artist/renders \
    --output-dir output/dynamic_scene/20260206_011742/artist/gifs \
    --blender-command "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" \
    --target-image data/dynamic_scene/artist/target.png
```

### 单个 .blend 文件（手动创建 rounds 结构）

如果只有一个 .blend 文件，创建目录结构后使用：

```bash
mkdir -p /tmp/rounds/1
cp my_scene.blend /tmp/rounds/1/state.blend

python tools/render_rounds_gif.py \
    --renders-dir /tmp/rounds \
    --output-dir /tmp/gifs \
    --blender-command blender
```

## 输入目录结构

```
renders-dir/
├── 1/
│   ├── state.blend          # 必需
│   └── Camera_f0001.png     # 可选（keyframe，会嵌入 MD）
├── 2/
│   ├── state.blend
│   ├── Camera_f0001.png
│   ├── Camera_f0100.png
│   └── Camera_f0200.png
├── ...
```

脚本会自动跳过没有 `state.blend` 的目录和不含 mesh 对象的场景。

## 输出目录结构

```
output-dir/
├── round_1_frames/           # 每轮的帧序列
│   ├── frame_000.png         # 第 0 帧 = 原始相机视角
│   ├── frame_001.png         # 旋转 10°
│   ├── ...
│   └── frame_035.png         # 旋转 350°
├── round_1.gif               # 合成的 GIF
├── round_2_frames/
├── round_2.gif
├── ...
RESULTS.md                    # 在 output-dir 的上级目录
```

## 核心算法：Ray-casting 锚点

传统的包围盒方法对 GPT 生成的场景不可靠（地板/墙壁可达数百单位），所以用 ray-casting：

```
1. 从原始相机位置，沿 ±20° 范围发射 9×9 = 81 条射线
2. scene.ray_cast() 检测与场景的交点，记录深度
3. 取所有命中深度的 median 值 D
4. anchor = cam_pos + cam_forward × D
5. 从原始相机位置（start_angle, elevation）开始，围绕 anchor 旋转 360°
```

这样 GIF 的第一帧就是原始相机视角（与 VIGA 渲染的 keyframe 一致），然后平滑过渡到旋转。

## 依赖

- **Blender** >= 4.0（使用 `BLENDER_EEVEE_NEXT` 引擎）
- **Pillow** (`pip install Pillow`)

## 注意事项

- Windows 下 Blender 路径含空格时需用引号包裹
- 动态场景（有物理模拟）会自动 `scene.frame_set(1)` 渲染初始状态
- 轨道半径 clamp 到 1.0~30.0 范围，防止极端情况
- 如果场景没有相机，会自动创建一个默认相机
