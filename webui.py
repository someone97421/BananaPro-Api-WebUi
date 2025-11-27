import os
import gradio as gr
from google import genai
from google.genai import types
from PIL import Image
import time
import datetime

# ==============================================================================
# 🌐 网络代理设置
# ==============================================================================
PROXY_URL = "http://127.0.0.1:7897" 
os.environ["http_proxy"] = PROXY_URL
os.environ["https_proxy"] = PROXY_URL

# ==============================================================================
# 🛠️ 辅助函数：日志与历史
# ==============================================================================
def get_time_str():
    return datetime.datetime.now().strftime("%H:%M:%S")

def append_log(current_log, message):
    """向日志文本追加新行"""
    new_line = f"[{get_time_str()}] {message}\n"
    if current_log is None:
        current_log = ""
    return current_log + new_line

def get_history_images(output_dir):
    """读取输出目录下的所有图片，按修改时间倒序排列"""
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), "outputs")
    
    if not os.path.exists(output_dir):
        return []

    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    images = []
    try:
        files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.lower().endswith(valid_exts)]
        files.sort(key=os.path.getmtime, reverse=True)
        images = files
    except Exception as e:
        print(f"读取历史失败: {e}")
        
    return images

# ==============================================================================
# 🧠 核心生成逻辑 (多图参考 -> 单次生成 -> 自动重试)
# ==============================================================================
def generate_image(api_key, prompt, ref_image_gallery, resolution, aspect_ratio, output_dir, current_logs):
    
    # 1. 检查 API Key
    logs = append_log(current_logs, "🚀 任务启动...")
    yield None, "⏳ 初始化...", gr.update(), logs 

    if not api_key:
        logs = append_log(logs, "❌ 错误：未提供 API Key")
        yield None, "❌ 缺少 API Key", gr.update(), logs
        return
    
    # 2. 检查/创建目录
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), "outputs")
    
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except Exception as e:
        logs = append_log(logs, f"❌ 目录创建失败: {e}")
        yield None, f"❌ 目录错误: {e}", gr.update(), logs
        return

    # 3. 初始化客户端
    try:
        logs = append_log(logs, "🔌 连接 API 客户端...")
        client = genai.Client(api_key=api_key)
    except Exception as e:
        logs = append_log(logs, f"❌ 客户端初始化失败: {e}")
        yield None, f"❌ API Key 错误: {e}", gr.update(), logs
        return

    # 4. 准备数据 (构建唯一的 contents 上下文列表)
    contents = [prompt]
    logs = append_log(logs, f"📝 提示词已装载")
    
    # --- 核心逻辑：多图打包 ---
    if ref_image_gallery:
        # 限制最多10张
        process_images = ref_image_gallery
        if len(ref_image_gallery) > 10:
            logs = append_log(logs, "⚠️ 图片超过10张，仅截取前10张作为参考")
            process_images = ref_image_gallery[:10]
        
        loaded_count = 0
        for i, img_entry in enumerate(process_images):
            try:
                # 兼容 Gradio Gallery 的不同返回格式
                if isinstance(img_entry, (tuple, list)):
                    img_path = img_entry[0]
                else:
                    img_path = img_entry
                
                img = Image.open(img_path)
                # 关键：将图片添加到同一个 contents 列表中
                contents.append(img)
                loaded_count += 1
            except Exception as e:
                logs = append_log(logs, f"⚠️ 图片 {i+1} 加载失败: {e}")
        
        if loaded_count > 0:
            logs = append_log(logs, f"📦 已将 {loaded_count} 张参考图打包进上下文")
            yield None, f"⏳ 已加载 {loaded_count} 张参考图...", gr.update(), logs

    # 5. 配置参数
    try:
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig( 
                image_size=resolution,           
                aspect_ratio=aspect_ratio 
            )
        )
    except AttributeError:
        logs = append_log(logs, "❌ 库版本过旧，请升级 google-genai")
        yield None, "❌ 库版本过旧", gr.update(), logs
        return

    # 6. 发送请求 (新增：自动重试逻辑)
    generated_images = []
    status_msg = ""
    
    # 重试参数配置
    max_retries = 3
    retry_delay = 5  # 秒
    response = None

    for attempt in range(max_retries):
        try:
            # 如果是重试，先提示并等待
            if attempt > 0:
                logs = append_log(logs, f"🔄 第 {attempt+1} 次尝试重新连接服务器...")
                yield None, f"⏳ 服务器繁忙，重试 ({attempt+1}/{max_retries})...", gr.update(), logs
                time.sleep(retry_delay)
            else:
                logs = append_log(logs, "☁️ 向 Google 发送单次生成请求...")
                yield None, "☁️ 生成中...", gr.update(), logs

            # --- API 调用 ---
            response = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=contents,
                config=config
            )
            
            # 如果成功运行到这行，说明没有报错，跳出循环
            break 

        except Exception as e:
            error_str = str(e)
            # 判断是否为 503 或 overloaded
            is_overloaded = "503" in error_str or "overloaded" in error_str.lower() or "UNAVAILABLE" in error_str
            
            # 如果是繁忙错误且还有重试次数，则 continue 继续循环
            if is_overloaded and attempt < max_retries - 1:
                logs = append_log(logs, f"⚠️ 检测到服务器拥堵 (503)，{retry_delay}秒后重试...")
                yield None, "⚠️ 服务器繁忙，准备重试...", gr.update(), logs
                continue
            
            # 如果是其他错误，或者重试次数用尽，则报错并退出
            logs = append_log(logs, f"❌ API 调用失败: {error_str}")
            new_history = get_history_images(output_dir)
            
            if "SSL" in error_str or "connection" in error_str.lower():
                logs = append_log(logs, "💡 请检查代理端口 (127.0.0.1:7897)")
                yield None, "❌ 网络错误", new_history, logs
            else:
                yield None, f"❌ 失败: {e}", new_history, logs
            return  # 彻底终止函数

    # 7. 处理响应 (只有上面 break 出循环后才会执行到这里)
    try:
        logs = append_log(logs, "✅ 服务器响应成功，正在下载图片...")
        
        if response and response.parts:
            for i, part in enumerate(response.parts):
                if part.inline_data:
                    img = part.as_image()
                    timestamp = int(time.time())
                    filename = f"gemini_{resolution}_{i}_{timestamp}.png"
                    full_path = os.path.join(output_dir, filename)
                    img.save(full_path)
                    generated_images.append(full_path)
                    logs = append_log(logs, f"💾 图片已保存: {filename}")
                elif part.text:
                    logs = append_log(logs, f"ℹ️ 模型反馈: {part.text}")
                    status_msg += f" {part.text}"
        
        new_history = get_history_images(output_dir)
        
        if generated_images:
            logs = append_log(logs, "🎉 任务完成")
            yield generated_images, "🎉 生成成功", new_history, logs
        else:
            logs = append_log(logs, "⚠️ 任务结束但未生成图片")
            yield None, "⚠️ 未生成图片", new_history, logs

    except Exception as e:
        logs = append_log(logs, f"❌ 图片保存失败: {str(e)}")
        yield None, f"❌ 保存错误: {e}", get_history_images(output_dir), logs

# ==============================================================================
# 🎨 界面布局
# ==============================================================================
css = """
#log_box { font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; }
"""

with gr.Blocks(title="Gemini 3 Pro Generator") as demo:
    gr.Markdown("## 🍌 Gemini 3 Pro 图像生成器")
    
    default_output_dir = os.path.join(os.getcwd(), "outputs")
    
    with gr.Row():
        # --- 主要操作区 ---
        with gr.Column(scale=4):
            with gr.Row():
                # 左半边：输入
                with gr.Column(scale=1):
                    api_key_input = gr.Textbox(label="API Key", type="password", placeholder="在此粘贴 API Key")
                    prompt_input = gr.Textbox(label="提示词 (Prompt)", lines=4, value="A futuristic city, 4k resolution, cinematic lighting.")
                    
                    # 支持多图上传的 Gallery
                    image_input = gr.Gallery(
                        label="上传参考图 (最多10张，模型将同时参考这些图片)", 
                        type="filepath", 
                        interactive=True, 
                        height=250, 
                        columns=4,
                        object_fit="contain"
                    )
                    
                    with gr.Accordion("⚙️ 高级设置", open=True):
                        with gr.Row():
                            res_dropdown = gr.Dropdown(choices=["1K", "2K", "4K"], value="2K", label="分辨率")
                            ratio_dropdown = gr.Dropdown(choices=["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"], value="16:9", label="宽高比")
                        output_dir_input = gr.Textbox(label="保存目录", value=default_output_dir)

                    run_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")
                    status_output = gr.Textbox(label="当前状态", interactive=False)

                # 右半边：当前结果预览
                with gr.Column(scale=1):
                    gallery = gr.Gallery(label="本次生成结果", columns=1, height=600, allow_preview=True)

        # --- 右侧侧边栏 ---
        with gr.Sidebar(position="right", label="历史与日志", open=True):
            
            with gr.Accordion("📠 运行日志", open=True):
                log_output = gr.Textbox(label="Console Log", lines=15, max_lines=20, elem_id="log_box", interactive=False, autoscroll=True)
                
            with gr.Accordion("📜 历史图库", open=False):
                refresh_btn = gr.Button("🔄 刷新历史")
                history_gallery = gr.Gallery(label="本地历史", columns=2, height=500, allow_preview=True)

    # --- 事件绑定 ---
    demo.load(fn=get_history_images, inputs=[output_dir_input], outputs=[history_gallery])
    
    run_btn.click(
        fn=generate_image,
        inputs=[api_key_input, prompt_input, image_input, res_dropdown, ratio_dropdown, output_dir_input, log_output],
        outputs=[gallery, status_output, history_gallery, log_output]
    )
    
    refresh_btn.click(
        fn=get_history_images,
        inputs=[output_dir_input],
        outputs=[history_gallery]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)