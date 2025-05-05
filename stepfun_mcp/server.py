"""
StepFun MCP Server

⚠️ 注意：本服务连接 StepFun API，部分工具调用会产生费用，请谨慎使用。
"""

import os
import logging
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
from pathlib import Path
from fastmcp import FastMCP
from mcp.types import TextContent, ImageContent
from stepfun_mcp.client import (
    StepFunClient,
    ChatCompletionRequest,
    ImageGenerationRequest,
    SpeechGenerationRequest,
    TextMessage,
    VisionMessage,
    VisionContentPartText,
    VisionContentPartImage,
)
from stepfun_mcp.const import *
from stepfun_mcp.exceptions import StepFunAPIError, StepFunAuthError, StepFunRequestError
from stepfun_mcp.utils import (
    build_output_path,
    build_output_file,
    image_to_base64_data_uri,
    save_audio_to_file,
    save_image_to_file,
)

# 加载环境变量
load_dotenv()
api_key = os.getenv(ENV_STEPFUN_API_KEY)
base_path = os.getenv(ENV_STEPFUN_MCP_BASE_PATH) or "~/Desktop/stepfun_mcp_output"
api_host = os.getenv(ENV_STEPFUN_API_HOST, "https://api.stepfun.com")
# resource_mode = os.getenv(ENV_RESOURCE_MODE) or RESOURCE_MODE_URL
resource_mode = "local"
fastmcp_log_level = os.getenv(ENV_FASTMCP_LOG_LEVEL) or "WARNING"

if not api_key:
    raise ValueError("STEPFUN_API_KEY 环境变量未设置")
if not api_host:
    raise ValueError("STEPFUN_API_HOST 环境变量未设置")

mcp = FastMCP("StepFun", log_level=fastmcp_log_level)
client = StepFunClient()

# 文本大模型
@mcp.tool(
    description="调用 StepFun 文本大模型生成聊天回复。\nArgs:\n  messages (list): 聊天历史，格式为 [{'role': 'user', 'content': 'xxx'}]\n  model (str, optional): 模型名称，默认 step-1。\nReturns:\n  TextContent: 回复内容"
)
def stepfun_chat_completion(
    messages: list,
    model: str = DEFAULT_TEXT_MODEL
) -> TextContent:
    try:
        stepfun_messages = [TextMessage(**msg) for msg in messages]
        logging.info(f"DEBUG stepfun_messages: {stepfun_messages}")
        req = ChatCompletionRequest(model=model, messages=stepfun_messages)
        logging.info(f"DEBUG req: {req}")
        resp = client.create_text_completion(req)
        return TextContent(type="text", text=resp.choices[0].message.content)
    except Exception as e:
        return TextContent(type="text", text=f"调用失败: {str(e)}")

# 多模态大模型
@mcp.tool(
    description="调用 StepFun 多模态大模型（支持图片+文本）。\nArgs:\n  messages (list): 聊天历史，格式为 [{'role': 'user', 'content': ...}]\n  model (str, optional): 模型名称，默认 step-1v。\nReturns:\n  TextContent: 回复内容"
)
def stepfun_vision_completion(
    messages: list,
    model: str = DEFAULT_VISION_MODEL
) -> TextContent:
    try:
        stepfun_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, str):
                stepfun_messages.append(TextMessage(role=role, content=content))
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append(VisionContentPartText(text=part.get("text")))
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url")
                        if url:
                            image_data_uri = image_to_base64_data_uri(url, base_path, resource_mode)
                            parts.append(VisionContentPartImage(image_url={"url": image_data_uri}))
                stepfun_messages.append(VisionMessage(role=role, content=parts))
        req = ChatCompletionRequest(model=model, messages=stepfun_messages)
        resp = client.create_vision_completion(req)
        return TextContent(type="text", text=resp.choices[0].message.content)
    except Exception as e:
        return TextContent(type="text", text=f"调用失败: {str(e)}")

# 文生图
@mcp.tool(
    description="调用 StepFun 文生图模型生成图片。\nArgs:\n  prompt (str): 文本描述\n  model (str, optional): 模型名称，默认 step-1x-medium。\n  n (int, optional): 生成图片数量，默认1。\n  size (str, optional): 图片尺寸。\n  seed (int, optional): 随机种子。\nReturns:\n  ImageContent: 图片内容"
)
def stepfun_text2img(
    prompt: str,
    model: str = DEFAULT_IMAGE_MODEL,
    n: int = 1,
    size: str = None,
    seed: int = None,
    output_directory: str = None
):
    try:
        req = ImageGenerationRequest(
            model=model,
            prompt=prompt,
            n=n,
            size=size,
            seed=seed,
            response_format="b64_json"  # if resource_mode == RESOURCE_MODE_URL else "b64_json"
        )
        resp = client.create_image_generation(req)
        # 只返回第一个图片
        img_data = resp.data[0]
        # logging.info(f"Image Base64 JSON: {img_data.b64_json}")
        output_path = build_output_path(output_directory, base_path)
        output_file = build_output_file("image", prompt, output_path, "png")
        save_image_to_file(img_data.b64_json, output_file)
        return TextContent(type="text", text=f"图像已保存: {output_file}")
    except Exception as e:
        logging.error(f"Image generation failed: {str(e)}")
        return TextContent(type="text", text=f"调用失败: {str(e)}")

# 文本转语音
@mcp.tool(
    description="调用 StepFun 语音合成（TTS）接口。\nArgs:\n  text (str): 要合成的文本\n  voice (str, optional): 声音类型，默认 cixingnansheng。\n  model (str, optional): 模型名称，默认 step-tts-mini。\nReturns:\n  TextContent: 返回音频文件路径或URL"
)
def stepfun_text2speech(
    text: str,
    voice: str = DEFAULT_VOICE_ID,
    model: str = DEFAULT_SPEECH_MODEL,
    output_directory: str = None
) -> TextContent:
    try:
        req = SpeechGenerationRequest(model=model, input=text, voice=voice)
        audio_bytes = client.create_speech_generation(req)
        if resource_mode == RESOURCE_MODE_URL:
            return TextContent(type="text", text="StepFun 暂不支持语音URL返回，请使用本地模式")
        else:
            output_path = build_output_path(output_directory, base_path)
            output_file = build_output_file("tts", text, output_path, "mp3")
            save_audio_to_file(audio_bytes, output_file)
            return TextContent(type="text", text=f"音频已保存: {output_file}")
    except Exception as e:
        return TextContent(type="text", text=f"调用失败: {str(e)}")

def main():
    """启动 StepFun MCP 服务器"""
    print("StepFun MCP Server 启动")
    mcp.run()
    

if __name__ == "__main__":
    main()