# 阶跃星辰 MCP server

## 简介

**本项目仅供个人学习交流使用。**

根据阶跃星辰开放平台[StepFun](https://platform.stepfun.com/docs/overview/concept)提供的各种模型能力，仿照[MiniMax MCP](https://github.com/MiniMax-AI/MiniMax-MCP/tree/main)的实现，开发了一个基于stepfun api 的 MCP server。

## 功能特性

- 支持文本大模型调用
- 支持视觉理解大模型调用
- 支持文生图模型调用
- 支持语音模型调用

## 快速开始

1. **安装**

```bash
pip install stepfun-mcp
# 或者
# git clone git@github.com:weidafeng/StepFunMCP.git
# pip install .
```

2. **配置mcp server config**

可参考：[mcp_server_config_uvx_demo.json](mcp_server_config_uvx_demo.json) 或者 [mcp_server_config_demo.json](mcp_server_config_demo.json)
```json
{
  "mcpServers": {
    "StepFun": {
      "command": "stepfun-mcp",
      "args": [
      ],
      "env": {
        "STEPFUN_API_KEY": "YOUR_STEPFUN_API_KEY_HERE",
        "STEPFUN_API_HOST": "https://api.stepfun.com",
        "STEPFUN_MCP_BASE_PATH": "YOUR_OUTPUT_DIR", 
        "STEPFUN_API_RESOURCE_MODE": "local"
      }
    }
  }
}

## 交流

欢迎关注
- 公众号：[特里斯丹](公众号-特里斯丹.png)
- 知乎：[特里斯丹](https://www.zhihu.com/people/wwdafg)

![](公众号-特里斯丹.png)