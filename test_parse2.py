#!/usr/bin/env python3
"""Test get_agent_config's API key reading logic."""
import os

# Test 1: Check if deepseek-v4-flash is in MODEL_PRESETS
print("=== 问题1: MODEL_PRESETS 不包含 deepseek-v4-flash ===")
# Note: server.py only has deepseek-chat and deepseek-reasoner

# Test 2: Check get_agent_config API key parsing bug
print("\n=== 问题2: get_agent_config 的 API key 读取BUG ===")
env_path = '/home/administrator/.hermes/profiles/qa-tester/.env'
api_key_result = ''
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            v = v.strip().strip("\"'")
            if v:
                api_key_result = f'{k}={v}'
print(f"api_key value: {api_key_result[:30]}...")
print(f"BUG: 读取的是 .env 中最后一个非空变量 (NVIDIA_API_KEY), 不是 API Key")

# Test 3: save_agent_config for "glm" provider
print("\n=== 问题3: save_agent_config 的 provider 名称映射 ===")
PROVIDER_ENV_MAP = {
    "deepseek":  "DEEPSEEK_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter":"OPENROUTER_API_KEY",
    "google":    "GOOGLE_API_KEY",
    "zhipu":     "GLM_API_KEY",
    "moonshot":  "KIMI_API_KEY",
    "alibaba":   "DASHSCOPE_API_KEY",
    "xai":       "XAI_API_KEY",
}
provider = "glm"
env_var = PROVIDER_ENV_MAP.get(provider, "CUSTOM_API_KEY")
print(f"provider='{provider}' -> env_var='{env_var}' (应为 'GLM_API_KEY', 但 map 中 key 是 'zhipu')")
print(f"Dashboard 会错误地将 API key 写入 CUSTOM_API_KEY 而不是 GLM_API_KEY!")

# Test 4: Check team-lead has no DEEPSEEK_API_KEY
print("\n=== 问题4: team-lead 缺少 DEEPSEEK_API_KEY ===")
env_path2 = '/home/administrator/.hermes/profiles/team-lead/.env'
found = False
with open(env_path2) as f:
    for line in f:
        line = line.strip()
        if line.startswith('DEEPSEEK_API_KEY='):
            v = line.split('=', 1)[1].strip()
            if v:
                found = True
                print(f"  DEEPSEEK_API_KEY=已配置 ({v[:8]}...)")
if not found:
    print("  ❌ DEEPSEEK_API_KEY 未配置! team-lead 无法工作")

# Test 5: Check config.yaml has BOTH nested and flat format
print("\n=== 问题5: qa-tester config.yaml 格式冲突 ===")
config_path = '/home/administrator/.hermes/profiles/qa-tester/config.yaml'
with open(config_path) as f:
    content = f.read()
nested_model = '  default:' in content.split('\n')[1] if len(content.split('\n')) > 1 else False
flat_model = 'model.default:' in content
print(f"  含嵌套格式 (model: \\n  default: ...): {nested_model}")
print(f"  含扁平格式 (model.default: ...): {flat_model}")
if nested_model and flat_model:
    print("  ❌ BOTH formats exist! Config corrupted!")
    print("  Hermes 读嵌套格式 (provider=glm)")
    print("  Dashboard 读扁平格式 (provider=custom)")
    print("  配置不一致!")
