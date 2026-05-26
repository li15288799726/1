#!/usr/bin/env python3
"""Test if get_agent_config can parse the nested YAML config."""
import os

config_path = "/home/administrator/.hermes/profiles/qa-tester/config.yaml"
env_path = "/home/administrator/.hermes/profiles/qa-tester/.env"

model_id = "deepseek-chat"
provider = "deepseek"
base_url = "https://api.deepseek.com"
api_key = ""

# Simulate get_agent_config parsing
if os.path.isfile(config_path):
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("model.default:"):
                val = line.split(":", 1)[1].strip().strip("\"'")
                if val:
                    model_id = val
            elif line.startswith("model.provider:"):
                val = line.split(":", 1)[1].strip().strip("\"'")
                if val:
                    provider = val
            elif line.startswith("model.base_url:"):
                val = line.split(":", 1)[1].strip().strip("\"'")
                if val:
                    base_url = val

print("=== Config.yaml content ===")
with open(config_path) as f:
    for i, line in enumerate(f, 1):
        print(f"  {i}: {line.rstrip()}")

print(f"\n=== Parsed result ===")
print(f"  model_id:  '{model_id}'")
print(f"  provider:  '{provider}'")
print(f"  base_url:  '{base_url}'")
print(f"  api_key_configured: {bool(api_key)}")
print()

# Let's also check the actual .env to see what the API key value is
if os.path.isfile(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k == "GLM_API_KEY":
                    v_show = v[:8] + "..." + v[-4:] if len(v) > 12 else v
                    print(f"  GLM_API_KEY = {v_show}")
                if k == "DEEPSEEK_API_KEY":
                    v_show = v[:8] + "..." + v[-4:] if len(v) > 12 else v
                    print(f"  DEEPSEEK_API_KEY = {v_show}")
                if k == "CUSTOM_API_KEY":
                    v_show = v[:8] + "..." + v[-4:] if len(v) > 12 else v
                    print(f"  CUSTOM_API_KEY = {v_show}")
