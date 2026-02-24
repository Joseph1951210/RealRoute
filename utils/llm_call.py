"""
utils/llm_call.py

This module calls the OpenAI Chat API with retry mechanism.
"""

import os
import requests
import time

# 设置环境变量以解决 Unicode 编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 清理可能包含非 ASCII 字符的代理环境变量（可能导致 latin-1 编码错误）
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if key in os.environ:
        value = os.environ[key]
        try:
            # 尝试用 latin-1 编码，如果失败则删除
            value.encode('latin-1')
        except UnicodeEncodeError:
            del os.environ[key]


def call_openai_chat(prompt: str, api_key: str, model: str, base_url: str, max_retries: int = 3) -> str:
    """Call OpenAI Chat API with retry mechanism

    Args:
        prompt: Prompt text
        api_key: API key
        model: Model name
        base_url: API base URL
        max_retries: Max retry times

    Returns:
        str: API response content
    """
    if not base_url:
        base_url = "https://api.openai.com/v1"
    url = f"{base_url}/chat/completions"
    # 确保 API key 是纯 ASCII 字符串
    api_key_str = str(api_key).strip()
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key_str}",
    }
    # 确保 prompt 是 UTF-8 编码的字符串
    prompt_str = str(prompt) if isinstance(prompt, str) else prompt
    payload = {
        "model": str(model),
        "messages": [{"role": "user", "content": prompt_str}],
        "temperature": 0,
    }
    
    session = requests.Session()
    retry_strategy = requests.adapters.Retry(
        total=max_retries,
        backoff_factor=2,  # Increase backoff time
        status_forcelist=[500, 502, 503, 504, 408, 429],  # Add timeout and rate limit status codes
        allowed_methods=["POST"]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy, pool_maxsize=100)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        response = session.post(url, headers=headers, json=payload, timeout=60)  # Increase timeout to 60 seconds
        response.raise_for_status()  # Check response status
        response.encoding = 'utf-8'  # 显式设置响应编码
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"🔴 API request error: {str(e)}")
        if isinstance(e, (requests.exceptions.ChunkedEncodingError, requests.exceptions.ReadTimeout)):
            print("Detected connection error, retrying...")
            # Special handling for connection errors
            for i in range(max_retries):
                try:
                    print(f"Retry #{i+1}...")
                    time.sleep(2 ** i)  # Exponential backoff
                    response = session.post(url, headers=headers, json=payload, timeout=60)
                    response.raise_for_status()
                    response.encoding = 'utf-8'  # 显式设置响应编码
                    return response.json()["choices"][0]["message"]["content"]
                except requests.exceptions.RequestException as retry_e:
                    print(f"Retry #{i+1} failed: {str(retry_e)}")
                    if i == max_retries - 1:  # If it's the last retry
                        print("All retries failed")
                        return ""
    except Exception as e:
        print(f"🔴 Other error: {str(e)}")
        return ""
    finally:
        session.close()
