import requests

api_key = "6CY7dleFpEvO3vMDcbXdEOE5Ih8i22G5"
chat_url = "https://codestral.mistral.ai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how can you help me today?"}
    ]
}

response = requests.post(chat_url, headers=headers, json=data)
print(response.json())
