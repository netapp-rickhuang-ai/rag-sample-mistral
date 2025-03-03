import requests

OPENAI_API_KEY = '6CY7dleFpEv03vMDcbXdE0E5Ih8i22G5'

Endpoints = {
    'FIM-completions': "https://codestral.mistral.ai/v1/fim/completions",
    'chat-completion': "https://codestral.mistral.ai/v1/chat/completions"
}

def open_agent(endpoint_name):
    url = Endpoints[endpoint_name]
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    response = requests.get(url, headers=headers)
    print(f"Opened agent {endpoint_name} with response status: {response.status_code}")

# Log opening of FIM and chat agents
open_agent('FIM-completions')
open_agent('chat-completion')