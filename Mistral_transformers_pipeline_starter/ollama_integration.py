import subprocess

def pull_llama_model():
    """
    Function to run the 'ollama pull llama3.1' command.
    """
    try:
        result = subprocess.run(['ollama', 'pull', 'llama3.1'], check=True, capture_output=True, text=True)
        print("Model pulled successfully:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error pulling model:")
        print(e.stderr)

# Example usage
if __name__ == "__main__":
    pull_llama_model()
