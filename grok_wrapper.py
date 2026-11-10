import requests  # send messages across the web.
import os
from dotenv import load_dotenv

load_dotenv()


# component 2: the client
class GroqClient:
    def __init__(self, api_key: str | None = None, base_url: str = "https://api.groq.com/openai/v1/chat/completions",
                 default_model: str = "qwen/qwen3.8-27b"):  # Using standard fallback

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.base_url = base_url
        self.available_models = self.list_available_models()
        self.default_model = self._auto_select_model(default_model)



    # DYNAMIC METHOD
    def list_available_models(self) -> list:
        """Fetches all model IDs currently available to this API key."""
        # Groq's model endpoint drops the /chat/completions part
        models_url = self.base_url.replace("/chat/completions", "/models")
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.get(models_url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not fetch models dynamically ({e})")
            return []

    def _auto_select_model(self, preferred: str) -> str:
        """Selects the preferred model if available, otherwise safely falls back to another text model."""
        if not self.available_models:
            return preferred  # Fallback to preferred if the API fetch failed

        if preferred in self.available_models:
            return preferred

        # Filter out non-chat models (like audio transcription or prompt guards)
        valid_chat_models = [
            m for m in self.available_models
            if "whisper" not in m.lower() and "guard" not in m.lower()
        ]

        if valid_chat_models:
            fallback = valid_chat_models[0]
            print(f"Preferred model '{preferred}' unavailable. Auto-switching to: {fallback}")
            return fallback

        # Absolute fallback if filtering leaves nothing
        return self.available_models[0]
    # -------------------------------------

    def chat(self, messages: list, model: str | None = None, temperature: float = 0.0) -> str:
        """component 3: Authorization Header (the passports)"""

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model or self.default_model, "messages": messages, "temperature": temperature}

        # send the http POST request to groqs web server
        response = requests.post(self.base_url, json=payload, headers=headers, timeout=10)

        print("http status code: ", response.status_code, "groq error body: ", response.text)

        # check for HTTP errors
        response.raise_for_status()
        # convert groqs json response into a python dict
        data = response.json()

        return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    client = GroqClient()
    key_prefix = client.api_key[:4] if client.api_key else "no key"
    print(f"using groq key with prefix {key_prefix}\n")

    # --- CALLING THE NEW METHOD ---
    print("Fetching available models for your API key...")
    try:
        available_models = client.list_available_models()
        print("Models you can use:")
        for m in available_models:
            print(f" - {m}")
    except Exception as e:
        print("Failed to fetch models:", e)
    # ------------------------------

    test_messages = [{"role": "user", "content": "Hello Groq! confirm that you are working"}]

    print("\nsending request to groq.... ")

    # We will let it use the default_model set in __init__, or you can override it here:
    # reply = client.chat(test_messages, model="llama3-8b-8192")
    reply = client.chat(test_messages)

    print("\n---Groq Response---\n", reply, "\n---End---\n")