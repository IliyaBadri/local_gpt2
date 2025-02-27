# local_gpt2

`local_gpt2` is a lightweight web server and token generation system that uses a GPT-2 model to generate text based on user-provided prompts. The system operates with a combination of HTTP and WebSocket servers for interacting with clients. It generates text in response to prompts and serves a web interface for visual interaction.

**WARNING:** *local_gpt2* lacks security measures and may contain various bugs and is only intended to be used localy.

## Features

- **WebSocket API**: Allows real-time prompt submission and token generation.
- **HTTP Server**: Serves a static web interface for interacting with the system.
- **Model Loading**: Loads the GPT-2 model from Hugging Face for text generation.
- **Recursive Token Generation**: Generates tokens recursively to extend text output.
- **Custom Prompt Pool**: Manages a pool of prompts and token lists for generation.

## Model Details

The system uses the GPT-2 model (`openai-community/gpt2`) from Hugging Face. The model is loaded on startup, and its tokenizer is used to encode and decode tokens during generation.

## Setup

### Prerequisites

- Python 3.7+ **(tested on v3.13.2)**
- `torch` (for model inference)
- `transformers` (for GPT-2 model)
- `tensorflow`
- `websockets` (for WebSocket communication)

To install the necessary dependencies, you can use the following commands:

**1.** Clone the repository to your local device:
```bash
git clone https://github.com/IliyaBadri/local_gpt2.git
```

**2.** Create a python virtual environment **(optional but recomended)**
```bash
python3 -m venv local_gpt2
```

**3.** Change directory into the project folder:
```bash
cd local_gpt2
```

**4.** Install dependencies:

 - **Windows**

	```bash
	.\Scripts\python.exe -m pip install -r requirements.txt
	```

 - **Linux:**

	```bash
	./bin/python3 -m pip install -r requirements.txt
	```

**5.** Run the script:

 - **Windows**

	```bash
	.\Scripts\python.exe main.py
	```

 - **Linux:**

	```bash
	./bin/python3 main.py
	```

By default, the server will start:

-   A WebSocket server on `ws://localhost:8001` for interacting with prompts.
-   An HTTP server on `http://localhost:8000` to serve a web interface.

### Interacting with the System

1.  **Web Interface**: Open `http://localhost:8000` in your browser. The page will allow you to submit a prompt and see the incoming tokens.
2.  **WebSocket Interface**: You can use WebSocket clients to send prompts and receive generated tokens. The WebSocket endpoint by default is `ws://localhost:8001`.

### WebSocket Message Format

To send a prompt via WebSocket, send a JSON message like the following:

```json
{
  "prompt": "Once upon a time"
}

```

The server will respond with generated tokens like:

```json
{
  "prompt_id": "<unique_prompt_id>",
  "tokens": ["Once", "upon", "a", "time", "there", "was", "a", "king"]
}

```

## Contributing

Feel free to fork the project and submit issues or pull requests with improvements. Contributions are always welcome!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

-   The GPT-2 model is provided by [Hugging Face](https://huggingface.co/).
-   The WebSocket server is powered by [WebSockets library](https://websockets.readthedocs.io/).
