from flask import Flask, Response, send_from_directory, request, jsonify
import json
import logging
import os
import time
import threading
import torch
from typing import Callable, List, Optional

def print_exception(exception: Exception):
    raise exception


def print_tokens(tokens: list[str]):
    for token in tokens:
        print(token, end='', flush=True)

class TokenEventSystem():
    subscriber_type = Callable[[List[str]], None]
    token_queue: List[str] = []
    subscribers: List[subscriber_type] = []

    @staticmethod
    def subscribe(subscriber: subscriber_type):
        TokenEventSystem.subscribers.append(subscriber) 

    @staticmethod
    def add_tokens(tokens: List[str]):
        TokenEventSystem.token_queue.extend(tokens)

    @staticmethod
    def create_worker():
        def worker():
            while True:
                if(len(TokenEventSystem.token_queue) > 0):
                    for subscriber in TokenEventSystem.subscribers:
                        subscriber(TokenEventSystem.token_queue)
                    TokenEventSystem.token_queue.clear()
                else:
                    time.sleep(0.1)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

class Model():
    def __init__(self):
        self.model_name = "openai-community/gpt2"
        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

class GenerationSession():
    def __init__(self, model: Model, prompt: str):
        self.model = model
        self.prompt = prompt
        self.prompt_tensor: torch.Tensor = self.model.tokenizer.encode(prompt, return_tensors="pt")
        self.output_tensor = self.prompt_tensor
    
    def get_attention_mask(self) -> torch.Tensor:
        attention_mask_array = []
        for sequence in self.output_tensor:
            mask = []
            for token in sequence:
                if token == self.model.tokenizer.eos_token_id:
                    mask.append(0)
                else:
                    mask.append(1)

            attention_mask_array.append(mask)

        return torch.tensor(attention_mask_array)
    
    def append_delta(self, generated_tensor: torch.Tensor, token_count: int) -> torch.Tensor:
        generated_tokens = []
        for i in range(token_count):
            generated_tokens.append(generated_tensor[0, -(token_count - i)]) 
        delta_tensor = torch.tensor([generated_tokens]) 
        self.output_tensor = torch.cat((self.output_tensor, delta_tensor), dim=1)
        return delta_tensor
    
    def generate_tokens(self, token_count: int) -> Optional[List[str]]:
        attention_mask = self.get_attention_mask()

        generated_tensor = self.model.model.generate(
            self.output_tensor,
            max_length = self.output_tensor.shape[1] + token_count,
            num_return_sequences = 1, 
            no_repeat_ngram_size = 2,
            top_p = 0.92,
            temperature=0.82,
            do_sample=True,
            pad_token_id=self.model.tokenizer.eos_token_id,
            attention_mask=attention_mask
        )

        delta_tensor = self.append_delta(generated_tensor, token_count)

        tokens: list[str] = []

        for i in range(len(delta_tensor[0])):
            tokens.append(self.model.tokenizer.decode(delta_tensor[0, i], skip_special_tokens=True))

        return tokens
    
    def generate_recursively(self, rounds: int, tokens_per_round: int):
        try:
            for _ in range(rounds):
                tokens = self.generate_tokens(tokens_per_round)
                if(tokens == None):
                    break
            TokenEventSystem.add_tokens(tokens)
        except Exception as exception:
            print_exception(exception)

def web_server_worker(model: Model, port=8000):
    static_path = os.path.abspath("./static")
    app = Flask(__name__, static_folder=static_path)
    logging.getLogger('werkzeug').disabled = True

    @app.route("/<path:filename>")
    def serve_static(filename):
        return send_from_directory(static_path, filename)
    
    @app.route("/generate", methods=["POST"])
    def generate():
        if request.method == 'POST':
            prompt = request.json["prompt"]
            if not prompt or prompt == None or prompt == "":
                return jsonify({"successful": False})
            
            try:
                session = GenerationSession(model, prompt)

                def generate_async():
                    session.generate_recursively(40, 5)  
            
                generation_thread = threading.Thread(target=generate_async, daemon=True)
                generation_thread.start()

                return jsonify({"successful": True})
            except Exception as exception:
                print_exception(exception)
                return jsonify({"successful": False})

    @app.route("/updates")
    def sse():
        def event_stream():
            tokens: List[str] = []
            TokenEventSystem.subscribe(tokens.extend)
            while True:
                if len(tokens) > 0:
                    tokens_to_send = []
                    tokens_to_send.extend(tokens)
                    tokens.clear()
                    yield f"data: {json.dumps({'tokens': tokens_to_send})}\n\n"
                else:
                    time.sleep(0.1)

        return Response(event_stream(), content_type="text/event-stream")
    
    print(f"[+] HTTP server is up ( http://localhost:{port}/index.html )")
    app.run("", port, threaded=True)

if __name__ == "__main__":
    models_path = os.path.abspath("./models")
    os.makedirs(models_path, exist_ok=True)

    os.environ["HF_HOME"] = models_path
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    model = Model()
    TokenEventSystem.create_worker()
    TokenEventSystem.subscribe(print_tokens)

    web_server_worker(model, 8000)

    