from flask import Flask, request, jsonify
from llama_cpp import Llama

# Load BioMistral-7B model with Metal (GPU) support
llm = Llama(
    model_path="./models/ggml-model-Q4_K_M.gguf",  # adjust path if needed
    n_ctx=2048,
    n_threads=8,      # you have 8 logical cores on M2
    n_gpu_layers=100, # enables GPU acceleration (use 100+ to fully offload)
    verbose=True      # optional: prints Metal logs to terminal
)

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400

    prompt = f"[INST] {question} [/INST]"
    response = llm(prompt, max_tokens=300, stop=["</s>"])
    answer = response["choices"][0]["text"].strip()

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5000)
