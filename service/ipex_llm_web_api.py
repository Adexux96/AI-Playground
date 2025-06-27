from apiflask import APIFlask
from flask import jsonify, request, Response, stream_with_context, current_app
import json
import logging # Added for callbacks

# Assuming llm_adapter.py will be in the same 'service' directory or Python path
# If llm_adapter is not found, it implies LlamaCPP/llama_adapter.py needs to be copied to service/llm_adapter.py first
from llm_adapter import LLM_SSE_Adapter
from llm_biz import chat as ipex_llm_chat, stop_generate as ipex_llm_stop_generate, dispose as ipex_llm_dispose
from backend_shared.params import LLMParams
from ipex_embedding import IpexEmbeddingModel # Make sure ipex_embedding.py is in service directory
from backend_shared import utils

# For SSE event generation
import queue
import threading


app = APIFlask(__name__)
# Placeholder for repo_id, ideally this comes from config or request
embedding_model = IpexEmbeddingModel.get_instance(repo_id=None)

# Setup basic logging
logging.basicConfig(level=logging.INFO)


@app.get("/health")
def health():
    return jsonify({"code": 0, "message": "success"})


@app.post("/api/llm/chat")
def llm_chat():
    params_json = request.get_json()
    # params_json.pop("print_metrics", None) # print_metrics is a valid field in LLMParams now
    llm_params = LLMParams(**params_json)

    # This will be replaced with direct call to ipex_llm_chat and custom SSE handling
    # sse_invoker = LLM_SSE_Adapter(llm_backend) # llm_backend no longer exists
    # it = sse_invoker.text_conversation(llm_params)
    # return Response(stream_with_context(it), content_type="text/event-stream")

    # Placeholder for new implementation
    def generate_sse_events():
        # Simple callbacks
        def load_model_callback(status: str):
            logging.info(f"IPEX LLM Load Model Status: {status}")
            sse_adapter.put_msg({"type": "load_model_status", "status": status})

        def text_out_callback(text_chunk: str, status_code: int):
            # status_code 1 for text, 2 for RAG source (not used here directly by ipex_llm_chat)
            if status_code == 1:
                sse_adapter.put_text(text_chunk)

        def metrics_callback(metrics: dict):
            logging.info(f"IPEX LLM Metrics: {metrics}")
            sse_adapter.put_msg({"type": "metrics", "data": metrics})

        def error_callback(err: Exception):
            logging.error(f"IPEX LLM Error: {str(err)}")
            sse_adapter.put_error(str(err))
            # Ensure queue is ended if an error occurs during generation
            sse_adapter.end()


        # LLM_SSE_Adapter is used here to format SSE messages
        # It was originally designed for a different flow, but its put_msg, put_text, put_error, end methods are useful
        sse_adapter = LLM_SSE_Adapter(None) # Pass None as backend, not used by put_xxx methods

        # We need a way for ipex_llm_chat (running in a thread) to send data to this generator
        # Using a queue is a common pattern for this.
        # However, ipex_llm_chat itself uses TextIteratorStreamer which is already a generator.
        # The challenge is that ipex_llm_chat is blocking and designed to be run in a thread if used with its own streamer.
        # The original LlamaCPP implementation's sse_invoker.text_conversation likely handles this threading.

        # For IPEX, ipex_llm_chat itself doesn't directly return a generator of SSE events.
        # It takes callbacks. One of these callbacks (text_out_callback) will receive text chunks.
        # This text_out_callback needs to push data into our SSE stream.

        # Let's try to run ipex_llm_chat in a separate thread,
        # and have its callbacks put data into the sse_adapter's queue.

        chat_thread = threading.Thread(
            target=ipex_llm_chat,
            args=(llm_params, load_model_callback, text_out_callback, metrics_callback, error_callback)
        )
        chat_thread.start()

        try:
            while True:
                try:
                    event = sse_adapter.event_queue.get(timeout=1) # Check for events periodically
                    if event is None: # End of stream signal from sse_adapter.end()
                        break
                    yield event
                except queue.Empty:
                    # If chat_thread has finished or died, we might not get an explicit None
                    if not chat_thread.is_alive() and sse_adapter.event_queue.empty():
                        break
                    continue # Continue waiting if thread is alive
        finally:
            # Ensure thread is joined
            if chat_thread.is_alive():
                # If the generator is stopped from client side, we need to signal ipex_llm_stop_generate
                # This is a bit tricky because stop_generate is global.
                # Consider if ipex_llm_chat needs a per-instance stop mechanism. For now, use global.
                ipex_llm_stop_generate()
                chat_thread.join(timeout=5) # Wait for thread to finish
            # Final check on queue in case thread put something before exiting
            while not sse_adapter.event_queue.empty():
                event = sse_adapter.event_queue.get_nowait()
                if event is None:
                    break
                yield event


    return Response(stream_with_context(generate_sse_events()), content_type="text/event-stream")


@app.post("/api/free")
def free():
    # llm_backend.unload_model() # Original
    ipex_llm_dispose()
    return jsonify({"code": 0, "message": "success"})


@app.get("/api/llm/stopGenerate")
def stop_llm_generate():
    # llm_backend.stop_generate = True # Original
    ipex_llm_stop_generate()
    return jsonify({"code": 0, "message": "success"})


@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    data = request.json

    encoding_format = data.get('encoding_format', 'float')
    input_data = data.get('input', None)

    if not input_data:
        return jsonify({"error": "Input text is required"}), 400

    if isinstance(input_data, str):
        input_texts = [input_data]
    elif isinstance(input_data, list):
        input_texts = input_data
    else:
        return jsonify({"error": "Input should be a string or list of strings"}), 400

    # Assuming IpexEmbeddingModel.get_instance() correctly sets up the model for embedding
    # If repo_id was None at init, embed_documents might need to load a default model or use one from params
    # For now, let's assume get_instance or a subsequent call configures it.
    # The user of this API might need to pass model info if not using a default.
    # This part might need refinement based on IpexEmbeddingModel's actual behavior.
    if 'model' in data and data['model']:
        # Potentially re-initialize or set model for embedding_model if API allows model selection per call
        # current_app.logger.info(f"Embedding requested for model: {data['model']}")
        # For now, we assume embedding_model is a singleton pre-configured or configured by first use.
        # If IpexEmbeddingModel is truly a singleton that gets configured by first call to embed_documents,
        # we might need to pass 'repo_id' from the request here.
        # For now, let's assume it uses a default or the one set at startup.
        pass


    embeddings_result = embedding_model.embed_documents(input_texts)

    # Get the model path from the instance, assuming it's populated after model load.
    # This might be None if model loads lazily and embed_documents hasn't been called with a repo_id yet.
    model_name_or_path = embedding_model.embedding_model_path if hasattr(embedding_model, 'embedding_model_path') else "ipex-embedding-default"


    response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": utils.convert_embedding(emb, encoding_format),
                "index": idx
            } for idx, emb in enumerate(embeddings_result)
        ],
        "model": model_name_or_path, # Use the actual model path/name
        "usage": {
            # This is a rough token count, actual tokenization depends on the model
            "prompt_tokens": sum(len(text.split()) for text in input_texts), # Placeholder
            "total_tokens": sum(len(text.split()) for text in input_texts)  # Placeholder
        }
    }

    return jsonify(response)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IPEX LLM Web Service") # Updated description
    parser.add_argument("--port", type=int, default=59998, help="Service listen port") # Updated port
    args = parser.parse_args()
    app.run(host="127.0.0.1", port=args.port, use_reloader=False)
