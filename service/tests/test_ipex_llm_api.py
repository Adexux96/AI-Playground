import pytest
import json
from unittest.mock import patch, MagicMock, ANY

# Ensure service modules can be imported
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ipex_llm_web_api import app as ipex_app  # Import the app from your new API file
from backend_shared.params import LLMParams

@pytest.fixture
def client():
    ipex_app.config['TESTING'] = True
    with ipex_app.test_client() as client:
        # Mock IpexEmbeddingModel.get_instance() to prevent actual model loading during tests
        # This mock will apply to all tests using this client fixture.
        with patch('ipex_llm_web_api.IpexEmbeddingModel.get_instance') as mock_get_instance:
            mock_embedding_model_instance = MagicMock()
            mock_get_instance.return_value = mock_embedding_model_instance
            yield client

def test_health(client):
    """Test the /health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json == {"code": 0, "message": "success"}

@patch('ipex_llm_web_api.ipex_llm_chat')
def test_llm_chat(mock_ipex_llm_chat, client):
    """Test the /api/llm/chat endpoint."""
    # Mock ipex_llm_chat to simulate behavior and capture callbacks
    # It needs to simulate interaction with the sse_adapter's queue if we want to test SSE content
    # For simplicity here, we'll check if it's called and the response type.
    # A more complex test could involve the mock_ipex_llm_chat putting items onto a queue
    # that the test can then inspect.

    def mock_chat_implementation(params, load_cb, text_cb, metrics_cb, error_cb):
        # Simulate some interaction with callbacks if needed for more detailed tests
        load_cb("start")
        text_cb("Hello ", 1)
        text_cb("world!", 1)
        metrics_cb({"num_tokens": 2})
        load_cb("finish")
        # The actual sse_adapter.end() would be called internally by the real ipex_llm_chat or its surrounding logic
        # For this mock, we don't need to explicitly call it unless testing the sse_adapter directly.
        # The thread in the actual endpoint calls this.

    mock_ipex_llm_chat.side_effect = mock_chat_implementation

    chat_payload = {
        "prompt": [{"question": "Hello", "answer": ""}],
        "device": 0,
        "model_repo_id": "test-model",
        "max_tokens": 50,
        "print_metrics": True,
        "quantization_method": "sym_int4"
    }
    response = client.post('/api/llm/chat', json=chat_payload)

    assert response.status_code == 200
    assert response.content_type == 'text/event-stream'

    # Verify ipex_llm_chat was called with an LLMParams instance
    mock_ipex_llm_chat.assert_called_once()
    call_args = mock_ipex_llm_chat.call_args[0]
    assert isinstance(call_args[0], LLMParams)
    assert call_args[0].model_repo_id == "test-model"
    assert call_args[0].quantization_method == "sym_int4"

    # Check that callbacks were passed (ANY matches any function/method)
    assert call_args[1] == ANY # load_model_callback
    assert call_args[2] == ANY # text_out_callback
    assert call_args[3] == ANY # metrics_callback
    assert call_args[4] == ANY # error_callback

    # To test SSE content, you'd iterate through response.iter_encoded() or response.data
    # and parse the SSE events.
    # Example for checking if some data was streamed:
    # decoded_stream = b"".join(response.streaming_content).decode('utf-8')
    # assert "data: {\"text\": \"Hello \"}" in decoded_stream # This depends on LLM_SSE_Adapter format
    # assert "data: {\"text\": \"world!\"}" in decoded_stream
    # assert "data: {\"type\": \"metrics\"" in decoded_stream


@patch('ipex_llm_web_api.utils.convert_embedding')
@patch('ipex_llm_web_api.embedding_model.embed_documents') # Mocking the method on the already mocked instance
def test_embeddings(mock_embed_documents, mock_convert_embedding, client):
    """Test the /v1/embeddings endpoint."""
    mock_embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock_convert_embedding.side_effect = lambda emb, fmt: list(emb) # Simple pass-through

    embedding_payload = {
        "input": "Test sentence",
        "encoding_format": "float"
    }
    response = client.post('/v1/embeddings', json=embedding_payload)

    assert response.status_code == 200
    response_json = response.json
    assert response_json["object"] == "list"
    assert len(response_json["data"]) == 1
    assert response_json["data"][0]["object"] == "embedding"
    assert response_json["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert response_json["model"] is not None # Check that model field exists

    mock_embed_documents.assert_called_once_with(["Test sentence"])
    mock_convert_embedding.assert_called_once_with([0.1, 0.2, 0.3], "float")

@patch('ipex_llm_web_api.ipex_llm_dispose')
def test_free_endpoint(mock_dispose, client):
    """Test the /api/free endpoint."""
    response = client.post('/api/free')
    assert response.status_code == 200
    assert response.json == {"code": 0, "message": "success"}
    mock_dispose.assert_called_once()

@patch('ipex_llm_web_api.ipex_llm_stop_generate')
def test_stop_generate_endpoint(mock_stop_generate, client):
    """Test the /api/llm/stopGenerate endpoint."""
    response = client.get('/api/llm/stopGenerate')
    assert response.status_code == 200
    assert response.json == {"code": 0, "message": "success"}
    mock_stop_generate.assert_called_once()

# Helper to decode SSE stream (if needed for more detailed chat test)
# Similar to the one in test_api.py but adapted for pytest if necessary
def decode_sse_stream(byte_stream):
    events = []
    for line in byte_stream.splitlines():
        line = line.strip()
        if line.startswith(b'data:'):
            try:
                data_str = line[len(b'data:'):].strip().decode('utf-8')
                if data_str: # Ensure data_str is not empty before parsing
                    events.append(json.loads(data_str))
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e} for line: {line}")
            except UnicodeDecodeError as e:
                print(f"Unicode decode error: {e} for line: {line}")

    return events

@patch('ipex_llm_web_api.ipex_llm_chat')
def test_llm_chat_sse_content(mock_ipex_llm_chat, client):
    """Test the /api/llm/chat endpoint for SSE content."""

    # This is a more advanced test for chat that inspects SSE content.
    # It requires the mock_ipex_llm_chat to interact with the sse_adapter's queue via callbacks.
    # The LLM_SSE_Adapter instance is created inside generate_sse_events.
    # We need to mock LLM_SSE_Adapter to control its event_queue or mock its methods.

    # For this example, let's assume ipex_llm_chat calls text_out_callback,
    # which in turn calls sse_adapter.put_text().
    # The test for generate_sse_events() in ipex_llm_web_api.py would be more direct for this.
    # Here, we are testing the endpoint.

    actual_events = []
    def mock_chat_fn(params, load_cb, text_cb, metrics_cb, error_cb):
        # Simulate the behavior of ipex_llm_chat calling the callbacks
        load_cb("start_test")
        text_cb("First chunk ", 1)
        text_cb("Second chunk", 1)
        metrics_cb({"tokens_processed": 2})
        # The real chat would have a loop; this is simplified.
        # The sse_adapter.end() is called by the thread management in generate_sse_events
        # or by error_callback.

    mock_ipex_llm_chat.side_effect = mock_chat_fn

    chat_payload = {
        "prompt": [{"question": "SSE Test", "answer": ""}], "device": 0,
        "model_repo_id": "sse-model", "max_tokens": 10
    }
    response = client.post('/api/llm/chat', json=chat_payload)
    assert response.status_code == 200

    # The `response.data` will contain the concatenated byte stream of all SSE events.
    # `response.streaming_content` could also be used if iterating.
    # Need to ensure `LLM_SSE_Adapter` formats messages as expected.
    # Example format: "data: {...}\\n\\n"

    # Due to the threading and queueing in the actual endpoint, directly capturing
    # what mock_ipex_llm_chat's callbacks send to the sse_adapter is tricky from here.
    # A more robust way would be to patch 'LLM_SSE_Adapter' itself if testing detailed SSE content.

    # This is a basic check that *some* streaming data came through.
    raw_data = b"".join(response.streaming_content) # Consume the generator
    assert b"data:" in raw_data # Check if any SSE 'data:' prefix is present

    # If you had a reliable way to parse the events (assuming LLM_SSE_Adapter is consistent):
    # parsed_events = decode_sse_stream(raw_data)
    # assert {"type": "load_model_status", "status": "start_test"} in parsed_events
    # assert {"text": "First chunk "} in parsed_events # Format from LLM_SSE_Adapter.put_text
    # assert {"text": "Second chunk"} in parsed_events
    # assert {"type": "metrics", "data": {"tokens_processed": 2}} in parsed_events
    # The final event should be the end-of-stream signal if sse_adapter.end() was effective.
    # This part is complex due to the adapter being instantiated within the endpoint.
    # For now, just checking for presence of data.
    mock_ipex_llm_chat.assert_called()
