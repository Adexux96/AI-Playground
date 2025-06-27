import sys
import os
import pytest
from unittest.mock import patch, MagicMock, ANY

# Ensure service modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) # For backend_shared

from backend_shared.params import LLMParams
import llm_biz # This will be service.llm_biz

# Minimal prompt for LLMParams
MINIMAL_PROMPT = [{"role": "user", "content": "Hello"}]

@pytest.fixture
def mock_dependencies():
    """Fixture to mock all external dependencies of llm_biz.chat"""
    with patch('llm_biz.AutoModelForCausalLM.from_pretrained') as mock_model_from_pretrained, \
         patch('llm_biz.AutoTokenizer.from_pretrained') as mock_tokenizer_from_pretrained, \
         patch('llm_biz.torch.xpu.set_device') as mock_set_device, \
         patch('llm_biz.torch.xpu.empty_cache') as mock_empty_cache, \
         patch('llm_biz.gc.collect') as mock_gc_collect, \
         patch('llm_biz.config') as mock_config, \
         patch('llm_biz.path.abspath') as mock_abspath, \
         patch('llm_biz.path.join') as mock_join, \
         patch('llm_biz.generate') as mock_generate, \
         patch('llm_biz.TextIteratorStreamer') as mock_streamer: # Mocking generate to simplify test

        # Configure mocks
        mock_config.service_model_paths = {'llm': '/dummy/llm_models_path'}
        mock_join.side_effect = lambda *args: "/".join(args) # Simple mock for os.path.join
        mock_abspath.side_effect = lambda p: p # Simple mock for os.path.abspath

        # Mock model and tokenizer instances to have necessary attributes if accessed
        mock_model_instance = MagicMock()
        mock_model_from_pretrained.return_value = mock_model_instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer_instance

        # Mock the generate function to avoid running the actual generation loop
        mock_generate.return_value = iter([]) # Return an empty iterator

        yield {
            "mock_model_from_pretrained": mock_model_from_pretrained,
            "mock_tokenizer_from_pretrained": mock_tokenizer_from_pretrained,
            "mock_set_device": mock_set_device,
            "mock_empty_cache": mock_empty_cache,
            "mock_gc_collect": mock_gc_collect,
            "mock_config": mock_config,
            "mock_abspath": mock_abspath,
            "mock_join": mock_join,
            "mock_generate": mock_generate,
            "mock_streamer": mock_streamer
        }

def test_default_quantization(mock_dependencies):
    """
    Test that chat uses default quantization ('sym_int4') when
    LLMParams.quantization_method is None.
    """
    params = LLMParams(
        prompt=MINIMAL_PROMPT,
        device=0,
        model_repo_id="test-model",
        max_tokens=50,
        quantization_method=None # Explicitly None, though it defaults to "sym_int4" in LLMParams
    )

    # Callbacks can be simple mocks or None if not crucial for this specific test path
    llm_biz.chat(params, None, None, None, None)

    mock_dependencies["mock_model_from_pretrained"].assert_called_once_with(
        "/dummy/llm_models_path/test-model".replace("/", os.path.sep), # Ensure path separator matches OS
        torch_dtype=llm_biz.torch.float16,
        trust_remote_code=True,
        load_in_low_bit="sym_int4" # This is what we are testing
    )

def test_explicit_quantization(mock_dependencies):
    """
    Test that chat uses the explicitly provided quantization method.
    """
    params = LLMParams(
        prompt=MINIMAL_PROMPT,
        device=0,
        model_repo_id="another-model",
        max_tokens=50,
        quantization_method="asym_int8"
    )

    llm_biz.chat(params, None, None, None, None)

    mock_dependencies["mock_model_from_pretrained"].assert_called_once_with(
        "/dummy/llm_models_path/another-model".replace("/", os.path.sep),
        torch_dtype=llm_biz.torch.float16,
        trust_remote_code=True,
        load_in_low_bit="asym_int8" # This is what we are testing
    )

def test_empty_string_quantization(mock_dependencies):
    """
    Test that chat uses default quantization ('sym_int4') when
    LLMParams.quantization_method is an empty string.
    """
    params = LLMParams(
        prompt=MINIMAL_PROMPT,
        device=0,
        model_repo_id="empty-string-model",
        max_tokens=50,
        quantization_method="" # Empty string
    )

    llm_biz.chat(params, None, None, None, None)

    # The behavior for empty string in llm_biz.py:
    # if not quantization_method_to_use: load_in_low_bit = "sym_int4"
    # So, it should default to "sym_int4"
    mock_dependencies["mock_model_from_pretrained"].assert_called_once_with(
        "/dummy/llm_models_path/empty-string-model".replace("/", os.path.sep),
        torch_dtype=llm_biz.torch.float16,
        trust_remote_code=True,
        load_in_low_bit="sym_int4" # This is what we are testing
    )

def test_llm_params_default_quantization_propagates_correctly(mock_dependencies):
    """
    Test that the default quantization_method from LLMParams ('sym_int4')
    is used if not overridden in chat() call.
    """
    # LLMParams defaults quantization_method to "sym_int4" if not provided
    params = LLMParams(
        prompt=MINIMAL_PROMPT,
        device=0,
        model_repo_id="default-param-model",
        max_tokens=50
        # quantization_method is omitted, so it uses default from LLMParams class
    )

    assert params.quantization_method == "sym_int4" # Verify assumption about LLMParams

    llm_biz.chat(params, None, None, None, None)

    mock_dependencies["mock_model_from_pretrained"].assert_called_once_with(
        "/dummy/llm_models_path/default-param-model".replace("/", os.path.sep),
        torch_dtype=llm_biz.torch.float16,
        trust_remote_code=True,
        load_in_low_bit="sym_int4" # This should be the default from LLMParams
    )

def test_model_path_construction(mock_dependencies):
    """Test that the model path is constructed as expected."""
    model_id = "namespace/model-name"
    expected_model_name_in_path = "namespace---model-name" # From model_repo_id.replace("/", "---")

    params = LLMParams(
        prompt=MINIMAL_PROMPT, device=0, model_repo_id=model_id, max_tokens=50
    )
    llm_biz.chat(params, None, None, None, None)

    # Check how os.path.join and os.path.abspath were called
    # mock_join was defined as side_effect: lambda *args: "/".join(args)
    # mock_abspath was side_effect: lambda p: p
    # mock_config.service_model_paths = {'llm': '/dummy/llm_models_path'}

    # Expected path before abspath: /dummy/llm_models_path/namespace---model-name
    # Expected path after abspath (mocked): /dummy/llm_models_path/namespace---model-name

    # Ensure os.path.join was called to create the final model_path
    # The first argument to from_pretrained is model_path
    called_model_path = mock_dependencies["mock_model_from_pretrained"].call_args[0][0]

    # Construct expected path based on mocks
    # Note: The mock_join uses '/' explicitly. If testing on Windows, the actual join might use '\'.
    # The .replace("/", os.path.sep) in assertions handles this for the final path.
    base_llm_path = mock_dependencies["mock_config"].service_model_paths['llm']

    # First call to mock_join: path.join(model_base_path, model_name)
    # model_base_path = /dummy/llm_models_path
    # model_name = namespace---model-name
    # result = /dummy/llm_models_path/namespace---model-name

    # This is what from_pretrained should receive:
    expected_path_to_from_pretrained = os.path.join(base_llm_path, expected_model_name_in_path)

    assert called_model_path == expected_path_to_from_pretrained.replace("/", os.path.sep)

    # Verify that os.path.join was called with the correct components
    # The path.join in llm_biz.py is: path.abspath(path.join(model_base_path, model_name))
    # So, mock_join is called with (model_base_path, model_name)
    # And mock_abspath is called with its result.

    # Check call to path.join
    mock_dependencies["mock_join"].assert_any_call(base_llm_path, expected_model_name_in_path)

    # Check call to path.abspath
    # The argument to abspath is the result of the join call.
    # Our mock_join returns "base_llm_path/expected_model_name_in_path"
    mock_dependencies["mock_abspath"].assert_any_call(f"{base_llm_path}/{expected_model_name_in_path}")

    # Check if the model loading part is only entered if _model is None or _last_repo_id changes
    # This requires running chat twice.
    llm_biz._model = MagicMock() # Simulate model already loaded
    llm_biz._last_repo_id = model_id # Simulate it's the same model

    # Reset call count for from_pretrained before second call
    mock_dependencies["mock_model_from_pretrained"].reset_mock()

    llm_biz.chat(params, None, None, None, None)
    mock_dependencies["mock_model_from_pretrained"].assert_not_called() # Should not be called as model is "cached"

    # Reset _model and _last_repo_id for other tests if they rely on initial state
    llm_biz._model = None
    llm_biz._last_repo_id = None
