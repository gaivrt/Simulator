# tools/llm_utils.py
# -*- coding: utf-8 -*-
"""
Centralized LLM interaction utilities for the project.
This module handles LLM client initialization, API calls, response cleaning,
and connectivity checks for different providers (OpenAI, Gemini).
"""
import json
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

# --- LLM Specific Imports ---
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai library not found. OpenAI provider will not work.")

try:
     import google.generativeai as genai
     GEMINI_AVAILABLE = True
except ImportError:
     GEMINI_AVAILABLE = False
     print("Warning: google-generativeai library not found. Gemini provider will not work.")
# ---------------------------

load_dotenv()

# --- LLM Config ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4-turbo") # Used if no specific model passed to call_llm_api
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_ARCHETYPE_MODEL", "gemini-1.5-flash") # Used if no specific model passed to call_llm_api

# --- DeepInfra Config (for Embeddings) ---
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_BASE_URL = os.getenv("DEEPINFRA_BASE_URL")
DEEPINFRA_EMBEDDING_MODEL = os.getenv("DEEPINFRA_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B") # Default if not in .env
# -----------------------------------------

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 1.2))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", 0.95))
OPENAI_FREQUENCY_PENALTY = float(os.getenv("OPENAI_FREQUENCY_PENALTY", 0.3))
OPENAI_PRESENCE_PENALTY = float(os.getenv("OPENAI_PRESENCE_PENALTY", 0.3))


# ---------------------------------

# --- LLM Client Initialization ---
openai_client = None
if LLM_PROVIDER == "openai":
     if OPENAI_AVAILABLE and OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None)
     else:
         print("Error: OpenAI selected as LLM_PROVIDER, but library or API Key/Base URL is missing from .env.")
elif LLM_PROVIDER == "gemini":
     if GEMINI_AVAILABLE and GEMINI_API_KEY:
         genai.configure(api_key=GEMINI_API_KEY)
     else:
          print("Error: Gemini selected as LLM_PROVIDER, but library or API Key is missing from .env.")
else:
    print(f"Warning: Unsupported LLM_PROVIDER specified in .env: '{LLM_PROVIDER}'. LLM calls will likely fail.")
# ---------------------------------

# --- DeepInfra Client Initialization (for Embeddings) ---
deepinfra_client = None
if OPENAI_AVAILABLE and DEEPINFRA_API_KEY and DEEPINFRA_BASE_URL:
    try:
        deepinfra_client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url=DEEPINFRA_BASE_URL)
        print("DeepInfra client for embeddings initialized successfully.")
    except Exception as e:
        print(f"Error initializing DeepInfra client: {e}")
else:
    if not OPENAI_AVAILABLE:
        print("Warning: OpenAI library not found. DeepInfra embeddings (via OpenAI compatible API) will not work.")
    if not DEEPINFRA_API_KEY or not DEEPINFRA_BASE_URL:
        print("Warning: DEEPINFRA_API_KEY or DEEPINFRA_BASE_URL missing from .env. DeepInfra embeddings will not work.")
# --------------------------------------------------------

def _clean_json_string(s: str) -> str:
    """
    From a string that might contain a Markdown JSON code block, extract the pure JSON part.
    Handles ```json ... ``` and falls back to stripping whitespace if no block is found.
    """
    if not isinstance(s, str):
        return ""
    # Regex to find content within ```json ... ```
    match = re.search(r"```json\s*([\s\S]*?)\s*```", s)
    if match:
        return match.group(1).strip()
    # Fallback: if no markdown block, just strip whitespace and newlines
    # This might help with minor formatting issues or if the LLM forgets the markdown block.
    return s.strip()

def call_llm_api(system_prompt_formatted: str, model_override: str = None, expect_json: bool = True):
    """
    Calls the configured LLM API (OpenAI or Gemini) with the provided system prompt.

    Args:
        system_prompt_formatted (str): The complete prompt to send to the LLM.
        model_override (str, optional): Specific model name to use, overriding the default.
        expect_json (bool, optional): If True, attempts to parse the response as JSON
                                     and uses response_format/mime_type accordingly.
                                     Defaults to True.

    Returns:
        dict or str or None: Parsed JSON object if expect_json is True and successful,
                             raw string if expect_json is False and successful,
                             or None if an error occurs.
    """
    raw_output = "N/A"
    try:
        if LLM_PROVIDER == "openai" and openai_client:
            model_to_use = model_override or OPENAI_DEFAULT_MODEL
            common_params = {
                "model": model_to_use,
                "messages": [{"role": "system", "content": system_prompt_formatted}],
                "temperature": LLM_TEMPERATURE,
                "top_p": LLM_TOP_P,
                "frequency_penalty": OPENAI_FREQUENCY_PENALTY,
                "presence_penalty": OPENAI_PRESENCE_PENALTY
            }
            if expect_json:
                common_params["response_format"] = {"type": "json_object"}
            
            completion = openai_client.chat.completions.create(**common_params)
            raw_output = completion.choices[0].message.content
            
            if expect_json:
                cleaned_output = _clean_json_string(raw_output)
                return json.loads(cleaned_output)
            return raw_output # Return raw string if not expecting JSON

        elif LLM_PROVIDER == "gemini" and GEMINI_AVAILABLE:
            model_to_use = model_override or GEMINI_DEFAULT_MODEL
            model_instance = genai.GenerativeModel(model_to_use)
            
            generation_config_params = {
                "temperature": LLM_TEMPERATURE,
                "top_p": LLM_TOP_P
            }
            if expect_json:
                generation_config_params["response_mime_type"] = "application/json"

            response = model_instance.generate_content(
                system_prompt_formatted,
                generation_config=genai.types.GenerationConfig(**generation_config_params)
            )
            if response.candidates and response.candidates[0].content.parts:
                raw_output = response.candidates[0].content.parts[0].text
                if expect_json:
                    cleaned_output = _clean_json_string(raw_output)
                    return json.loads(cleaned_output)
                return raw_output # Return raw string if not expecting JSON
            else:
                print(f"Gemini API Error: No valid candidates or content parts in response. Prompt: {system_prompt_formatted[:200]}...")
                return None
        else:
            print(f"LLM provider '{LLM_PROVIDER}' not supported or client not initialized.")
            return None

    except json.JSONDecodeError as e:
        print(f"LLM API Error: Failed to decode JSON. Error: {e}")
        print(f"----- LLM Raw Output for prompt ({system_prompt_formatted[:100]}...) Start -----\n{raw_output}\n----- LLM Raw Output End -----")
        return None
    except Exception as e:
        print(f"LLM API Error ({type(e).__name__}): {e}. Prompt: {system_prompt_formatted[:200]}...")
        # import traceback
        # traceback.print_exc() # Uncomment for more detailed traceback during debugging
        return None

def get_embeddings_from_api(texts: list[str], model: str = None) -> list[list[float]] | None:
    """
    Generates embeddings for a list of texts using the DeepInfra API (OpenAI compatible).

    Args:
        texts (list[str]): A list of texts to embed.
        model (str, optional): The embedding model to use. Defaults to DEEPINFRA_EMBEDDING_MODEL from .env.

    Returns:
        list[list[float]] | None: A list of embedding vectors, or None if an error occurs.
    """
    if not deepinfra_client:
        print("Error: DeepInfra client not initialized. Cannot get embeddings.")
        return None
    if not texts:
        print("Warning: No texts provided to get_embeddings_from_api.")
        return []

    actual_model = model or DEEPINFRA_EMBEDDING_MODEL
    if not actual_model:
        print("Error: No embedding model specified for DeepInfra (check .env DEEPINFRA_EMBEDDING_MODEL or model argument).")
        return None
    
    try:
        response = deepinfra_client.embeddings.create(
            input=texts,
            model=actual_model,
            encoding_format='float'  # Explicitly set encoding_format to float
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings
    except Exception as e:
        print(f"DeepInfra Embedding API Error ({type(e).__name__}): {e}. Model: {actual_model}, Texts count: {len(texts)}")
        # Log first text for debugging if needed, carefully considering privacy
        # if texts: print(f"First text snippet: {texts[0][:100]}...")
        return None

def check_llm_connectivity(verbose: bool = True) -> bool:
    """Checks connectivity to the configured LLM provider."""
    if verbose:
        print("\n--- Checking LLM Connectivity ---")
    
    if LLM_PROVIDER == "openai":
        if openai_client:
            try:
                openai_client.models.list(limit=1) # Reduced limit for faster check
                if verbose: print("OpenAI connection successful.")
                return True
            except Exception as e:
                if verbose: print(f"Error: OpenAI connection failed. Details: {e}")
                return False
        else:
            if verbose: print("OpenAI client not initialized (check API key and library).")
            return False
    elif LLM_PROVIDER == "gemini":
        if GEMINI_AVAILABLE and GEMINI_API_KEY and (GEMINI_DEFAULT_MODEL or os.getenv("GEMINI_ARCHETYPE_MODEL")):
            try:
                model_name_to_check = GEMINI_DEFAULT_MODEL or os.getenv("GEMINI_ARCHETYPE_MODEL")
                model = genai.GenerativeModel(model_name_to_check)
                model.count_tokens("test connectivity") # Simple way to test
                if verbose: print("Gemini connection successful.")
                return True
            except Exception as e:
                if verbose: print(f"Error: Gemini connection failed. Details: {e}")
                return False
        else:
            if verbose: print("Gemini client not initialized (check API key, library, and model name).")
            return False
    else:
        if verbose: print(f"LLM_PROVIDER '{LLM_PROVIDER}' is not supported for connectivity check or not configured.")
        return False

# It's good practice to also add a connectivity check for the embedding client if possible,
# or at least a status based on initialization.
def check_embedding_connectivity(verbose: bool = True) -> bool:
    if verbose:
        print("\n--- Checking Embedding Client (DeepInfra) Connectivity ---")
    if deepinfra_client:
        try:
            # Attempt a very small, cheap operation if one exists, e.g., list models if allowed and cheap.
            # For DeepInfra via OpenAI client, a simple check might be to see if the client exists.
            # A more robust check would be a minimal API call, but that might incur costs.
            # For now, we'll rely on successful initialization.
            # A test call could be: get_embeddings_from_api(["test"], model=DEEPINFRA_EMBEDDING_MODEL)
            # but that would print errors if it fails during a general check.
            # So, for now, if client exists, we assume basic config is ok.
            # A true connectivity test for embeddings might be done by trying to embed a test string.
            print("DeepInfra client is initialized. Further connectivity depends on API key and endpoint validity.")
            # Example actual test (can be verbose/costly for a simple check):
            # test_embedding = get_embeddings_from_api(["test connectivity"], model=DEEPINFRA_EMBEDDING_MODEL)
            # if test_embedding:
            #     print("Successfully fetched test embedding from DeepInfra.")
            #     return True
            # else:
            #     print("Failed to fetch test embedding from DeepInfra.")
            #     return False
            return True # Assuming initialization means configured.
        except Exception as e:
            if verbose: print(f"Error with DeepInfra client: {e}")
            return False
    else:
        if verbose: print("DeepInfra client for embeddings is not initialized (check .env for DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL and ensure 'openai' library is installed).")
        return False

if __name__ == '__main__':
    print("Running llm_utils.py directly for testing connectivity...")
    # Basic connectivity test
    llm_ok = check_llm_connectivity()
    if llm_ok:
        print("LLM connection appears to be working.")
    else:
        print("LLM connection test failed. Please check your .env configuration and LLM provider status.")
    
    embedding_ok = check_embedding_connectivity()
    if embedding_ok:
        print("Embedding client (DeepInfra) appears to be configured.")
        # You could add a test call here if desired:
        # print("\nAttempting a test embedding call to DeepInfra...")
        # test_embeds = get_embeddings_from_api(["hello world"], model=DEEPINFRA_EMBEDDING_MODEL)
        # if test_embeds:
        #     print(f"Test embedding call successful. Embedding vector dim: {len(test_embeds[0])}")
        # else:
        #     print("Test embedding call failed.")
    else:
        print("Embedding client (DeepInfra) test failed or not configured. Please check your .env and library installation.")

        
        # Example of calling the API (ensure your .env is set up)
        # This is a generic prompt, replace with something your model expects
        # test_prompt = "Translate 'hello world' to French. Respond in JSON with a single key 'translation'."
        # print(f"\nAttempting a test API call with prompt: {test_prompt}")
        # response = call_llm_api(test_prompt, expect_json=True)
        
        # if response:
        #     print(f"Test API call successful. Response: {response}")
        # else:
        #     print("Test API call failed or returned None.") 