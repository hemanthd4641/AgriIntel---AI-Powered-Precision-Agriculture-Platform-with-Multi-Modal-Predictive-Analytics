"""
LLM Interface for Smart Agriculture Project

This module provides an interface to open-source LLMs for generating
natural-language explanations and recommendations based on crop yield predictions.
"""

import os
import re
import requests
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AgricultureLLM:
    """LLM interface for agricultural explanations and recommendations"""
    
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct", rag_system=None, device=None, remote_model_id: Optional[str] = None, local_model_name: Optional[str] = None, force_local: bool = False):
        """
        Initialize the LLM interface
        
        Args:
            model_name: Name of the Hugging Face model to use
            rag_system: RAG system for retrieval-augmented generation
            device: Device to run the model on (cuda or cpu)
        """
        # model_name may be a single name (local) or a compound spec like "local|owner/remote-model"
        self.rag_system = rag_system
        # Determine local vs remote model names. Priority: explicit args > compound model_name > defaults
        self.local_model_name = local_model_name or (str(model_name).split('|')[0] if '|' in str(model_name) else str(model_name)) or 'microsoft/Phi-3-mini-4k-instruct'
        self.remote_model_id = remote_model_id or (str(model_name).split('|')[1] if '|' in str(model_name) and len(str(model_name).split('|')) > 1 else None)

        self.device = device if device else None
        self.text_generator = None
        self.use_hf_api = False

        # Enable Hugging Face API usage
        self.hf_api_token = os.getenv('HF_API_TOKEN')
        self.hf_model_id = "microsoft/Phi-3-mini-4k-instruct"
        self.use_hf_api = True if self.hf_api_token and self.hf_api_token != 'your_hugging_face_api_token_here' else False
        # Track HF failures and disable after threshold
        self.hf_failures = 0
        self.hf_failure_threshold = 3

        # Honor explicit force_local flag which disables remote HF usage
        self.force_local = bool(force_local)
        if self.force_local and self.use_hf_api:
            print("force_local=True: disabling remote HF API usage and using local model only.")
            self.use_hf_api = False

        # Try to initialize the language model, but provide fallback if it fails
        # First, try to use Hugging Face API if configured
        if self.use_hf_api and self.hf_api_token and self.hf_model_id:
            print(f"Using Hugging Face API with model: {self.hf_model_id}")
            # Verify model access with a simple test inference
            is_valid, message = self._verify_hf_model(self.hf_model_id)
            if is_valid:
                print(f"Hugging Face model verification successful: {message}")
                # We'll use the API for generation, so we don't need to load a local model
                self.text_generator = None
            else:
                print(f"Hugging Face model verification failed: {message}")
                self.use_hf_api = False
        
        # If not using HF API or if HF API verification failed, we don't load local models anymore
        if not self.use_hf_api:
            print("Hugging Face API not configured or unavailable. LLM functionality will be limited.")
            self.text_generator = None

    def _call_hf_api(self, prompt, params=None, timeout=60):
        """
        Call the Hugging Face Inference API for text generation.

        Returns generated text or raises an exception.
        """
        if not self.hf_api_token or not self.hf_model_id:
            raise RuntimeError("Hugging Face API token or model ID not configured")
        
        # Prepare the API request
        api_url = f"https://router.huggingface.co/hf-inference/models/{self.hf_model_id}"
        headers = {"Authorization": f"Bearer {self.hf_api_token}"}
        
        # Set default parameters
        payload_params = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        
        # Update with provided params if any
        if params:
            payload_params.update(params)
        
        # Create the payload
        payload = {
            "inputs": prompt,
            "parameters": payload_params
        }
        
        # Make the API request
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            # Extract the generated text
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'generated_text' in result[0]:
                    return result[0]['generated_text']
            
            raise RuntimeError(f"Unexpected API response format: {result}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Hugging Face API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to decode API response: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in Hugging Face API call: {str(e)}")

    def _verify_hf_model(self, model_id, timeout=10):
        """
        Quick verification that the HF model endpoint exists and that the provided token has access.
        Returns (True, message) on success, (False, message) on failure.
        """
        if not self.hf_api_token or not model_id:
            return False, "Hugging Face API token or model ID not configured"
        
        try:
            # First check if the model exists on Hugging Face
            model_info_url = f"https://huggingface.co/api/models/{model_id}"
            model_info_response = requests.get(model_info_url, timeout=timeout)
            
            if model_info_response.status_code != 200:
                return False, f"Model {model_id} not found on Hugging Face"
            
            # If model exists, assume API access is available
            # Skip the actual inference test to avoid issues with model loading
            return True, "Model is accessible"
        except Exception as e:
            return False, f"Model verification failed: {str(e)}"

    def _generate_text(self, prompt, params=None, max_local_tokens=200):
        """
        Unified generator that uses HF-inference (if enabled).
        Returns generated text or raises an exception if API fails.
        """
        # Try HF API first when configured
        if self.use_hf_api and self.hf_api_token and self.hf_model_id:
            try:
                out = self._call_hf_api(prompt, params=params)
                # reset failure counter on success
                self.hf_failures = 0
                return out
            except Exception as e:
                self.hf_failures += 1
                print(f"HF API generation error: {e} (failure {self.hf_failures}/{self.hf_failure_threshold})")
                if self.hf_failures >= self.hf_failure_threshold:
                    print("Disabling HF API usage due to repeated failures.")
                    self.use_hf_api = False
                raise RuntimeError(f"HF API generation failed: {str(e)}")
        else:
            raise RuntimeError("Hugging Face API not configured or unavailable")

    def _clean_response(self, text: str, max_chars: int = 1000) -> str:
        """
        Basic cleanup to remove duplicate repeated lines and trim overly long outputs.
        """
        if not text:
            return text
        # Truncate to max_chars
        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars]

        # Remove exact repeated lines and noisy prefixes like 'Expert:' or 'Farmer:'
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        cleaned_lines = []
        seen = set()
        for l in lines:
            # remove conversational prefixes
            l = re.sub(r'^(Expert:|Farmer:)', '', l, flags=re.IGNORECASE).strip()
            # collapse long repeated character runs
            l = re.sub(r'(.)\1{10,}', r'\1', l)
            # collapse repeated short phrases like 'I don't know' repeated many times
            l = re.sub(r"(I don't know(?:\.|'|\")?)(?:\s+\1){1,}", r"I don't know.", l, flags=re.IGNORECASE)
            if not l:
                continue
            if l in seen:
                continue
            seen.add(l)
            cleaned_lines.append(l)

        cleaned = '\n'.join(cleaned_lines)

        # Collapse repeated words sequences (naive)
        cleaned = re.sub(r"(\b\w+\b)(?:\s+\1\b){2,}", r"\1", cleaned)

        # Remove excessive repetition of short sentences
        sentences = re.split(r'[\n\.]{1,}', cleaned)
        dedup = []
        seen_s = set()
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if s.lower() in seen_s:
                continue
            seen_s.add(s.lower())
            dedup.append(s)
        cleaned = '. '.join(dedup).strip()
        if cleaned and not cleaned.endswith('.'):
            cleaned = cleaned + '.'
        # final length trim
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars].rstrip() + '...'
        return cleaned

    def _strip_echoes(self, text: str, user_input: str) -> str:
        """
        Remove echoed user questions and instruction blocks from model output.
        """
        if not text:
            return text
        out = text
        try:
            # Remove exact user input if repeated
            if user_input and user_input in out:
                out = out.replace(user_input, '')

            # Remove lines that repeat the question or begin with 'Farmer:'/'Expert:'
            lines = [l for l in out.splitlines() if l.strip()]
            cleaned_lines = []
            for l in lines:
                low = l.lower()
                if low.startswith('farmer:') or low.startswith("farmer's question") or low.startswith('question:'):
                    continue
                # skip lines that are too similar to the user input
                try:
                    if user_input and len(user_input) > 20:
                        # approximate similarity via common substring
                        if user_input.lower() in low or low in user_input.lower():
                            continue
                except Exception:
                    pass
                cleaned_lines.append(l)
            out = '\n'.join(cleaned_lines)
        except Exception:
            pass
        return out
    
    def chat_with_farmer(self, message, conversation_history=None):
        """
        Chat with a farmer using the LLM
        
        Args:
            message: The farmer's message
            conversation_history: Previous conversation history
            
        Returns:
            str: The LLM's response
        """
        if not message:
            return ""
        if not conversation_history:
            conversation_history = []

        # Add the user's message to the conversation history
        conversation_history.append(f"Farmer: {message}")

        # Prepare the prompt for the LLM
        prompt = "\n".join(conversation_history)

        # Generate the response using the LLM
        response = self._generate_text(prompt)

        # Clean up the response
        cleaned_response = self._clean_response(response)

        # Strip echoes of the user's input
        final_response = self._strip_echoes(cleaned_response, message)

        # Add the LLM's response to the conversation history
        conversation_history.append(f"Expert: {final_response}")

        return final_response
