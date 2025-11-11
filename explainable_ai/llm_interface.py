"""
LLM Interface for Smart Agriculture Project

This module provides an interface to open-source LLMs for generating
natural-language explanations and recommendations based on crop yield predictions.
"""

import torch
import os
import re
import requests
import json
from typing import Optional, Dict, Any
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global cache for LLM models and tokenizers with thread safety
_model_cache: Dict[str, Any] = {}
_tokenizer_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()

# Detect whether accelerate is available; if not, avoid using device_map="auto"
try:
    import accelerate as _accelerate  # noqa: F401
    _ACCELERATE_AVAILABLE = True
except Exception:
    _ACCELERATE_AVAILABLE = False

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

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            # Verify model access
            is_valid, message = self._verify_hf_model(self.hf_model_id)
            if is_valid:
                print(f"Hugging Face model verification successful: {message}")
                # We'll use the API for generation, so we don't need to load a local model
                self.text_generator = None
            else:
                print(f"Hugging Face model verification failed: {message}")
                self.use_hf_api = False
        
        # If not using HF API or if HF API verification failed, try to load local model
        if not self.use_hf_api:
            try:
                # Import here to avoid issues if transformers is not installed
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                # Detect offline mode via env vars
                offline_env = bool(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE") or os.environ.get("AGRI_LLM_OFFLINE"))

                print(f"Attempting to load local LLM model: {self.local_model_name}")

                # Check if model is already cached with thread safety
                model_key = f"{self.local_model_name}_{self.device}"
                
                with _cache_lock:
                    if model_key in _model_cache and model_key in _tokenizer_cache:
                        print(f"Using cached model and tokenizer for {self.local_model_name}")
                        cached_model = _model_cache[model_key]
                        cached_tokenizer = _tokenizer_cache[model_key]
                        
                        # Create text generation pipeline with cached model and tokenizer
                        pipe_kwargs = dict(
                            task="text-generation",
                            model=cached_model,
                            tokenizer=cached_tokenizer,
                            max_new_tokens=256,
                            temperature=0.0,
                            do_sample=False,
                            top_p=1.0,
                        )
                        # If model was loaded with accelerate (device_map present), don't pass a specific device
                        if getattr(cached_model, "hf_device_map", None) is None:
                            pipe_kwargs["device"] = 0 if self.device.type == "cuda" else -1
                        self.text_generator = pipeline(**pipe_kwargs)
                    else:
                        print(f"Loading model {self.local_model_name} for the first time...")
                        # Load model and tokenizer with optimized settings for caching
                        def _load_llm(local_files_only: bool):
                            tok = AutoTokenizer.from_pretrained(self.local_model_name, local_files_only=local_files_only)
                            mdl = AutoModelForCausalLM.from_pretrained(
                                self.local_model_name,
                                dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                                low_cpu_mem_usage=True,
                                device_map=("auto" if (_ACCELERATE_AVAILABLE and not local_files_only) else None),
                                local_files_only=local_files_only
                            )
                            return tok, mdl

                        try:
                            tokenizer, model = _load_llm(local_files_only=offline_env)
                        except Exception as e:
                            # If network failure, switch to offline attempt
                            msg = str(e)
                            if ("NameResolutionError" in msg) or ("Failed to resolve" in msg) or ("MaxRetryError" in msg) or ("ConnectionError" in msg):
                                print("Network unavailable for HuggingFace hub; attempting offline load from cache.")
                                try:
                                    tokenizer, model = _load_llm(local_files_only=True)
                                except Exception:
                                    raise
                            else:
                                raise
                        
                        # Move model to device if not using device_map
                        if model.device != self.device and "auto" not in str(getattr(model, "device_map", "")):
                            model = model.to(self.device)
                        
                        # Cache the model and tokenizer for future use
                        _model_cache[model_key] = model
                        _tokenizer_cache[model_key] = tokenizer
                        print(f"Cached model and tokenizer for {self.local_model_name}")
                        
                        # Create text generation pipeline
                        pipe_kwargs = dict(
                            task="text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            max_new_tokens=256,
                            temperature=0.0,
                            do_sample=False,
                            top_p=1.0,
                        )
                        # If model was loaded with accelerate (device_map present), don't pass a specific device
                        if getattr(model, "hf_device_map", None) is None:
                            pipe_kwargs["device"] = 0 if self.device.type == "cuda" else -1
                        self.text_generator = pipeline(**pipe_kwargs)
                
                print(f"Successfully loaded LLM model: {self.local_model_name}")
            except Exception as e:
                print(f"Warning: Could not initialize local LLM model {self.local_model_name}. Using fallback. Error: {str(e)}")
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
            # Use the Hugging Face Inference API endpoint
            api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
            headers = {"Authorization": f"Bearer {self.hf_api_token}"}
            # Simple model info request
            response = requests.get(api_url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return True, "Model is accessible"
            else:
                return False, f"Model access failed with status {response.status_code}: {response.text}"
        except Exception as e:
            return False, f"Model verification failed: {str(e)}"

    def _generate_text(self, prompt, params=None, max_local_tokens=200):
        """
        Unified generator that prefers HF-inference (if enabled) then falls back to local pipeline.
        Returns generated text or raises an exception if both backends fail.
        """
        # If force_local is set, skip any HF API attempts and use the local pipeline only
        if getattr(self, 'force_local', False):
            # intentionally skip HF and go straight to local generation
            hf_attempted = False
        else:
            hf_attempted = True

        # Try HF API first when configured (unless force_local)
        if hf_attempted and self.use_hf_api and self.hf_api_token and self.hf_model_id:
            try:
                out = self._call_hf_api(prompt, params=params)
                # reset failure counter on success
                self.hf_failures = 0
                return out
            except Exception as e:
                self.hf_failures += 1
                print(f"HF API generation error: {e} (failure {self.hf_failures}/{self.hf_failure_threshold})")
                if self.hf_failures >= self.hf_failure_threshold:
                    print("Disabling HF API usage due to repeated failures; will use local model instead.")
                    self.use_hf_api = False

        # Try local pipeline
        if self.text_generator:
            try:
                local_max = params.get('max_new_tokens', max_local_tokens) if params else max_local_tokens
                # Keep deterministic defaults unless the caller requested otherwise
                response = self.text_generator(
                    prompt,
                    max_new_tokens=local_max,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=50256,
                    top_p=1.0
                )
                full = response[0].get('generated_text') if isinstance(response[0], dict) else str(response[0])
                # Normalize and reduce common repetition patterns
                if isinstance(full, str):
                    # Robustly remove echoed prompt fragments: find a long suffix of the prompt that appears
                    # in the generated text and strip everything up to and including that occurrence.
                    def _remove_prompt_echo(prompt_text, generated_text):
                        if not prompt_text or not generated_text:
                            return generated_text
                        try:
                            # consider suffix lengths from 300 down to 30 characters
                            max_check = min(300, len(prompt_text))
                            for n in range(max_check, 29, -1):
                                suffix = prompt_text[-n:]
                                idx = generated_text.find(suffix)
                                if idx != -1:
                                    # remove up to end of suffix
                                    return generated_text[idx + len(suffix):]
                        except Exception:
                            pass
                        return generated_text

                    try:
                        full = _remove_prompt_echo(prompt, full)
                    except Exception:
                        pass
                    # Remove repeated conversational prefixes like 'Expert:' or 'Farmer:'
                    full = re.sub(r'(?i)(Expert:\s*)+', '', full)
                    full = re.sub(r'(?i)(Farmer:\s*)+', '', full)

                    # Collapse repeated 'I don't know' or similar phrases
                    full = re.sub(r"(I don't know[\.!\s]*){2,}", "I don't know. ", full, flags=re.IGNORECASE)

                    # Strip common echoed system prompts or instruction blocks that GPT-2 may repeat
                    try:
                        # Remove the exact first-line instruction if echoed
                        full = re.sub(r'You are an agricultural expert assistant[\s\S]{0,300}?\n', '', full, flags=re.IGNORECASE)
                        # Remove repeated boilerplate phrases
                        full = re.sub(r'Provide helpful, concise, and practical advice[\s\S]{0,200}?', '', full, flags=re.IGNORECASE)
                        # Remove any stray 'Farmer:' or 'Expert:' labels left at line starts
                        full = re.sub(r'^(Farmer:|Expert:)\s*', '', full, flags=re.IGNORECASE | re.MULTILINE)
                        # If the original prompt substring exists, try to remove it
                        if prompt and prompt in full:
                            try:
                                full = full.replace(prompt, '', 1)
                            except Exception:
                                pass
                        else:
                            # Remove long common prefixes between prompt and full (model may echo back a large chunk)
                            try:
                                a = prompt or ''
                                b = full or ''
                                i = 0
                                max_i = min(len(a), len(b))
                                while i < max_i and a[i] == b[i]:
                                    i += 1
                                # if a significant common prefix exists, strip it
                                if i > 50:
                                    full = full[i:]
                            except Exception:
                                pass

                        # Remove explicit "Farmer's question" or role echoes if present
                        try:
                            full = re.sub(r"(?i)farmer(?:'s)?\s+question:\s*", '', full)
                            full = re.sub(r"(?i)question:\s*", '', full, count=1)
                            full = re.sub(r"(?i)farmer:\s*", '', full)
                            full = re.sub(r"(?i)expert:\s*", '', full)
                        except Exception:
                            pass
                    except Exception:
                        pass

                    # Keep only the first few meaningful lines to avoid runaway repetition
                    lines = [l.strip() for l in full.splitlines() if l.strip()]
                    if len(lines) > 8:
                        lines = lines[:8]
                    full = '\n'.join(lines)

                    # Final whitespace cleanup
                    full = full.strip()

                return full
            except Exception as e:
                print(f"Local generation error: {e}")

        # If neither backend produced output, raise
        raise RuntimeError("No available LLM backend produced output")

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
        # Build a deterministic prompt for chat with explicit instruction for Phi-3 Mini
        prompt = (
            "You are an agricultural expert assistant helping farmers with their questions. "
            "Only use the PROVIDED CONTEXT below to answer. Do NOT hallucinate or invent facts. "
            "If the answer is not present in the context, respond: 'I don't have enough information in the provided context to answer that.'\n\n"
            "Return a short numbered list (1-3 items) of practical, actionable steps the farmer can take. Be specific, concrete, and concise. "
            "Each item should be no longer than 140 characters.\n\n"
        )

        # Add conversation history if available (last 3 turns)
        if conversation_history:
            for turn in conversation_history[-3:]:
                prompt += f"Farmer: {turn.get('user','')}\nExpert: {turn.get('assistant','')}\n"

        # Add current question
        prompt += f"\nFarmer: {message}\nExpert:"

        # Try unified generation (HF preferred unless force_local, then local only)
        relevant = None
        try:
            # Include relevant docs in the prompt if available (strongly encourage grounding)
            if self.rag_system:
                try:
                    relevant = self.rag_system.retrieve_relevant_documents(message, k=5)
                except Exception:
                    relevant = None

                if relevant:
                    # If we have retrieved docs, prefer a doc-grounded short answer first
                    try:
                        doc_answer = self._doc_grounded_answer(message, relevant)
                    except Exception:
                        doc_answer = ''
                    if doc_answer and len(doc_answer) > 10:
                        return doc_answer

                    prompt += "\nCONTEXT (use only these excerpts):\n"
                    # Keep shorter, high-signal snippets
                    for i, d in enumerate(relevant[:5], 1):
                        snippet = d.get('content', d.get('page_content', '')) or ''
                        snippet = ' '.join(snippet.split())[:350]
                        prompt += f"{i}) {snippet}\n"
                    prompt += "\n"

            # If we have relevant documents, prefer a document-grounded extraction to ensure factual answers
            params = {"max_new_tokens": 220, "temperature": 0.0, "top_p": 1.0}
            # Document-grounded extraction attempted above; if present it would have returned early

            gen = self._generate_text(prompt, params=params, max_local_tokens=220)
            if isinstance(gen, str):
                # Strip echoed prompt if present
                out = gen.strip()
            else:
                out = str(gen)

            # Truncate and sanitize
            out = self._clean_response(out, max_chars=700)

            # If output is too short or looks like garbage, fall back to doc-based extraction
            if not out or len(out) < 20 or out.lower().count('\n') == 0 and len(out.split()) < 6:
                return self._fallback_chat_response(message, relevant_docs=relevant if relevant else None)

            return out
        except Exception as e:
            print(f"Unified generation failed for chat: {e}")
            return self._fallback_chat_response(message, relevant_docs=relevant)
    
    def _fallback_chat_response(self, message, relevant_docs=None):
        """
        Provide a fallback chat response when LLM is not available
        
        Args:
            message: The farmer's message
            
        Returns:
            str: Fallback response
        """
        # If relevant documents were provided, build a concise bullet summary grounded in them
        if relevant_docs:
            # Build a numbered, concise, document-grounded answer (1-3 items)
            items = []
            for doc in relevant_docs[:3]:
                text = doc.get('content', doc.get('page_content', '')) or ''
                # Prefer sentences that include keywords from the user message
                sentences = [s.strip() for s in re.split(r'[\n\.]\s+', text) if s.strip()]
                picked = None
                qwords = [w.lower() for w in re.findall(r'\w{4,}', message)]
                for s in sentences:
                    sl = s.lower()
                    if any(q in sl for q in qwords):
                        picked = s
                        break
                if not picked and sentences:
                    picked = sentences[0]
                if picked:
                    picked = re.sub(r'\s+', ' ', picked).strip()
                    if len(picked) > 220:
                        picked = picked[:217].rstrip() + '...'
                    items.append(picked)

            if not items:
                # fallback: short extracts
                items = [re.sub(r'\s+', ' ', (d.get('content', d.get('page_content', '')) or '')[:200]).strip() + '...' for d in relevant_docs[:2]]

            # Create numbered list
            out = ''
            for i, it in enumerate(items[:3], 1):
                out += f"{i}) {it.strip()}\n"

            # Add sources if available
            srcs = []
            for d in relevant_docs[:3]:
                meta = d.get('metadata', {}) or {}
                title = meta.get('title') or meta.get('source') or d.get('source') or d.get('title')
                if title and title not in srcs:
                    srcs.append(title)
            if srcs:
                out += f"Sources: {', '.join(srcs)}"
            return out.strip()
        else:
            # Simple compact rule-based responses when no docs are available
            message_lower = message.lower()
            # Targeted concise rule-based answers for common farmer queries
            if 'aphid' in message_lower:
                return "1) Scout field edges and undersides of leaves daily; 2) Encourage natural predators (ladybirds); 3) Use insecticidal soap or neem when threshold exceeded."
            if 'nitrogen deficiency' in message_lower or ('ndvi' in message_lower and 'low' in message_lower) or 'yellowing from the bottom' in message_lower:
                return "1) Look for uniform yellowing of older leaves; 2) Apply a balanced N top-dress (e.g., urea) at recommended rate; 3) Confirm with soil/test before heavy applications."
            if 'seed rate' in message_lower or 'seed rate per' in message_lower or 'seed rate per hectare' in message_lower:
                return "Typical soybean seed rate: 70-100 kg/ha (adjust by variety and seed size). Check local recommendations."
            if 'bacterial wilt' in message_lower or 'bacterial wilt' in message_lower:
                return "1) Remove and destroy infected plants; 2) Avoid moving soil/water between fields; 3) Use resistant varieties and crop rotation."
            if 'post-harvest' in message_lower or 'transport' in message_lower or 'storage' in message_lower:
                return "1) Harvest at correct maturity; 2) Cool produce quickly and use padded crates; 3) Avoid bruising, sort and remove damaged items before transport."
            if any(keyword in message_lower for keyword in ['tomato', 'grow', 'cultivat']):
                return "Quick tips for tomatoes: 1) 6-8h sun; 2) well-draining soil pH 6.0-6.8; 3) consistent watering; 4) stake plants; 5) monitor pests."
            if any(keyword in message_lower for keyword in ['soil', 'fertil', 'nutri']):
                return "Quick soil tips: test soil, add compost, rotate crops, and apply balanced fertilizer per test results."
            if any(keyword in message_lower for keyword in ['pest', 'insect', 'bug']):
                return "Quick pest tips: monitor regularly, use IPM (beneficials + cultural controls), and apply targeted treatments only when needed."
            if any(keyword in message_lower for keyword in ['disease', 'fungus', 'blight']):
                return "Quick disease tips: improve airflow, avoid overhead watering, remove infected plants, and use resistant varieties."
            if any(keyword in message_lower for keyword in ['water', 'irrigat', 'drought']):
                return "Quick irrigation tips: water deeply, use mulch, consider drip irrigation, and schedule watering for cooler parts of the day."

            return "I can help with crop selection, soil, pests, irrigation, and more — please provide a specific question or location/climate for targeted advice."
    
    def _doc_grounded_answer(self, question, relevant_docs, max_items: int = 3) -> str:
        """
        Synthesize a concise, numbered answer from retrieved documents without calling the LLM.

        Args:
            question: user question
            relevant_docs: list of dicts with 'content' and 'metadata'
            max_items: maximum numbered items to return

        Returns:
            str: numbered answer or empty string if not possible
        """
        if not relevant_docs:
            return ''

        # Collect candidate sentences from top documents
        candidates = []
        qwords = {w.lower() for w in re.findall(r"\w{4,}", question)}
        for d in relevant_docs[:6]:
            text = d.get('content', d.get('page_content', '')) or ''
            # split into sentences, keep short ones
            sents = [s.strip() for s in re.split(r'[\n\.]\s+', text) if s.strip()]
            for s in sents:
                if len(s) < 12:
                    continue
                low = s.lower()
                # word-overlap based score
                sent_words = {w for w in re.findall(r"\w{4,}", low)}
                overlap = len(qwords & sent_words)
                score = overlap * 3
                # also compute fractional overlap to boost short good matches
                if qwords:
                    frac = overlap / max(1, len(qwords))
                    score += int(frac * 2)
                # boost imperatives / actionable verbs
                if re.search(r'\b(apply|use|test|monitor|rotate|irrigat|water|fertili|control|prune|stake|mulch|harvest|store|drip)\b', low):
                    score += 1
                # prefer shorter actionable sentences
                score += max(0, 10 - (len(s) // 50))
                candidates.append({'sent': s, 'score': score, 'source': (d.get('metadata') or {}).get('title') or d.get('title') or ''})

        if not candidates:
            return ''

        # Sort candidates by score and deduplicate similar sentences
        candidates.sort(key=lambda x: x['score'], reverse=True)
        picked = []
        seen = set()
        for c in candidates:
            s = re.sub(r'\s+', ' ', c['sent']).strip()
            s_short = s.lower()[:120]
            if s_short in seen:
                continue
            seen.add(s_short)
            picked.append((s, c.get('source', '')))
            if len(picked) >= max_items:
                break

        if not picked:
            return ''

        # Build numbered output
        out_lines = []
        for i, (s, src) in enumerate(picked, 1):
            line = f"{i}) {s}"
            if src:
                line += f" (source: {src})"
            out_lines.append(line)

        return '\n'.join(out_lines)

    def generate_explanation(self, prediction, weather_conditions=None, ndvi_value=None):
        """
        Generate a natural-language explanation for a crop yield prediction
        
        Args:
            prediction: Predicted crop yield value
            weather_conditions: Summary of recent weather conditions
            ndvi_value: Current NDVI value indicating vegetation health
            
        Returns:
            str: Natural-language explanation
        """
        # Unified generation with fallback
        prompt = f"""
        Based on the analysis of satellite imagery and weather data, the predicted crop yield is {prediction:.2f} tons per hectare.
        
        Analysis parameters:
        - Weather conditions: {weather_conditions if weather_conditions else 'Variable conditions'}
        - Vegetation health (NDVI): {ndvi_value if ndvi_value else 'Moderate health'}
        
        Please provide a clear, concise explanation of what this prediction means for the farmer, including:
        1. Whether this yield is above or below average
        2. Key factors that contributed to this prediction
        3. Any concerns or positive indicators
        4. Recommendations for optimizing yield
        """
        try:
            params = {"max_new_tokens": 200, "temperature": 0.0, "top_p": 1.0}
            out = self._generate_text(prompt, params=params, max_local_tokens=200)
            return out.strip() if isinstance(out, str) else str(out).strip()
        except Exception as e:
            print(f"Explanation generation failed: {e}")
            return self._fallback_explanation(prediction, weather_conditions, ndvi_value)
    
    def _fallback_explanation(self, prediction, weather_conditions=None, ndvi_value=None):
        """
        Provide a fallback explanation when LLM is not available
        
        Args:
            prediction: Predicted crop yield value
            weather_conditions: Summary of recent weather conditions
            ndvi_value: Current NDVI value indicating vegetation health
            
        Returns:
            str: Fallback explanation
        """
        explanation = f"Based on the analysis, the predicted crop yield is {prediction:.2f} tons per hectare. "
        
        if weather_conditions:
            explanation += f"The weather conditions have been {weather_conditions.lower()}. "
        
        # Simple rule-based vegetation health assessment
        if ndvi_value:
            if ndvi_value > 0.7:
                explanation += "Vegetation health is excellent, which is a positive indicator for yield. "
            elif ndvi_value > 0.5:
                explanation += "Vegetation health is moderate, which should support a reasonable yield. "
            else:
                explanation += "Vegetation health is below optimal, which may impact the final yield. "
        
        explanation += "For best results, ensure proper irrigation and fertilization practices are followed."
        return explanation
    
    def generate_recommendations(self, prediction, weather_forecast=None, soil_conditions=None):
        """
        Generate actionable recommendations based on prediction and conditions
        
        Args:
            prediction: Predicted crop yield value
            weather_forecast: Upcoming weather forecast
            soil_conditions: Current soil conditions
            
        Returns:
            str: Actionable recommendations
        """
        # Unified generation with fallback
        prompt = f"""
        Based on a predicted crop yield of {prediction:.2f} tons per hectare, provide specific recommendations for a farmer to:
        
        1. Optimize their current farming practices
        2. Address any potential issues identified in the analysis
        3. Maximize their final yield
        
        Context:
        - Weather forecast: {weather_forecast if weather_forecast else 'Normal conditions expected'}
        - Soil conditions: {soil_conditions if soil_conditions else 'Adequate fertility'}
        
        Provide 3-5 specific, actionable recommendations that a farmer can implement immediately.
        """
        try:
            params = {"max_new_tokens": 300, "temperature": 0.7, "top_p": 1.0}
            out = self._generate_text(prompt, params=params, max_local_tokens=300)
            return out.strip() if isinstance(out, str) else str(out).strip()
        except Exception as e:
            print(f"Recommendation generation failed: {e}")
            return self._fallback_recommendations(prediction, weather_forecast, soil_conditions)
    
    def _fallback_recommendations(self, prediction, weather_forecast=None, soil_conditions=None):
        """
        Provide fallback recommendations when LLM is not available
        
        Args:
            prediction: Predicted crop yield value
            weather_forecast: Upcoming weather forecast
            soil_conditions: Current soil conditions
            
        Returns:
            str: Fallback recommendations
        """
        recommendations = [
            "Monitor soil moisture levels regularly and adjust irrigation accordingly.",
            "Apply balanced fertilizers based on soil test results to optimize nutrient availability.",
            "Implement pest and disease monitoring programs to prevent yield losses.",
            "Consider crop rotation practices to maintain soil health and reduce disease pressure.",
            "Keep detailed records of farming practices and yields to improve future predictions."
        ]
        
        return " ".join(recommendations)
    
    def answer_question(self, question):
        """
        Answer a farmer's question using RAG-augmented generation
        
        Args:
            question: Farmer's question about agriculture
            
        Returns:
            dict: Answer and source documents
        """
        try:
            # Retrieve relevant documents using RAG system (compatibility fallback)
            relevant_docs = []
            if self.rag_system and getattr(self.rag_system, 'unified_kb', None):
                try:
                    # UnifiedAgriculturalKB may expose different method names; try multiple
                    unified = self.rag_system.unified_kb
                    if hasattr(unified, 'query_relevant_documents'):
                        relevant_docs = unified.query_relevant_documents(question, k=3)
                    elif hasattr(unified, 'query_knowledge_base'):
                        relevant_docs = unified.query_knowledge_base(question, k=3)
                    else:
                        # Fall back to RAG system's own query method
                        relevant_docs = self.rag_system.query_knowledge_base(question, k=3)
                except Exception as e:
                    print(f"Error querying knowledge base: {str(e)}")
                    relevant_docs = []
            
            # Fast path: small deterministic rule-based answers for common, factual farmer queries
            rb = self._rule_based_answer(question)
            if rb:
                return {"answer": rb, "source_documents": []}

            # Generate answer with context using unified generator (HF preferred, then local)
            if relevant_docs:
                # Prefer a doc-grounded extraction to avoid LLM hallucination when documents are available
                try:
                    doc_answer = self._doc_grounded_answer(question, relevant_docs)
                except Exception:
                    doc_answer = ''

                def _serialize(doc):
                    meta = doc.get('metadata', {}) or {}
                    title = meta.get('title') or meta.get('source') or doc.get('source') or doc.get('title') or meta.get('name') or ''
                    source = meta.get('source') or meta.get('path') or doc.get('source') or doc.get('id') or ''
                    snippet = (doc.get('content', doc.get('page_content', '')) or '')[:300]
                    return {'title': str(title), 'source': str(source), 'snippet': str(snippet)}

                sources_clean = [_serialize(d) for d in relevant_docs[:3]]

                if doc_answer and len(doc_answer) > 10:
                    # Return the concise doc-grounded answer with sources
                    return {
                        "answer": doc_answer,
                        "source_documents": sources_clean
                    }

                # If doc-grounded extraction didn't return a good answer, try RAG-augmented LLM generation
                try:
                    # Create prompt with context
                    context = "Relevant agricultural information:\n"
                    for i, doc in enumerate(relevant_docs[:3], 1):
                        content = doc.get('content', doc.get('page_content', ''))[:200]
                        context += f"{i}. {content}\n"

                    prompt = f"""You are an agricultural expert. Using only the information below, provide a short numbered list (3 items max) of practical actions the farmer can take in answer to the question.

{context}

Question: {question}

Answer:"""

                    params = {"max_new_tokens": 200, "temperature": 0.0, "top_p": 1.0}
                    out = self._generate_text(prompt, params=params, max_local_tokens=200)
                    answer = self._clean_response(out.strip() if isinstance(out, str) else str(out), max_chars=800)

                    # Clean up the answer
                    if '.' in answer:
                        last_period = answer.rfind('.')
                        answer = answer[:last_period + 1]

                    sources_clean = [_serialize(d) for d in relevant_docs[:3]]

                    # If the LLM generated an answer that looks like an echo of the prompt or is low-quality,
                    # fall back to doc-grounded answer if possible.
                    cleaned_answer = answer if len(answer) > 10 else ''
                    if not cleaned_answer:
                        # Try doc grounded as a fallback
                        try:
                            doc_answer2 = self._doc_grounded_answer(question, relevant_docs)
                            if doc_answer2 and len(doc_answer2) > 10:
                                return {"answer": doc_answer2, "source_documents": sources_clean}
                        except Exception:
                            pass

                    return {
                        "answer": cleaned_answer if cleaned_answer else self._fallback_chat_response(question),
                        "source_documents": sources_clean
                    }
                except Exception as e:
                    print(f"Error in RAG-augmented generation: {str(e)}")
                    # Fallback to basic answer with retrieved documents
                    if relevant_docs:
                        answer = "Based on agricultural knowledge: "
                        for doc in relevant_docs[:2]:
                            content = doc.get('content', doc.get('page_content', ''))[:100]
                            answer += f"{content}... "
                        answer += "How can I help you further with your farming questions?"
                    else:
                        answer = self._fallback_chat_response(question)

                    # Serialize fallback sources
                    def _serialize_f(doc):
                        meta = doc.get('metadata', {}) or {}
                        title = meta.get('title') or meta.get('source') or doc.get('source') or doc.get('title') or meta.get('name') or ''
                        source = meta.get('source') or meta.get('path') or doc.get('source') or doc.get('id') or ''
                        snippet = (doc.get('content', doc.get('page_content', '')) or '')[:300]
                        return {'title': str(title), 'source': str(source), 'snippet': str(snippet)}

                    sources_clean = [_serialize_f(d) for d in (relevant_docs[:3] if relevant_docs else [])]

                    return {
                        "answer": answer,
                        "source_documents": sources_clean
                    }
            else:
                # Fallback when no LLM or no documents
                answer = self._fallback_chat_response(question)
                return {
                    "answer": answer,
                    "source_documents": []
                }
        except Exception as e:
            print(f"Error in question answering: {str(e)}")
            return {
                "answer": self._fallback_chat_response(question),
                "source_documents": []
            }
    


    def _rule_based_answer(self, question: str) -> str:
        """
        Provide compact, factual rule-based answers for a small set of high-value queries.
        Returns a short string answer or empty string if no rule applies.
        """
        if not question:
            return ''
        q = question.lower()
        # Tomato harvesting question
        if 'tomato' in q and any(k in q for k in ['harvest', 'when to harvest', 'picking', 'maturity', 'ripen']):
            # Give practical, variety-agnostic guidance
            ans_lines = [
                "1) Harvest when fruits reach full color for the variety (red/yellow/orange) and give slightly to gentle pressure.",
                "2) Typical maturity is 50–85 days after transplanting depending on cultivar — check seed packet for days-to-maturity.",
                "3) Pick ripe fruits every 2–3 days; avoid harvesting based on season week alone (use color and firmness)."
            ]
            return '\n'.join(ans_lines)

        return ''

# Example usage
if __name__ == "__main__":
    # Initialize LLM (using a small model for demonstration)
    # In practice, you might want to use a larger model like "microsoft/Phi-3-mini-128k-instruct" or a specialized agricultural model
    try:
        llm = AgricultureLLM(model_name="microsoft/Phi-3-mini-4k-instruct")
    except:
        llm = AgricultureLLM()  # Use fallback
    
    # Example prediction explanation
    explanation = llm.generate_explanation(
        prediction=3.5,
        weather_conditions="Recent drought conditions with below-average rainfall",
        ndvi_value=0.45
    )
    print("Explanation:", explanation)
    
    # Example recommendations
    recommendations = llm.generate_recommendations(
        prediction=3.5,
        weather_forecast="Rain expected next week",
        soil_conditions="Nitrogen levels low"
    )
    print("Recommendations:", recommendations)