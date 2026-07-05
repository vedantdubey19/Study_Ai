"""
LLM Provider Module for Study AI
Supports multiple AI backends: Ollama (local), Google Gemini (API), Groq (API).
"""

import os
import json
import ollama
import requests
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

# -------------------------
# Provider Configuration
# -------------------------
PROVIDERS = {
    "Ollama (Local)": {
        "models": ["gemma3:1b"],
        "requires_key": False,
        "icon": "🏠",
    },
    "Google Gemini": {
        "models": ["gemini-2.0-flash", "gemini-1.5-flash"],
        "requires_key": True,
        "env_key": "GEMINI_API_KEY",
        "icon": "✨",
    },
    "Groq": {
        "models": ["llama-3.3-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"],
        "requires_key": True,
        "env_key": "GROQ_API_KEY",
        "icon": "⚡",
    },
}


def get_available_providers() -> dict:
    """Return providers with their availability status."""
    available = {}
    for name, config in PROVIDERS.items():
        if config["requires_key"]:
            key = os.getenv(config["env_key"], "")
            has_key = bool(key.strip())
        else:
            has_key = True

        available[name] = {
            **config,
            "available": has_key,
        }
    return available


def get_api_key(provider_name: str) -> str:
    """Get the API key for a provider from environment."""
    config = PROVIDERS.get(provider_name, {})
    env_key = config.get("env_key", "")
    return os.getenv(env_key, "")


# -------------------------
# Ollama (Local)
# -------------------------
def generate_ollama(prompt: str, model: str = "gemma3:1b") -> str:
    """Generate a response using local Ollama."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# -------------------------
# Google Gemini (API)
# -------------------------
def generate_gemini(prompt: str, model: str = "gemini-2.0-flash") -> str:
    """Generate a response using Google Gemini API."""
    api_key = get_api_key("Google Gemini")
    if not api_key:
        raise ValueError("Gemini API key not found. Add GEMINI_API_KEY to your .env file.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 4096,
        },
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        error_msg = response.json().get("error", {}).get("message", response.text)
        raise ValueError(f"Gemini API error ({response.status_code}): {error_msg}")

    result = response.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]


# -------------------------
# Groq (API)
# -------------------------
def generate_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Generate a response using Groq API."""
    api_key = get_api_key("Groq")
    if not api_key:
        raise ValueError("Groq API key not found. Add GROQ_API_KEY to your .env file.")

    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4096,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        error_msg = response.json().get("error", {}).get("message", response.text)
        raise ValueError(f"Groq API error ({response.status_code}): {error_msg}")

    result = response.json()
    return result["choices"][0]["message"]["content"]


# -------------------------
# Unified Generation
# -------------------------
def generate(prompt: str, provider: str, model: str) -> str:
    """
    Unified generation function. Routes to the correct provider.
    """
    if provider == "Ollama (Local)":
        return generate_ollama(prompt, model)
    elif provider == "Google Gemini":
        return generate_gemini(prompt, model)
    elif provider == "Groq":
        return generate_groq(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
