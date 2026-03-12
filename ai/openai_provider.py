"""
OpenAI Provider - Enhanced version for LinkedIn Auto Poster
"""

import os
import logging
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class OpenAIProvider:
    """Enhanced OpenAI provider with better error handling and flexibility"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Default parameters
        self.default_model = "gpt-4o-mini"
        self.default_temperature = 0.85
        self.default_max_tokens = 500
    
    def generate_completion(self, prompt: str, temperature: float = None, 
                          max_tokens: int = None, model: str = None,
                          system_message: str = None) -> str:
        """
        Generate completion using OpenAI API
        
        Args:
            prompt: The input prompt
            temperature: Creativity level (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            model: Model to use
            system_message: Custom system message (optional)
            
        Returns:
            Generated text completion
        """
        try:
            sys_msg = system_message or "You are a Bengali software developer writing authentic LinkedIn posts."
            
            response = self.client.chat.completions.create(
                model=model or self.default_model,
                messages=[
                    {
                        "role": "system", 
                        "content": sys_msg
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                top_p=0.92,
                frequency_penalty=0.3,
                presence_penalty=0.2,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Log usage for monitoring
            logger.debug(f"OpenAI API call - Model: {model or self.default_model}, "
                        f"Tokens used: {response.usage.total_tokens}")
            
            return content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def generate(self, prompt: str, api_key: str = None) -> str:
        """
        Legacy method for backward compatibility
        """
        if api_key and api_key != self.api_key:
            # Create temporary client with different key
            temp_client = OpenAI(api_key=api_key)
            try:
                response = temp_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI API response: {e}")
                raise Exception("OpenAI API failed")
        else:
            return self.generate_completion(prompt, model="gpt-3.5-turbo", max_tokens=512)
    
    def validate_api_key(self) -> bool:
        """
        Validate API key by making a simple API call
        
        Returns:
            True if API key is valid, False otherwise
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            logger.info("OpenAI API key validation successful")
            return True
            
        except Exception as e:
            logger.error(f"OpenAI API key validation failed: {e}")
            return False
    
    def get_available_models(self) -> list:
        """
        Get list of available models
        
        Returns:
            List of available model names
        """
        try:
            models = self.client.models.list()
            model_names = [model.id for model in models.data if 'gpt' in model.id]
            return sorted(model_names)
            
        except Exception as e:
            logger.error(f"Failed to fetch available models: {e}")
            return ["gpt-3.5-turbo", "gpt-4o-mini"]  # Fallback defaults
