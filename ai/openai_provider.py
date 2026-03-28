"""
OpenAI Provider - Enhanced version for LinkedIn Auto Poster
"""

import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

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
        self.default_model: str = "gpt-4o-mini"
        self.default_temperature: float = 0.85
        self.default_max_tokens: int = 500

    def generate_completion(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
    ) -> str:
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
            sys_msg = (
                system_message
                or "You are a Bengali software developer writing authentic LinkedIn posts."
            )
            resolved_model = model or self.default_model
            resolved_temperature = (
                temperature if temperature is not None else self.default_temperature
            )
            resolved_max_tokens = (
                max_tokens if max_tokens is not None else self.default_max_tokens
            )

            response = self.client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
                top_p=0.92,
                frequency_penalty=0.3,
                presence_penalty=0.2,
            )

            # content can be None per the OpenAI SDK types
            raw_content: Optional[str] = response.choices[0].message.content
            content = (raw_content or "").strip()

            # usage can be None per the OpenAI SDK types
            if response.usage is not None:
                logger.debug(
                    f"OpenAI API call - Model: {resolved_model}, "
                    f"Tokens used: {response.usage.total_tokens}"
                )

            return content

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def generate(self, prompt: str, api_key: Optional[str] = None) -> str:
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
                    max_tokens=512,
                )
                raw: Optional[str] = response.choices[0].message.content
                return (raw or "").strip()
            except Exception as e:
                logger.error(f"OpenAI API response: {e}")
                raise Exception("OpenAI API failed")
        else:
            return self.generate_completion(
                prompt, model="gpt-3.5-turbo", max_tokens=512
            )

    def validate_api_key(self) -> bool:
        """
        Validate API key by making a simple API call

        Returns:
            True if API key is valid, False otherwise
        """
        try:
            self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )
            logger.info("OpenAI API key validation successful")
            return True

        except Exception as e:
            logger.error(f"OpenAI API key validation failed: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """
        Get list of available models

        Returns:
            List of available model names
        """
        try:
            models = self.client.models.list()
            model_names = [m.id for m in models.data if "gpt" in m.id]
            return sorted(model_names)

        except Exception as e:
            logger.error(f"Failed to fetch available models: {e}")
            return ["gpt-3.5-turbo", "gpt-4o-mini"]  # Fallback defaults
