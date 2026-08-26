"""Application configuration for LLM evaluation runs."""

import os
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Config:
    """Store static configuration values used across the application."""

    MODEL_NAMES: ClassVar[set[str]] = {"llama", "gemini", "grok", "gpt", "qwen-coder", 'deepseek', 'deepseek-coder', 'glm'}
    MODELS: ClassVar[dict[str, str]] = {
        "llama": "llama-3.3-70b-versatile",
        "gemini": "gemini-3.1-flash-lite",
        "grok": "grok-4.1-fast",
        "gpt": "gpt-oss:20b",
        "qwen-coder": "qwen2.5-coder:7b",
        'deepseek': 'deepseek-r1:14b',
        'deepseek-coder': "deepseek-coder:6.7b",
        'glm': ' glm-4.7-flash',
    }
    MODEL_PROVIDERS: ClassVar[dict[str, str]] = {
        "llama": "grok",
        "gemini": "gemini",
        "grok": "grok",
        "gpt": "ollama",
        "qwen-coder": "ollama",
        'deepseek': 'ollama',
        'deepseek-coder': "ollama",
        'glm': 'ollama',
    }
    DEFAULT_MODEL_ID: ClassVar[str] = "llama"
    MODEL: ClassVar[str] = MODELS[DEFAULT_MODEL_ID]

    TEMPERATURE: float = 0.2  # 2.0
    INTER_INVOCATIONS_WAITING: float = 0  #  5.0  # seconds between LLM invocations (grok=5s)
    LLM_QUOTA_REACHED_WAITING: float = 24 * 60 * 60
    OLLAMA_HOST: ClassVar[str] = os.environ.get(
        "OLLAMA_HOST",
        "http://127.0.0.1:11434",
    )
    SYSTEM_PROMPT: ClassVar[str] = ""

    N_SAMPLES: int = 100
    N_PROBLEMS: int = 30  # 164 problems in HumanEval
    TEST_TIMEOUT: float = 30.0

    LOG_DIR: str = "logs"

    RESULTS_DIR: str = "results"
    MEASUREMENTS_FILE_NAME: str = "measurements.csv"
    RESULTS_FILE_NAME: str = "results.csv"

    def __post_init__(self) -> None:
        """Validate that all known model ids have configured names."""
        missing_model_ids: set[str] = set(self.MODEL_NAMES) - set(self.MODELS)
        assert not missing_model_ids, (
            "MODEL_NAMES contains ids not present in MODELS: "
            f"{sorted(missing_model_ids)}"
        )
        missing_provider_ids: set[str] = set(self.MODEL_NAMES) - set(
            self.MODEL_PROVIDERS
        )
        assert not missing_provider_ids, (
            "MODEL_NAMES contains ids not present in MODEL_PROVIDERS: "
            f"{sorted(missing_provider_ids)}"
        )

    @classmethod
    def get_model_name(cls, model_id: str) -> str:
        """Return the provider-specific model name for a configured model id."""
        if model_id not in cls.MODEL_NAMES:
            expected_model_ids: list[str] = sorted(cls.MODEL_NAMES)
            raise ValueError(
                f"Unknown model id '{model_id}'. "
                f"Expected one of: {expected_model_ids}"
            )
        return cls.MODELS[model_id]

    @classmethod
    def get_model_provider(cls, model_id: str) -> str:
        """Return the provider client key for a configured model id."""
        if model_id not in cls.MODEL_NAMES:
            expected_model_ids: list[str] = sorted(cls.MODEL_NAMES)
            raise ValueError(
                f"Unknown model id '{model_id}'. "
                f"Expected one of: {expected_model_ids}"
            )
        return cls.MODEL_PROVIDERS[model_id]


config = Config()
