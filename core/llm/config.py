"""
============================================================
NOONGIL-X
LLM Configuration Manager
============================================================

Author : NOONGIL-X
Purpose:
Central configuration system for every LLM module.

Features
--------
✓ Automatic project root detection
✓ JSON configuration loading
✓ Default values
✓ Path management
✓ Ollama configuration
✓ Future OpenAI support
✓ Future Gemini support
✓ Logging configuration
✓ Generation parameters
✓ Helper methods
============================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORE_DIR = PROJECT_ROOT / "core"

LLM_DIR = CORE_DIR / "llm"

PROMPT_DIR = LLM_DIR / "prompt"

OUTPUT_DIR = PROJECT_ROOT / "output"

CACHE_DIR = OUTPUT_DIR / "llm_cache"

MEMORY_DIR = OUTPUT_DIR / "memory"

LOG_DIR = OUTPUT_DIR / "logs"

CONFIG_FILE = LLM_DIR / "llm_config.json"


# ============================================================
# Default Configuration
# ============================================================

DEFAULT_CONFIG = {

    "backend": "ollama",

    "model": "llama3.2:3b",

    "temperature": 0.3,

    "top_p": 0.9,

    "top_k": 40,

    "repeat_penalty": 1.1,

    "num_predict": 512,

    "context_window": 8192,

    "request_timeout": 120,

    "keep_alive": "30m",

    "stream": False,

    "cache": True,

    "memory": True,

    "json_mode": True,

    "verbose": True,

    "host": "http://localhost:11434",

    "max_history": 15
}


# ============================================================
# Dataclass
# ============================================================

@dataclass
class LLMConfig:

    backend: str = "ollama"

    model: str = "llama3.2:3b"

    temperature: float = 0.3

    top_p: float = 0.9

    top_k: int = 40

    repeat_penalty: float = 1.1

    num_predict: int = 512

    context_window: int = 8192

    request_timeout: int = 120

    keep_alive: str = "30m"

    stream: bool = False

    cache: bool = True

    memory: bool = True

    json_mode: bool = True

    verbose: bool = True

    host: str = "http://localhost:11434"

    max_history: int = 15

    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Config Manager
# ============================================================

class ConfigManager:

    def __init__(self):

        self.config_path = CONFIG_FILE

        self.config = self.load()

    # --------------------------------------------------------

    def create_default(self):

        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:

            json.dump(DEFAULT_CONFIG, f, indent=4)

    # --------------------------------------------------------

    def load(self) -> LLMConfig:

        if not self.config_path.exists():

            self.create_default()

        with open(self.config_path, "r", encoding="utf-8") as f:

            data = json.load(f)

        merged = DEFAULT_CONFIG.copy()

        merged.update(data)

        known = {
            k: merged.pop(k)
            for k in list(DEFAULT_CONFIG.keys())
        }

        return LLMConfig(
            **known,
            extra=merged
        )

    # --------------------------------------------------------

    def reload(self):

        self.config = self.load()

    # --------------------------------------------------------

    def save(self):

        data = self.to_dict()

        with open(self.config_path, "w", encoding="utf-8") as f:

            json.dump(data, f, indent=4)

    # --------------------------------------------------------

    def to_dict(self):

        d = vars(self.config).copy()

        extra = d.pop("extra")

        d.update(extra)

        return d

    # --------------------------------------------------------

    def update(self, **kwargs):

        for key, value in kwargs.items():

            if hasattr(self.config, key):

                setattr(self.config, key, value)

            else:

                self.config.extra[key] = value

        self.save()

    # --------------------------------------------------------

    @property
    def backend(self):

        return self.config.backend

    @property
    def model(self):

        return self.config.model

    @property
    def host(self):

        return self.config.host

    @property
    def temperature(self):

        return self.config.temperature

    @property
    def json_mode(self):

        return self.config.json_mode

    @property
    def timeout(self):

        return self.config.request_timeout


# ============================================================
# Singleton
# ============================================================

config = ConfigManager()


# ============================================================
# Ensure Directories Exist
# ============================================================

for directory in [
    OUTPUT_DIR,
    CACHE_DIR,
    MEMORY_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print("NOONGIL LLM CONFIGURATION")

    print("=" * 60)

    print()

    print(config.to_dict())

    print()

    print("Project Root :", PROJECT_ROOT)

    print("LLM Folder   :", LLM_DIR)

    print("Prompt Folder:", PROMPT_DIR)

    print("Output Folder:", OUTPUT_DIR)

    print()

    print("Configuration Loaded Successfully")