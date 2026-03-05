"""src.output — Saída do relatório (console e TTS)."""

from src.output.console_reporter import ConsoleReporter
from src.output.tts_integration import TTSIntegration

__all__ = ["ConsoleReporter", "TTSIntegration"]
