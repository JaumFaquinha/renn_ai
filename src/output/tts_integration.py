"""
TTSIntegration — FASE 6 (Placeholder)

Integração de voz para feedback do engenheiro de corrida.
Provider configurável via TTS_PROVIDER no .env.

Providers suportados:
- "none"       : modo texto apenas (padrão)
- "elevenlabs" : ElevenLabs API (requer ELEVENLABS_API_KEY)
- "azure"      : Azure Cognitive Services TTS (requer AZURE_SPEECH_KEY)

Este módulo é um placeholder funcional — retorna imediatamente em modo
"none" sem bloquear o loop principal.
"""

import logging
import queue
import threading
from typing import Optional

from config.settings import (
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    TTS_PROVIDER,
)

logger = logging.getLogger(__name__)


class TTSIntegration:
    """
    Wrapper de síntese de voz com fila assíncrona.

    Não bloqueia o loop principal. Mensagens são enfileiradas e
    processadas em thread separada.

    Uso:
        tts = TTSIntegration()
        tts.start()
        tts.speak("Perdeu 0.3 segundos na Parabolica. Frenagem tardia.")
        tts.stop()
    """

    def __init__(self, provider: str = TTS_PROVIDER) -> None:
        self._provider = provider
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False

        logger.info("TTSIntegration inicializada", extra={"provider": provider})

        if provider == "none":
            logger.info("TTS desabilitado (provider=none) — modo texto apenas")
        elif provider == "elevenlabs":
            self._validate_elevenlabs()
        elif provider == "azure":
            self._validate_azure()
        else:
            logger.warning("Provider TTS desconhecido — fallback para none", extra={"provider": provider})
            self._provider = "none"

    def start(self) -> None:
        """Inicia a thread de processamento de TTS."""
        if self._provider == "none":
            return

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="tts-worker")
        self._thread.start()
        logger.debug("Thread TTS iniciada")

    def stop(self) -> None:
        """Para a thread de TTS graciosamente."""
        self._running = False
        self._queue.put(None)  # Poison pill
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.debug("Thread TTS encerrada")

    def speak(self, message: str) -> None:
        """
        Enfileira uma mensagem para síntese de voz.

        Args:
            message: texto a ser sintetizado.
        """
        if self._provider == "none":
            logger.info("[TTS] %s", message)
            return

        self._queue.put(message)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """Processa mensagens da fila em thread separada."""
        while self._running:
            message = self._queue.get()
            if message is None:
                break
            try:
                self._synthesize(message)
            except Exception as exc:
                logger.error("Erro na síntese de voz", extra={"error": str(exc)})

    def _synthesize(self, message: str) -> None:
        """Despacha para o provider correto."""
        if self._provider == "elevenlabs":
            self._speak_elevenlabs(message)
        elif self._provider == "azure":
            self._speak_azure(message)

    def _speak_elevenlabs(self, message: str) -> None:
        """Síntese via ElevenLabs API — implementar na Fase 6."""
        # TODO: Fase 6
        # from elevenlabs import generate, play
        # audio = generate(text=message, voice=ELEVENLABS_VOICE_ID, api_key=ELEVENLABS_API_KEY)
        # play(audio)
        logger.info("[TTS-ElevenLabs placeholder] %s", message)

    def _speak_azure(self, message: str) -> None:
        """Síntese via Azure Cognitive Services — implementar na Fase 6."""
        # TODO: Fase 6
        # import azure.cognitiveservices.speech as speechsdk
        # config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        # synthesizer = speechsdk.SpeechSynthesizer(speech_config=config)
        # synthesizer.speak_text_async(message).get()
        logger.info("[TTS-Azure placeholder] %s", message)

    # ------------------------------------------------------------------
    # Validação de credenciais
    # ------------------------------------------------------------------

    def _validate_elevenlabs(self) -> None:
        if not ELEVENLABS_API_KEY:
            logger.warning("ELEVENLABS_API_KEY não configurada — TTS desabilitado")
            self._provider = "none"
        if not ELEVENLABS_VOICE_ID:
            logger.warning("ELEVENLABS_VOICE_ID não configurada — TTS desabilitado")
            self._provider = "none"

    def _validate_azure(self) -> None:
        if not AZURE_SPEECH_KEY:
            logger.warning("AZURE_SPEECH_KEY não configurada — TTS desabilitado")
            self._provider = "none"
        if not AZURE_SPEECH_REGION:
            logger.warning("AZURE_SPEECH_REGION não configurada — TTS desabilitado")
            self._provider = "none"
