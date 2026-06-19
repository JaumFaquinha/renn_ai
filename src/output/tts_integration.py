"""
TTSIntegration — FASE 6

Integração de voz para feedback do engenheiro de corrida.
Provider configurável via TTS_PROVIDER no .env.

Providers suportados (ordem de prioridade prática para sim-racing):

| Provider     | Custo            | Latência típica  | Offline | Qualidade |
| ------------ | ---------------- | ---------------- | ------- | --------- |
| pyttsx3      | Free             | 50–150ms         | sim     | mid       |
| edge_tts     | Free, sem chave  | 250–500ms        | não     | neural    |
| elevenlabs   | 0,5 créd/char    | ~200–500ms (Flash)| não     | excelente |
| azure        | Free 500k/m      | 200–500ms        | não     | excelente |
| none         | —                | 0                | —       | — (texto) |

Fallback automático: se o provider escolhido falhar (ImportError, sem chave,
sem rede), cai para `pyttsx3` (Windows SAPI5 — declarado no requirements.txt;
requer `pip install pyttsx3`, que puxa `comtypes`. O engine SAPI5 em si é nativo
do Win10/11, mas o binding Python não). Se nem pyttsx3 instalar, cai para "none".

Cada chamada de speak() registra `synthesis_ms` e `audio_ms` em log estruturado
para o benchmark e diagnóstico de latência.

Integração no run_session.py — inalterada:
    tts = TTSIntegration()
    tts.start()
    tts.speak("Perdeu 0.3 segundos na Parabolica. Frenagem tardia.")
    tts.stop()
"""

import logging
import queue
import threading
import time
from typing import Callable, Optional

from config.settings import (
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    TTS_PROVIDER,
)

# Settings opcionais (carregados defensivamente — se faltarem, default sensato)
try:
    from config.settings import (
        TTS_FALLBACK,
        TTS_LANGUAGE,
        TTS_MAX_MESSAGE_CHARS,
        TTS_MIN_INTERVAL_S,
        TTS_VOICE_NAME,
    )
except ImportError:
    TTS_LANGUAGE = "pt-BR"
    TTS_VOICE_NAME = ""
    TTS_FALLBACK = "pyttsx3"
    TTS_MAX_MESSAGE_CHARS = 140
    TTS_MIN_INTERVAL_S = 3.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vozes default por provider (PT-BR otimizado)
# ---------------------------------------------------------------------------
_DEFAULT_EDGE_VOICES_PT_BR = [
    "pt-BR-AntonioNeural",      # masculina, neutra
    "pt-BR-FranciscaNeural",    # feminina, neutra
    "pt-BR-ThalitaMultilingualNeural",  # feminina, expressiva (multilíngue)
]

_VALID_PROVIDERS = {"none", "pyttsx3", "edge_tts", "elevenlabs", "azure"}


class TTSIntegration:
    """
    Wrapper de síntese de voz com fila assíncrona e fallback automático.

    O loop principal (run_session.py @20Hz) NÃO é bloqueado — speak() apenas
    enfileira a mensagem. Síntese e playback acontecem em thread separada.

    Latência observada (lap end → primeira palavra ouvida) por provider:
    - pyttsx3:    ~150–300ms total (síntese + playback offline)
    - edge_tts:   ~400–800ms total (download MP3 + decode + play)
    - elevenlabs: ~200–400ms total (Flash v2.5)
    - azure:      ~300–600ms total

    Cada chamada de speak() emite logs estruturados com timing real para
    monitoramento contínuo via TensorBoard ou stdout.
    """

    def __init__(self, provider: str = TTS_PROVIDER) -> None:
        if provider not in _VALID_PROVIDERS:
            logger.warning(
                "Provider TTS desconhecido — fallback para none",
                extra={"provider": provider},
            )
            provider = "none"

        self._provider = provider
        self._effective_provider = provider  # pode mudar via fallback
        self._fallback_provider = TTS_FALLBACK
        self._language = TTS_LANGUAGE
        self._voice_name = TTS_VOICE_NAME
        self._max_chars = TTS_MAX_MESSAGE_CHARS
        self._min_interval_s = TTS_MIN_INTERVAL_S
        self._last_spoken_at: float = 0.0

        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False

        # Handle do engine inicializado preguiçosamente na thread worker
        # (alguns providers ex: pyttsx3 prendem o objeto à thread criadora)
        self._engine_ready: bool = False
        self._synthesizer: Optional[Callable[[str], float]] = None
        # Referência ao engine pyttsx3 — permite que stop() interrompa um
        # playback longo em andamento (ver _init_pyttsx3 / stop).
        self._pyttsx3_engine: Optional[object] = None

        if self._provider != "none":
            self._validate()

        logger.info(
            "TTSIntegration inicializada",
            extra={
                "provider": self._provider,
                "language": self._language,
                "fallback": self._fallback_provider,
            },
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Inicia a thread de processamento de TTS."""
        if self._provider == "none":
            return

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="tts-worker")
        self._thread.start()
        logger.debug("Thread TTS iniciada")

    def stop(self) -> None:
        """Para a thread de TTS graciosamente. Drena a fila por até 5s."""
        self._running = False
        # Best-effort: interrompe um playback em andamento (ex.: pyttsx3 no meio
        # de uma fala longa) para o join abaixo não estourar o timeout. A chamada
        # é cross-thread (COM/SAPI5), por isso protegida — no pior caso é no-op.
        if self._pyttsx3_engine is not None:
            try:
                self._pyttsx3_engine.stop()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._queue.put(None)  # Poison pill
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.debug("Thread TTS encerrada")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def speak(self, message: str, priority: bool = False) -> None:
        """
        Enfileira uma mensagem para síntese de voz.

        - Não bloqueia: queue.put é O(1).
        - Aplica truncamento (TTS_MAX_MESSAGE_CHARS) — mensagens longas
          aumentam TTFB de síntese; engenheiro de corrida é direto e curto.
        - Aplica cooldown (TTS_MIN_INTERVAL_S): evita falar 2 alertas em
          janela curta — usuário não consegue processar overlap.

        Args:
            message: texto a falar.
            priority: se True, ignora o cooldown — para alertas que não podem
                ser suprimidos (ex.: sinal de volta inválida).
        """
        if not message:
            return

        truncated = self._truncate(message)

        if self._provider == "none":
            logger.info("[TTS] %s", truncated)
            return

        now = time.monotonic()
        if not priority and (now - self._last_spoken_at) < self._min_interval_s:
            logger.debug(
                "TTS suprimido por cooldown",
                extra={"since_last_s": round(now - self._last_spoken_at, 2)},
            )
            return
        self._last_spoken_at = now

        self._queue.put(truncated)

    # ------------------------------------------------------------------
    # Worker (thread separada)
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """Processa mensagens da fila em thread separada com instrumentação."""
        # Inicialização preguiçosa do engine (deve acontecer NESTA thread
        # para providers como pyttsx3 que ligam o engine ao thread-id criador)
        if not self._engine_ready:
            self._initialize_engine()

        while self._running:
            message = self._queue.get()
            if message is None:
                break

            t_start = time.monotonic()
            try:
                synthesis_ms = self._synthesize(message)
                total_ms = (time.monotonic() - t_start) * 1000
                audio_ms = total_ms - synthesis_ms
                logger.info(
                    "TTS spoken",
                    extra={
                        "provider": self._effective_provider,
                        "chars": len(message),
                        "synthesis_ms": round(synthesis_ms, 1),
                        "audio_ms": round(max(0.0, audio_ms), 1),
                        "total_ms": round(total_ms, 1),
                    },
                )
            except Exception as exc:
                logger.error(
                    "Erro na síntese de voz",
                    extra={"error": str(exc), "provider": self._effective_provider},
                )
                # Tentativa de fallback uma única vez por mensagem
                if self._effective_provider != self._fallback_provider:
                    logger.info(
                        "Tentando fallback TTS",
                        extra={"from": self._effective_provider, "to": self._fallback_provider},
                    )
                    self._effective_provider = self._fallback_provider
                    self._engine_ready = False
                    self._initialize_engine()

    # ------------------------------------------------------------------
    # Engine initialization (provider switch)
    # ------------------------------------------------------------------

    def _initialize_engine(self) -> None:
        """Inicializa o synthesizer do provider efetivo. Cai para pyttsx3
        se a inicialização falhar; cai para 'none' se nem pyttsx3 instalar."""
        provider = self._effective_provider
        synthesizer: Optional[Callable[[str], float]] = None

        if provider == "pyttsx3":
            synthesizer = self._init_pyttsx3()
        elif provider == "edge_tts":
            synthesizer = self._init_edge_tts()
        elif provider == "elevenlabs":
            synthesizer = self._init_elevenlabs()
        elif provider == "azure":
            synthesizer = self._init_azure()

        if synthesizer is None and provider != self._fallback_provider:
            logger.warning(
                "Inicialização do provider falhou — fallback",
                extra={"from": provider, "to": self._fallback_provider},
            )
            self._effective_provider = self._fallback_provider
            if self._fallback_provider == "pyttsx3":
                synthesizer = self._init_pyttsx3()

        if synthesizer is None:
            logger.warning("Nenhum provider TTS disponível — degradando para 'none'")
            self._effective_provider = "none"

        self._synthesizer = synthesizer
        self._engine_ready = True

    def _synthesize(self, message: str) -> float:
        """Despacha para o synthesizer. Retorna tempo de síntese em ms."""
        if self._synthesizer is None or self._effective_provider == "none":
            logger.info("[TTS-fallback-text] %s", message)
            return 0.0
        return self._synthesizer(message)

    # ------------------------------------------------------------------
    # Provider: pyttsx3 (Windows SAPI5 — offline, sempre disponível)
    # ------------------------------------------------------------------

    def _init_pyttsx3(self) -> Optional[Callable[[str], float]]:
        try:
            import pyttsx3  # type: ignore
        except ImportError:
            logger.warning(
                "pyttsx3 não instalado — execute: pip install pyttsx3",
            )
            return None

        try:
            engine = pyttsx3.init()
            # Seleção de voz em dois passes — a intenção explícita do usuário
            # (TTS_VOICE_NAME) tem prioridade sobre a heurística de idioma:
            #   1) match pelo nome pedido (ex.: "Daniel")
            #   2) fallback heurístico pt-BR por nome (ex.: "Microsoft Maria
            #      Desktop - Portuguese(Brazil)")
            # Obs.: o driver SAPI5 do pyttsx3 não popula `voice.languages` no
            # Windows (lista vazia), por isso o match é feito apenas pelo nome.
            voices = engine.getProperty("voices") or []

            def _select_voice():
                if self._voice_name:
                    wanted = self._voice_name.lower()
                    for v in voices:
                        if wanted in (getattr(v, "name", "") or "").lower():
                            return v
                for v in voices:
                    vname = (getattr(v, "name", "") or "").lower()
                    if "portuguese" in vname or "pt-br" in vname or "português" in vname:
                        return v
                return None

            target = _select_voice()
            if target is not None:
                engine.setProperty("voice", target.id)
                logger.info("pyttsx3 voz selecionada", extra={"voice": target.name})
            elif self._voice_name:
                logger.warning(
                    "pyttsx3 voz pedida não encontrada — usando voz padrão do sistema",
                    extra={"requested": self._voice_name},
                )
            engine.setProperty("rate", 190)  # ~190wpm — engenheiro objetivo
        except Exception as exc:
            logger.warning("pyttsx3 init falhou", extra={"error": str(exc)})
            return None

        # Exposto para stop() poder interromper um playback longo (cross-thread).
        self._pyttsx3_engine = engine

        def synth(message: str) -> float:
            t0 = time.monotonic()
            engine.say(message)
            try:
                engine.runAndWait()  # bloqueia ATÉ o áudio terminar — síncrono no SAPI5
            except RuntimeError:
                # Bug conhecido do pyttsx3: "run loop already started" quando um
                # loop anterior não foi encerrado. Encerra o loop preso e tenta
                # uma vez mais antes de propagar o erro ao worker.
                try:
                    engine.endLoop()
                except Exception:
                    pass
                engine.runAndWait()
            # No SAPI5, runAndWait inclui síntese E playback. Não dá para separar
            # sem reescrever via stream — reportamos o total como synthesis_ms.
            return (time.monotonic() - t0) * 1000

        return synth

    # ------------------------------------------------------------------
    # Provider: edge-tts (Microsoft Edge — free, neural, online)
    # ------------------------------------------------------------------

    def _init_edge_tts(self) -> Optional[Callable[[str], float]]:
        try:
            import asyncio  # noqa: F401  (usado dentro do synth)
            import edge_tts  # type: ignore
        except ImportError:
            logger.warning(
                "edge-tts não instalado — execute: pip install edge-tts",
            )
            return None

        try:
            import sounddevice  # noqa: F401
            import soundfile  # noqa: F401
        except ImportError:
            logger.warning(
                "sounddevice/soundfile não instalados — execute: "
                "pip install sounddevice soundfile",
            )
            return None

        voice = self._voice_name or _DEFAULT_EDGE_VOICES_PT_BR[0]
        logger.info("edge-tts voz selecionada", extra={"voice": voice})

        def synth(message: str) -> float:
            import asyncio
            import io

            import edge_tts
            import sounddevice as sd
            import soundfile as sf

            async def _stream_to_buffer() -> bytes:
                buf = io.BytesIO()
                communicate = edge_tts.Communicate(message, voice)
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        buf.write(chunk["data"])
                return buf.getvalue()

            t0 = time.monotonic()
            audio_bytes = asyncio.run(_stream_to_buffer())
            t_synth = (time.monotonic() - t0) * 1000

            data, sr = sf.read(io.BytesIO(audio_bytes))
            sd.play(data, sr)
            sd.wait()
            return t_synth

        return synth

    # ------------------------------------------------------------------
    # Provider: ElevenLabs (Flash v2.5 — premium, online, voz clonada)
    # ------------------------------------------------------------------

    def _init_elevenlabs(self) -> Optional[Callable[[str], float]]:
        if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
            logger.warning("ElevenLabs sem credenciais — pulando init")
            return None

        try:
            from elevenlabs.client import ElevenLabs  # type: ignore
        except ImportError:
            logger.warning(
                "elevenlabs não instalado — execute: pip install elevenlabs",
            )
            return None

        try:
            import sounddevice  # noqa: F401
            import soundfile  # noqa: F401
        except ImportError:
            logger.warning(
                "sounddevice/soundfile não instalados — execute: "
                "pip install sounddevice soundfile",
            )
            return None

        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        def synth(message: str) -> float:
            import io

            import sounddevice as sd
            import soundfile as sf

            t0 = time.monotonic()
            # Flash v2.5 — modelo mais rápido do ElevenLabs. Os ~75ms divulgados
            # são tempo de INFERÊNCIA no servidor; o end-to-end percebido fica em
            # ~200–500ms somando rede + decode de MP3 + playback. PCM (pcm_22050)
            # eliminaria o decode, mas é headerless e exigiria enquadrar o buffer
            # manualmente — overkill para um relatório curto de fim de volta.
            audio_iter = client.text_to_speech.convert(
                voice_id=ELEVENLABS_VOICE_ID,
                model_id="eleven_flash_v2_5",
                output_format="mp3_44100_128",
                text=message,
            )
            audio_bytes = b"".join(audio_iter)
            t_synth = (time.monotonic() - t0) * 1000

            # Observabilidade de custo: Flash bilha 0,5 crédito/caractere. Logamos
            # a estimativa determinística — o header `character-cost` da resposta
            # seria a fonte autoritativa, mas o SDK abstrai o response no caminho
            # de streaming do convert().
            logger.info(
                "ElevenLabs síntese",
                extra={
                    "chars": len(message),
                    "estimated_credits": round(len(message) * 0.5, 1),
                },
            )

            data, sr = sf.read(io.BytesIO(audio_bytes))
            sd.play(data, sr)
            sd.wait()
            return t_synth

        return synth

    # ------------------------------------------------------------------
    # Provider: Azure Cognitive Services Speech
    # ------------------------------------------------------------------

    def _init_azure(self) -> Optional[Callable[[str], float]]:
        if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
            logger.warning("Azure Speech sem credenciais — pulando init")
            return None

        try:
            import azure.cognitiveservices.speech as speechsdk  # type: ignore
        except ImportError:
            logger.warning(
                "azure-cognitiveservices-speech não instalado — execute: "
                "pip install azure-cognitiveservices-speech",
            )
            return None

        try:
            speech_config = speechsdk.SpeechConfig(
                subscription=AZURE_SPEECH_KEY,
                region=AZURE_SPEECH_REGION,
            )
            voice = self._voice_name or "pt-BR-AntonioNeural"
            speech_config.speech_synthesis_voice_name = voice
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            logger.info("Azure TTS voz selecionada", extra={"voice": voice})
        except Exception as exc:
            logger.warning("Azure init falhou", extra={"error": str(exc)})
            return None

        def synth(message: str) -> float:
            t0 = time.monotonic()
            future = synthesizer.speak_text_async(message)
            result = future.get()  # bloqueia até síntese + playback
            t_total = (time.monotonic() - t0) * 1000

            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.warning(
                    "Azure síntese não completou",
                    extra={"reason": str(result.reason)},
                )
            return t_total

        return synth

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate(self, message: str) -> str:
        """Trunca a mensagem para reduzir TTFB de síntese.

        Engenheiro de corrida fala curto: '0,3s perdidos na Parabolica,
        frenagem tardia' (52 chars) é o tom alvo.
        """
        if len(message) <= self._max_chars:
            return message
        cut = message[: self._max_chars].rsplit(" ", 1)[0]
        return cut + "..."

    def _validate(self) -> None:
        """Validação de credenciais para providers que exigem chave.
        Não falha — apenas loga warning. A inicialização real acontece
        preguiçosamente na thread worker para evitar bloqueio aqui."""
        if self._provider == "elevenlabs":
            if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
                logger.warning(
                    "ELEVENLABS_API_KEY/VOICE_ID não configurados — fallback será usado"
                )
        elif self._provider == "azure":
            if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
                logger.warning(
                    "AZURE_SPEECH_KEY/REGION não configurados — fallback será usado"
                )
