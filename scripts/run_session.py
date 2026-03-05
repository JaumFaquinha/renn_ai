"""
run_session.py — Entrypoint principal do Engenheiro de Corrida IA.

Inicia o loop de leitura de telemetria a 20Hz, grava voltas em JSON
e exibe análise no terminal ao cruzar a linha de chegada.

Uso:
    python scripts/run_session.py [--track monza] [--rate 20]

Requer:
    - Assetto Corsa rodando em Windows
    - Shared Memory habilitada nas opções do jogo
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

# Adiciona o root do projeto ao path para imports relativos funcionarem
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import LOG_LEVEL, SAMPLING_RATE_HZ
from src.analysis.lap_analyzer import LapAnalyzer
from src.analysis.pattern_detector import PatternDetector
from src.analysis.report_builder import ReportBuilder
from src.memory.shared_memory_reader import SharedMemoryReader, snapshot_to_dict
from src.output.console_reporter import ConsoleReporter
from src.output.tts_integration import TTSIntegration
from src.recording.lap_recorder import LapRecorder

# --- Logging ---
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_session")

_RUNNING = True


def _signal_handler(sig, frame) -> None:
    global _RUNNING
    logger.info("Sinal de interrupção recebido — encerrando...")
    _RUNNING = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Engenheiro de Corrida IA — Sessão de Telemetria")
    parser.add_argument("--track", default="monza", help="ID da pista (default: monza)")
    parser.add_argument(
        "--rate", type=int, default=SAMPLING_RATE_HZ,
        help=f"Frequência de amostragem em Hz (default: {SAMPLING_RATE_HZ})",
    )
    return parser.parse_args()


def run(track_id: str, rate_hz: int) -> None:
    """Loop principal de coleta e análise de telemetria."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    interval_s = 1.0 / rate_hz
    lap_count = 0
    best_lap_data = None

    recorder = LapRecorder(track_id=track_id)
    analyzer = LapAnalyzer()
    detector = PatternDetector()
    reporter = ConsoleReporter()
    tts = TTSIntegration()
    report_builder = ReportBuilder(track_id=track_id)

    tts.start()

    logger.info(
        "Iniciando sessão",
        extra={"track": track_id, "rate_hz": rate_hz},
    )
    logger.info("Aguardando Assetto Corsa...")

    with SharedMemoryReader() as reader:
        # Tentar conectar em loop até o AC iniciar
        while _RUNNING and not reader.is_connected:
            reader.connect()
            if not reader.is_connected:
                time.sleep(2.0)

        if not reader.is_connected:
            logger.info("Sessão encerrada antes de conectar ao AC.")
            return

        # Ler dados estáticos da sessão
        static = reader.read_static()
        detector = PatternDetector(max_rpm=static.maxRpm or 8000)
        logger.info(
            "Conectado ao AC",
            extra={
                "car": static.carModel,
                "track": static.track,
                "track_length_m": static.trackSPlineLength,
            },
        )

        while _RUNNING:
            t_start = time.monotonic()

            snapshot = reader.read()
            if not snapshot.read_ok:
                time.sleep(interval_s)
                continue

            snap_dict = snapshot_to_dict(snapshot)
            completed_lap = recorder.process_snapshot(snap_dict)

            if completed_lap is not None:
                lap_count += 1

                # Precisa de pelo menos 2 voltas para análise
                if best_lap_data is None:
                    best_lap_data = completed_lap
                    analyzer.set_best_lap(best_lap_data)
                    logger.info("Primeira volta gravada — aguardando segunda volta para comparação.")
                else:
                    # Analisar
                    analyzed = analyzer.analyze(completed_lap["mini_sectors"])
                    pattern_results = {
                        i: detector.detect(sector)
                        for i, sector in enumerate(analyzed)
                    }
                    top = analyzer.top_loss_sectors(analyzed)
                    top_patterns = {
                        i: pattern_results[original_idx]
                        for i, (original_idx, _) in enumerate(
                            sorted(
                                [(i, s) for i, s in enumerate(analyzed)],
                                key=lambda x: x[1].get("delta_vs_best", 0.0),
                                reverse=True,
                            )[:5]
                        )
                    }
                    lap_report = report_builder.build(
                        lap_number=completed_lap["lap_number"],
                        lap_time_ms=completed_lap["lap_time_ms"],
                        analyzed_sectors=analyzed,
                        pattern_results={i: detector.detect(s) for i, s in enumerate(analyzed)},
                    )
                    reporter.report(lap_report)

                    # Atualizar melhor volta se necessário
                    if completed_lap["lap_time_ms"] < best_lap_data["lap_time_ms"]:
                        best_lap_data = completed_lap
                        analyzer.set_best_lap(best_lap_data)
                        logger.info("Nova melhor volta! Referência atualizada.")

                    # Feedback de voz (top perda)
                    if lap_report.top_sectors:
                        top_sector = lap_report.top_sectors[0]
                        zone = top_sector.corner_name or f"spline {top_sector.track_position:.2f}"
                        cause = top_sector.causes[0].cause if top_sector.causes else "perda não identificada"
                        tts.speak(
                            f"Perdeu {top_sector.delta_vs_best_s:.2f} segundos em {zone}. {cause}."
                        )

            # Manter frequência de amostragem
            elapsed = time.monotonic() - t_start
            sleep_time = interval_s - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    tts.stop()
    logger.info("Sessão encerrada", extra={"voltas_completadas": lap_count})


if __name__ == "__main__":
    args = parse_args()
    run(track_id=args.track, rate_hz=args.rate)
