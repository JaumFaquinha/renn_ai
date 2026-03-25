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

from config.settings import LOG_LEVEL, MODELS_DIR, SAMPLING_RATE_HZ
from src.analysis.lap_analyzer import LapAnalyzer
from src.analysis.pattern_detector import PatternDetector
from src.analysis.report_builder import ReportBuilder
from src.memory.shared_memory_reader import SharedMemoryReader, snapshot_to_dict
from src.models.sector_model import SectorModel
from src.output.console_reporter import ConsoleReporter
from src.output.tts_integration import TTSIntegration
from src.persistence.supabase_client import SupabaseClient
from src.persistence.lap_uploader import LapUploader
from src.persistence.query_service import QueryService
from src.recording.lap_recorder import LapRecorder

# --- Logging ---
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_session")

_RUNNING = True

# Mapeamento do int de SPageFileGraphic.session para string canônica
_SESSION_TYPE_MAP: dict[int, str] = {
    0: "practice",
    1: "qualifying",
    2: "race",
    3: "hotlap",
    4: "time_attack",
    5: "drift",
    6: "drag",
}


def _session_int_to_str(session_int: int) -> str:
    """Converte SPageFileGraphic.session (int) para string canônica de sessão."""
    return _SESSION_TYPE_MAP.get(session_int, "unknown")


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
    session_id = None

    # --- Inicializar persistência (no-op se SUPABASE_ENABLED=false) ---
    sb_client = SupabaseClient()
    uploader = LapUploader(sb_client)
    query_service = QueryService(sb_client)

    if sb_client.is_enabled:
        sb_client.health_check()

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
            tts.stop()
            return

        # Ler dados estáticos da sessão (uma vez)
        static = reader.read_static()
        car_model = static.carModel or "unknown"

        # Lê primeiro snapshot antes de criar o recorder para capturar
        # session_type dinamicamente (SPageFileGraphic.session) e temperaturas iniciais.
        first_snap = reader.read()
        initial_air_temp = 0.0
        initial_road_temp = 0.0
        session_type_str = "unknown"
        if first_snap.read_ok:
            session_type_str = _session_int_to_str(first_snap.graphics.session)
            initial_air_temp = first_snap.physics.airTemp
            initial_road_temp = first_snap.physics.roadTemp

        detector = PatternDetector(max_rpm=static.maxRpm or 8000)

        # Criar recorder com metadados completos
        recorder = LapRecorder(
            track_id=track_id,
            car_model=car_model,
            session_type=session_type_str,
        )

        # Carregar SectorModel treinado (se disponível para esta pista)
        sector_model = SectorModel(track_id=track_id)
        model_path = MODELS_DIR / f"{track_id}.pkl"
        if sector_model.load(str(model_path)):
            logger.info(
                "SectorModel carregado",
                extra={"track_id": track_id, "n_sectors": sector_model.n_training_sectors},
            )
        else:
            logger.info(
                "SectorModel não encontrado para esta pista — sem scores de anomalia. "
                "Execute scripts/train_model.py --track %s após coletar voltas.",
                track_id,
            )

        logger.info(
            "Conectado ao AC",
            extra={
                "car": car_model,
                "track": static.track,
                "track_length_m": static.trackSPlineLength,
                "session_type": session_type_str,
            },
        )

        session_id = uploader.create_session(
            track_id=track_id,
            car_model=car_model,
            session_type=session_type_str,
            air_temp=initial_air_temp,
            road_temp=initial_road_temp,
        )

        # Carregar personal best histórico do Supabase (Fase 7C)
        historical_pb = query_service.get_personal_best(
            track_id=track_id,
            car_model=car_model,
        )
        if historical_pb:
            logger.info(
                "Personal best histórico carregado: %dms",
                historical_pb["lap_time_ms"],
                extra={"session_date": historical_pb.get("session_date")},
            )
            analyzer.set_best_lap({"mini_sectors": historical_pb["mini_sectors"]})
            best_lap_data = historical_pb

        try:
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
                    current_lap_time = completed_lap["lap_time_ms"]

                    # Determinar se é nova melhor volta (sessão e histórico)
                    is_session_best = (
                        best_lap_data is None
                        or current_lap_time < best_lap_data["lap_time_ms"]
                    )
                    # all-time best: só relevante se temos histórico do Supabase
                    is_alltime_best = is_session_best and historical_pb is not None and (
                        current_lap_time < historical_pb["lap_time_ms"]
                    )

                    if best_lap_data is None:
                        # Primeira volta sem histórico: usá-la como referência
                        best_lap_data = completed_lap
                        analyzer.set_best_lap(best_lap_data)
                        logger.info("Primeira volta gravada — aguardando segunda volta para comparação.")
                        # Enfileirar mesmo a primeira volta (dados históricos)
                        uploader.enqueue_lap(
                            completed_lap=completed_lap,
                            lap_report=None,
                            is_session_best=True,
                            is_alltime_best=historical_pb is None,  # PB se não há histórico
                        )
                    else:
                        # Analisar vs melhor volta de referência
                        analyzed = analyzer.analyze(completed_lap["mini_sectors"])

                        # Calcular padrões uma única vez (corrige Bug 1)
                        pattern_results = {
                            i: detector.detect(sector)
                            for i, sector in enumerate(analyzed)
                        }

                        # Scores de anomalia do SectorModel (em lote — 1 chamada numpy)
                        model_scores: dict[int, float] = {}
                        if sector_model.is_trained:
                            batch_scores = sector_model.predict_batch(analyzed)
                            model_scores = dict(enumerate(batch_scores))

                        lap_report = report_builder.build(
                            lap_number=completed_lap["lap_number"],
                            lap_time_ms=current_lap_time,
                            analyzed_sectors=analyzed,
                            pattern_results=pattern_results,
                            model_scores=model_scores,
                        )

                        # Busca histórico de delta por setor (Fase 7D)
                        # Chamada síncrona mas fora do loop 20Hz — aceitável
                        sector_history = {}
                        if lap_report.top_sectors:
                            positions = [s.track_position for s in lap_report.top_sectors]
                            sector_history = query_service.get_sectors_history_batch(
                                track_id=track_id,
                                car_model=car_model,
                                positions=positions,
                            )

                        reporter.report(lap_report, sector_history=sector_history)

                        # Enfileirar upload assíncrono
                        uploader.enqueue_lap(
                            completed_lap=completed_lap,
                            lap_report=lap_report,
                            is_session_best=is_session_best,
                            is_alltime_best=is_alltime_best,
                        )

                        # Atualizar referência se nova melhor volta
                        if is_session_best:
                            best_lap_data = completed_lap
                            analyzer.set_best_lap(best_lap_data)
                            logger.info(
                                "Nova melhor volta! Referência atualizada.",
                                extra={"lap_time_ms": current_lap_time},
                            )

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

        finally:
            # Atualizar contagem final de voltas na sessão
            if session_id:
                query_service.update_session_lap_count(session_id, lap_count)

            uploader.stop(timeout_s=10.0)
            tts.stop()
            logger.info("Sessão encerrada", extra={"voltas_completadas": lap_count})


if __name__ == "__main__":
    args = parse_args()
    run(track_id=args.track, rate_hz=args.rate)
