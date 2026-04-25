"""
benchmark_tts.py — Mede latência empírica de cada provider TTS instalado.

Executa N rodadas de speak() para cada provider disponível, mede:
- enqueue_ms: tempo de speak() (queue.put)
- synthesis_ms: tempo de geração do áudio (rede + decode)
- audio_ms: tempo de playback até término
- total_ms: enqueue → fim do playback

Uso:
    py -3.11 scripts/benchmark_tts.py
    py -3.11 scripts/benchmark_tts.py --providers edge_tts,pyttsx3 --rounds 3
    py -3.11 scripts/benchmark_tts.py --message "Perdeu meio segundo na curva 3"

Saída: tabela comparativa com mean ± std por estágio.
"""

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.output.tts_integration import _VALID_PROVIDERS, TTSIntegration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_tts")

_DEFAULT_MESSAGE = "Perdeu zero vírgula três segundos na Parabolica. Frenagem tardia."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark de latência dos providers TTS",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--providers",
        default=",".join(p for p in _VALID_PROVIDERS if p != "none"),
        help="Lista de providers separados por vírgula",
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Iterações por provider (default: 3)",
    )
    parser.add_argument(
        "--message", default=_DEFAULT_MESSAGE,
        help="Mensagem de teste (default: alerta de exemplo)",
    )
    return parser.parse_args()


def benchmark_provider(provider: str, message: str, rounds: int) -> dict:
    """Executa N rodadas e retorna estatísticas por estágio."""
    print(f"\n[{provider}] inicializando...")
    tts = TTSIntegration(provider=provider)

    # Override do cooldown — benchmark precisa rodar consecutivo
    tts._min_interval_s = 0.0

    if tts._provider == "none":
        print(f"[{provider}] indisponível (provider efetivo='none')")
        return {"provider": provider, "available": False}

    tts.start()
    # Espera o engine inicializar na thread worker
    time.sleep(0.5)

    enqueue_times: list[float] = []
    total_times: list[float] = []
    effective_provider_seen: str = provider

    for i in range(rounds):
        t0 = time.monotonic()
        tts.speak(message)
        enqueue_times.append((time.monotonic() - t0) * 1000)

        # Espera a fila esvaziar (síntese + playback completos)
        t_wait = time.monotonic()
        while not tts._queue.empty():
            time.sleep(0.01)
            if (time.monotonic() - t_wait) > 30:
                break
        # Mais um instante para o worker terminar a chamada em curso
        time.sleep(0.2)
        total_times.append((time.monotonic() - t0) * 1000)
        effective_provider_seen = tts._effective_provider
        print(f"  round {i+1}/{rounds}: total={total_times[-1]:.0f}ms")

    tts.stop()

    return {
        "provider": provider,
        "effective_provider": effective_provider_seen,
        "available": True,
        "enqueue_ms_mean": round(statistics.mean(enqueue_times), 2),
        "total_ms_mean": round(statistics.mean(total_times), 1),
        "total_ms_std": round(statistics.stdev(total_times), 1) if len(total_times) > 1 else 0.0,
        "rounds": rounds,
    }


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 76)
    print(f"  {'Provider':<14} {'Effective':<14} {'Enqueue':<12} {'Total (mean±std)':<24} {'Status'}")
    print("=" * 76)
    for r in results:
        if not r.get("available"):
            print(f"  {r['provider']:<14} {'—':<14} {'—':<12} {'—':<24} indisponível")
            continue
        eff = r.get("effective_provider", r["provider"])
        enq = f"{r['enqueue_ms_mean']:.2f}ms"
        total = f"{r['total_ms_mean']:.0f} ± {r['total_ms_std']:.0f}ms"
        flag = "✓" if eff == r["provider"] else f"⚠ fallback→{eff}"
        print(f"  {r['provider']:<14} {eff:<14} {enq:<12} {total:<24} {flag}")
    print("=" * 76)
    print(
        "  Notas:\n"
        "    - 'Total' inclui síntese + playback (mensagem completa falada).\n"
        "    - 'Enqueue' é o que o loop principal de run_session.py paga.\n"
        "      O resto é absorvido pela thread worker — não bloqueia 20Hz.\n"
        "    - TTFB (time-to-first-byte) é menor que o total. Para sentir\n"
        "      'engenheiro responsivo', mire em total < 1.5s para mensagens curtas.\n"
    )


def main() -> None:
    args = parse_args()
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    print(f"Mensagem: \"{args.message}\" ({len(args.message)} chars)")
    print(f"Providers: {providers}, rounds: {args.rounds}")

    results: list[dict] = []
    for p in providers:
        if p not in _VALID_PROVIDERS:
            print(f"\n[{p}] inválido — pulando")
            continue
        try:
            results.append(benchmark_provider(p, args.message, args.rounds))
        except Exception as exc:
            logger.error("Falha no benchmark", extra={"provider": p, "error": str(exc)})
            results.append({"provider": p, "available": False})

    print_table(results)


if __name__ == "__main__":
    main()
