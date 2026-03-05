"""
SharedMemoryReader — Leitura em tempo real da Shared Memory do Assetto Corsa.

Usa mmap (stdlib Python) para acessar diretamente as três structs do AC:
- SPageFilePhysics  (Local\\acpmf_physics)
- SPageFileGraphic  (Local\\acpmf_graphics)
- SPageFileStatic   (Local\\acpmf_static)

Referência: docs/ACSharedMemoryDocumentation.pdf
"""

import ctypes
import logging
import mmap
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from src.memory.graphics_page import SPageFileGraphic
from src.memory.physics_page import SPageFilePhysics
from src.memory.static_page import SPageFileStatic

logger = logging.getLogger(__name__)

# Nomes das Shared Memory do AC (protocolo Windows)
_SM_PHYSICS = "Local\\acpmf_physics"
_SM_GRAPHICS = "Local\\acpmf_graphics"
_SM_STATIC = "Local\\acpmf_static"

# FILE_MAP_READ — permissão mínima para OpenFileMappingW
_FILE_MAP_READ = 0x0004


def _mapping_exists(name: str) -> bool:
    """
    Verifica se um named file mapping existe no Windows.

    Usa OpenFileMappingW (não CreateFileMapping) para distinguir entre
    "AC está rodando e criou o mapping" vs "mmap.mmap() criaria um novo".
    Retorna False em plataformas não-Windows.
    """
    if sys.platform != "win32":
        return False
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenFileMappingW(_FILE_MAP_READ, False, name)
    if handle:
        kernel32.CloseHandle(handle)
        return True
    return False


@dataclass
class TelemetrySnapshot:
    """
    Snapshot atômico de telemetria contendo os três frames sincronizados.

    Todos os campos são cópias seguras dos dados da Shared Memory.
    O timestamp_ns é capturado imediatamente antes da leitura.
    """

    timestamp_ns: int
    physics: SPageFilePhysics
    graphics: SPageFileGraphic
    static: SPageFileStatic
    read_ok: bool = True
    error: Optional[str] = None


class SharedMemoryReader:
    """
    Leitor de Shared Memory do Assetto Corsa via mmap.

    Mantém handles abertos para as três regiões de memória e expõe
    uma interface de leitura estável a qualquer frequência. O GIL do
    Python é liberado durante a cópia da memória via ctypes.

    Uso típico:
        reader = SharedMemoryReader()
        with reader:
            while True:
                snapshot = reader.read()
                if snapshot.read_ok:
                    process(snapshot)
                time.sleep(1 / 20)  # 20Hz
    """

    def __init__(self) -> None:
        self._mm_physics: Optional[mmap.mmap] = None
        self._mm_graphics: Optional[mmap.mmap] = None
        self._mm_static: Optional[mmap.mmap] = None
        self._connected: bool = False
        self._static_cache: Optional[SPageFileStatic] = None

    # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "SharedMemoryReader":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Conexão
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Abre os handles para as três regiões de Shared Memory do AC.

        Retorna True se conectado com sucesso, False se o AC não estiver
        rodando ou a memória não estiver disponível.

        Usa OpenFileMappingW para verificar se o AC criou os mappings antes
        de abri-los, evitando criar regiões vazias acidentalmente.
        """
        if not _mapping_exists(_SM_PHYSICS):
            logger.warning(
                "Shared Memory do AC não encontrada — AC está rodando?",
                extra={"mapping": _SM_PHYSICS},
            )
            self._connected = False
            return False

        try:
            self._mm_physics = mmap.mmap(
                -1,
                ctypes.sizeof(SPageFilePhysics),
                _SM_PHYSICS,
                access=mmap.ACCESS_READ,
            )
            self._mm_graphics = mmap.mmap(
                -1,
                ctypes.sizeof(SPageFileGraphic),
                _SM_GRAPHICS,
                access=mmap.ACCESS_READ,
            )
            self._mm_static = mmap.mmap(
                -1,
                ctypes.sizeof(SPageFileStatic),
                _SM_STATIC,
                access=mmap.ACCESS_READ,
            )
            self._connected = True
            logger.info(
                "Shared Memory conectada",
                extra={
                    "physics_size": ctypes.sizeof(SPageFilePhysics),
                    "graphics_size": ctypes.sizeof(SPageFileGraphic),
                    "static_size": ctypes.sizeof(SPageFileStatic),
                },
            )
            return True

        except OSError as exc:
            logger.warning(
                "Falha ao conectar na Shared Memory — AC está rodando?",
                extra={"error": str(exc)},
            )
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Fecha todos os handles de mmap com segurança."""
        for name, mm in [
            ("physics", self._mm_physics),
            ("graphics", self._mm_graphics),
            ("static", self._mm_static),
        ]:
            if mm is not None:
                try:
                    mm.close()
                    logger.debug("mmap fechado", extra={"region": name})
                except Exception as exc:
                    logger.error(
                        "Erro ao fechar mmap",
                        extra={"region": name, "error": str(exc)},
                    )

        self._mm_physics = None
        self._mm_graphics = None
        self._mm_static = None
        self._connected = False
        self._static_cache = None
        logger.info("Shared Memory desconectada")

    @property
    def is_connected(self) -> bool:
        """True se os handles de mmap estão abertos."""
        return self._connected

    # ------------------------------------------------------------------
    # Leitura
    # ------------------------------------------------------------------

    def read(self) -> TelemetrySnapshot:
        """
        Lê um snapshot atômico da Shared Memory.

        Captura timestamp antes da leitura para garantir ordenação correta.
        Retorna TelemetrySnapshot com read_ok=False em caso de erro.
        """
        timestamp_ns = time.monotonic_ns()

        if not self._connected:
            return TelemetrySnapshot(
                timestamp_ns=timestamp_ns,
                physics=SPageFilePhysics(),
                graphics=SPageFileGraphic(),
                static=SPageFileStatic(),
                read_ok=False,
                error="Não conectado à Shared Memory",
            )

        try:
            physics = self._read_struct(self._mm_physics, SPageFilePhysics)
            graphics = self._read_struct(self._mm_graphics, SPageFileGraphic)

            # Static é lida apenas uma vez (dados não mudam durante a sessão)
            static = self._read_static()

            return TelemetrySnapshot(
                timestamp_ns=timestamp_ns,
                physics=physics,
                graphics=graphics,
                static=static,
                read_ok=True,
            )

        except Exception as exc:
            logger.error(
                "Erro durante leitura da Shared Memory",
                extra={"error": str(exc)},
            )
            return TelemetrySnapshot(
                timestamp_ns=timestamp_ns,
                physics=SPageFilePhysics(),
                graphics=SPageFileGraphic(),
                static=SPageFileStatic(),
                read_ok=False,
                error=str(exc),
            )

    def read_static(self) -> SPageFileStatic:
        """
        Lê (ou retorna do cache) os dados estáticos da sessão.

        Os dados estáticos não mudam durante uma sessão. O cache é
        invalidado apenas na reconexão.
        """
        return self._read_static()

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    @staticmethod
    def _read_struct(mm: mmap.mmap, struct_type: type) -> ctypes.Structure:
        """
        Copia bytes da mmap para uma instância da struct ctypes.

        A operação é O(sizeof(struct)) e thread-safe via GIL.
        """
        mm.seek(0)
        raw = mm.read(ctypes.sizeof(struct_type))
        instance = struct_type()
        ctypes.memmove(ctypes.addressof(instance), raw, len(raw))
        return instance

    def _read_static(self) -> SPageFileStatic:
        """Lê SPageFileStatic com cache por sessão."""
        if self._static_cache is not None:
            return self._static_cache

        static = self._read_struct(self._mm_static, SPageFileStatic)
        self._static_cache = static
        logger.info(
            "Dados estáticos lidos",
            extra={
                "track": static.track,
                "car": static.carModel,
                "track_length_m": static.trackSPlineLength,
                "sector_count": static.sectorCount,
            },
        )
        return static

    def invalidate_static_cache(self) -> None:
        """
        Invalida o cache de dados estáticos.

        Chamar quando uma nova sessão é detectada (volta 0 após voltas > 0).
        """
        self._static_cache = None
        logger.debug("Cache estático invalidado")


# ------------------------------------------------------------------
# Snapshot → dict (para serialização JSON)
# ------------------------------------------------------------------

def snapshot_to_dict(snapshot: TelemetrySnapshot) -> dict:
    """
    Converte um TelemetrySnapshot para dicionário serializável em JSON.

    Retorna apenas os campos do schema de mini-setor definido em CLAUDE.md §4.5,
    mais campos de validação de volta (§4.2).
    """
    p = snapshot.physics
    g = snapshot.graphics

    return {
        # --- Posicionamento ---
        "track_position": g.normalizedCarPosition,
        "delta_vs_best": p.performanceMeter,

        # --- Inputs do Piloto ---
        "throttle": p.gas,
        "brake": p.brake,
        "steering": p.steerAngle,
        "gear": p.gear,
        "rpms": p.rpms,
        "clutch": p.clutch,

        # --- Velocidade e Dinâmica ---
        "speed_kmh": p.speedKmh,
        "gforce_x": p.accG[0],
        "gforce_y": p.accG[1],
        "gforce_z": p.accG[2],
        "local_ang_vel_x": p.localAngularVel[0],
        "local_ang_vel_y": p.localAngularVel[1],
        "local_ang_vel_z": p.localAngularVel[2],

        # --- Tração ---
        "wheel_slip_fl": p.wheelSlip[0],
        "wheel_slip_fr": p.wheelSlip[1],
        "wheel_slip_rl": p.wheelSlip[2],
        "wheel_slip_rr": p.wheelSlip[3],
        "tc_active": p.tc,
        "abs_active": p.abs,
        "drs_active": int(p.drs),
        "drs_available": p.drsAvailable,
        "brake_bias": p.brakeBias,

        # --- Condições ---
        "surface_grip": g.surfaceGrip,
        "air_temp": p.airTemp,
        "road_temp": p.roadTemp,

        # --- Campos de validação (não gravados no schema de mini-setor) ---
        "_status": g.status,
        "_flag": g.flag,
        "_number_of_tyres_out": p.numberOfTyresOut,
        "_pit_limiter_on": p.pitLimiterOn,
        "_is_in_pit": g.isInPit,
        "_is_in_pit_lane": g.isInPitLane,
        "_penalty_time": g.penaltyTime,
        "_car_damage_max": max(p.carDamage),
        "_is_ai_controlled": p.isAIControlled,

        # --- Tempos de sessão ---
        "_i_current_time_ms": g.iCurrentTime,
        "_i_best_time_ms": g.iBestTime,
        "_i_last_time_ms": g.iLastTime,
        "_last_sector_time_ms": g.lastSectorTime,
        "_current_sector_index": g.currentSectorIndex,
        "_completed_laps": g.completedLaps,
    }
