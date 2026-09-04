"""
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : IP Webcam Pro Connection Manager
File    : layer1/hardware/connection_manager.py

This module connects the laptop-hosted NOONGIL-X runtime directly to the
"IP Webcam" / "IP Webcam Pro" Android application.

The phone runs the IP Webcam server. NOONGIL-X runs on the laptop.

Main endpoints used
-------------------
/shot.jpg      Single camera frame
/video         MJPEG video stream
/audio.wav     Microphone audio stream, when enabled
/sensors.json  Enabled phone sensor values
/status.json   IP Webcam server status
/              Browser control page / fallback reachability check

Typical address shown by the app:
    http://192.168.1.8:8080

Important
---------
The laptop and phone must normally be connected to the same Wi-Fi network.
Enable audio and sensor options inside IP Webcam when you need them.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, urlopen


# ============================================================
# ENUMS
# ============================================================


class ConnectionState(str, Enum):
    """Current state of the IP Webcam connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    STOPPED = "stopped"


class IPWebcamFeature(str, Enum):
    """Features that may be available from IP Webcam."""

    CAMERA_SNAPSHOT = "camera_snapshot"
    VIDEO_STREAM = "video_stream"
    AUDIO_STREAM = "audio_stream"
    SENSORS = "sensors"
    STATUS = "status"
    TORCH_CONTROL = "torch_control"
    FOCUS_CONTROL = "focus_control"


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class IPWebcamConfig:
    """Configuration for the Android IP Webcam server."""

    host: str = "192.168.1.8"
    port: int = 8080
    scheme: str = "http"

    username: Optional[str] = None
    password: Optional[str] = None

    connect_timeout: float = 4.0
    request_timeout: float = 10.0
    heartbeat_interval: float = 5.0

    auto_reconnect: bool = True
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0
    reconnect_backoff: float = 1.5

    output_dir: str = "output/layer1/hardware"
    log_level: str = "INFO"

    root_endpoint: str = "/"
    snapshot_endpoint: str = "/shot.jpg"
    video_endpoint: str = "/video"
    audio_endpoint: str = "/audio.wav"
    sensors_endpoint: str = "/sensors.json"
    status_endpoint: str = "/status.json"

    def validate(self) -> None:
        if not self.host.strip():
            raise ValueError("IP Webcam host cannot be empty")

        if not 1 <= self.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")

        if self.scheme not in {"http", "https"}:
            raise ValueError("Scheme must be 'http' or 'https'")

        if self.connect_timeout <= 0:
            raise ValueError("connect_timeout must be greater than zero")

        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero")

        if self.heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be greater than zero")

        if self.reconnect_attempts < 0:
            raise ValueError("reconnect_attempts cannot be negative")


@dataclass
class EndpointAvailability:
    """Availability result for one IP Webcam endpoint."""

    endpoint: str
    available: bool
    content_type: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class IPWebcamStatus:
    """Serializable connection status."""

    state: ConnectionState = ConnectionState.DISCONNECTED
    connected: bool = False
    base_url: Optional[str] = None

    last_connected_at: Optional[float] = None
    last_heartbeat_at: Optional[float] = None
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None
    reconnect_count: int = 0

    available_features: List[str] = field(default_factory=list)
    endpoints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    server_status: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data


# ============================================================
# EXCEPTIONS
# ============================================================


class IPWebcamError(RuntimeError):
    """Base error for the IP Webcam bridge."""


class IPWebcamConnectionError(IPWebcamError):
    """Raised when the app server cannot be reached."""


class IPWebcamRequestError(IPWebcamError):
    """Raised when an endpoint request fails."""


class IPWebcamNotConnectedError(IPWebcamError):
    """Raised when device data is requested before connecting."""


class IPWebcamFeatureUnavailableError(IPWebcamError):
    """Raised when an unavailable feature is requested."""


# ============================================================
# LOGGER
# ============================================================


def _create_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("noongil.layer1.hardware.ip_webcam")

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ============================================================
# CONNECTION MANAGER
# ============================================================


class ConnectionManager:
    """
    Connect NOONGIL-X directly to IP Webcam Pro.

    Responsibilities
    ----------------
    - Build all IP Webcam endpoint URLs.
    - Verify that the phone server is reachable.
    - Detect available camera, video, audio, status, and sensor endpoints.
    - Maintain a heartbeat.
    - Reconnect after a temporary Wi-Fi interruption.
    - Fetch and save individual camera frames.
    - Provide OpenCV-compatible video source URLs.
    - Read JSON status and sensor information.
    - Send basic IP Webcam control commands.
    """

    def __init__(
        self,
        config: Optional[IPWebcamConfig] = None,
        *,
        status_callback: Optional[Callable[[IPWebcamStatus], None]] = None,
    ) -> None:
        self.config = config or IPWebcamConfig()
        self.config.validate()

        self.logger = _create_logger(self.config.log_level)
        self.status_callback = status_callback

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._reconnect_thread: Optional[threading.Thread] = None

        self._status = IPWebcamStatus(base_url=self.base_url)

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # URLS
    # --------------------------------------------------------

    @property
    def base_url(self) -> str:
        return (
            f"{self.config.scheme}://"
            f"{self.config.host.strip()}:{self.config.port}"
        )

    def get_url(self, endpoint: str) -> str:
        """Return a complete IP Webcam endpoint URL."""

        return urljoin(
            self.base_url.rstrip("/") + "/",
            endpoint.lstrip("/"),
        )

    @property
    def snapshot_url(self) -> str:
        return self.get_url(self.config.snapshot_endpoint)

    @property
    def video_url(self) -> str:
        return self.get_url(self.config.video_endpoint)

    @property
    def audio_url(self) -> str:
        return self.get_url(self.config.audio_endpoint)

    @property
    def sensors_url(self) -> str:
        return self.get_url(self.config.sensors_endpoint)

    @property
    def status_url(self) -> str:
        return self.get_url(self.config.status_endpoint)

    # --------------------------------------------------------
    # PUBLIC STATUS
    # --------------------------------------------------------

    @property
    def state(self) -> ConnectionState:
        with self._lock:
            return self._status.state

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._status.connected

    def get_status(self) -> IPWebcamStatus:
        """Return an independent copy of the latest state."""

        with self._lock:
            return IPWebcamStatus(
                state=self._status.state,
                connected=self._status.connected,
                base_url=self._status.base_url,
                last_connected_at=self._status.last_connected_at,
                last_heartbeat_at=self._status.last_heartbeat_at,
                last_error=self._status.last_error,
                latency_ms=self._status.latency_ms,
                reconnect_count=self._status.reconnect_count,
                available_features=list(self._status.available_features),
                endpoints={
                    name: dict(value)
                    for name, value in self._status.endpoints.items()
                },
                server_status=dict(self._status.server_status),
            )

    def supports(self, feature: IPWebcamFeature | str) -> bool:
        value = feature.value if isinstance(feature, IPWebcamFeature) else feature
        return value in self.get_status().available_features

    def _set_state(
        self,
        state: ConnectionState,
        *,
        connected: Optional[bool] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._status.state = state

            if connected is not None:
                self._status.connected = connected

            if error is not None:
                self._status.last_error = error
            elif state == ConnectionState.CONNECTED:
                self._status.last_error = None

            snapshot = self.get_status()

        self._save_status(snapshot)

        if self.status_callback:
            try:
                self.status_callback(snapshot)
            except Exception as exc:
                self.logger.warning("Status callback failed: %s", exc)

    def _save_status(self, status: IPWebcamStatus) -> None:
        path = self.output_dir / "ip_webcam_connection_status.json"
        temporary = path.with_suffix(".tmp")

        try:
            temporary.write_text(
                json.dumps(status.to_dict(), indent=4, default=str),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as exc:
            self.logger.debug("Unable to save status file: %s", exc)

    # --------------------------------------------------------
    # CONNECT / DISCONNECT
    # --------------------------------------------------------

    def connect(self) -> IPWebcamStatus:
        """
        Connect to the IP Webcam server and inspect its available endpoints.
        """

        if self._stop_event.is_set():
            self._stop_event.clear()

        self._set_state(ConnectionState.CONNECTING, connected=False)
        self.logger.info("Connecting to IP Webcam Pro at %s", self.base_url)

        try:
            root_result = self._probe_endpoint(
                self.config.root_endpoint,
                read_limit=512,
            )

            snapshot_result = self._probe_endpoint(
                self.config.snapshot_endpoint,
                read_limit=64,
            )

            if not root_result.available and not snapshot_result.available:
                raise IPWebcamConnectionError(
                    f"IP Webcam server is not reachable at {self.base_url}"
                )

            endpoint_results = self.detect_available_endpoints(
                known_snapshot=snapshot_result,
            )

            status_data: Dict[str, Any] = {}
            if endpoint_results["status"].available:
                try:
                    status_data = self.get_server_status(
                        require_connection=False
                    )
                except IPWebcamError:
                    status_data = {}

            available_features = self._features_from_endpoints(
                endpoint_results
            )

            with self._lock:
                self._status.connected = True
                self._status.base_url = self.base_url
                self._status.last_connected_at = time.time()
                self._status.last_heartbeat_at = time.time()
                self._status.latency_ms = (
                    snapshot_result.latency_ms
                    if snapshot_result.available
                    else root_result.latency_ms
                )
                self._status.available_features = available_features
                self._status.endpoints = {
                    name: asdict(result)
                    for name, result in endpoint_results.items()
                }
                self._status.server_status = status_data

            self._set_state(ConnectionState.CONNECTED, connected=True)
            self._start_heartbeat()

            self.logger.info(
                "Connected. Available features: %s",
                ", ".join(available_features) or "basic server access",
            )
            return self.get_status()

        except Exception as exc:
            self._set_state(
                ConnectionState.ERROR,
                connected=False,
                error=str(exc),
            )
            self.logger.error("IP Webcam connection failed: %s", exc)

            if isinstance(exc, IPWebcamError):
                raise

            raise IPWebcamConnectionError(str(exc)) from exc

    def disconnect(self) -> None:
        """Disconnect locally without stopping the phone application."""

        self.logger.info("Disconnecting from IP Webcam Pro")
        self._stop_heartbeat()

        with self._lock:
            self._status.connected = False
            self._status.available_features = []
            self._status.endpoints = {}
            self._status.server_status = {}
            self._status.latency_ms = None

        self._set_state(ConnectionState.DISCONNECTED, connected=False)

    def stop(self) -> None:
        """Stop heartbeat and reconnect operations."""

        self.logger.info("Stopping IP Webcam connection manager")
        self._stop_event.set()
        self._stop_heartbeat()

        reconnect_thread = self._reconnect_thread
        if (
            reconnect_thread
            and reconnect_thread.is_alive()
            and reconnect_thread is not threading.current_thread()
        ):
            reconnect_thread.join(timeout=2.0)

        with self._lock:
            self._status.connected = False

        self._set_state(ConnectionState.STOPPED, connected=False)

    def reconnect(self) -> bool:
        """Reconnect using the same IP address and port."""

        self._set_state(ConnectionState.RECONNECTING, connected=False)

        attempts = max(1, self.config.reconnect_attempts)
        delay = self.config.reconnect_delay

        for attempt in range(1, attempts + 1):
            if self._stop_event.is_set():
                return False

            self.logger.info(
                "Reconnect attempt %d/%d",
                attempt,
                attempts,
            )

            try:
                self.connect()

                with self._lock:
                    self._status.reconnect_count += 1

                self._save_status(self.get_status())
                return True

            except IPWebcamError as exc:
                self.logger.warning(
                    "Reconnect attempt %d failed: %s",
                    attempt,
                    exc,
                )

            if attempt < attempts:
                self._stop_event.wait(delay)
                delay *= self.config.reconnect_backoff

        self._set_state(
            ConnectionState.ERROR,
            connected=False,
            error="Maximum IP Webcam reconnection attempts reached",
        )
        return False

    # --------------------------------------------------------
    # ENDPOINT DETECTION
    # --------------------------------------------------------

    def detect_available_endpoints(
        self,
        *,
        known_snapshot: Optional[EndpointAvailability] = None,
    ) -> Dict[str, EndpointAvailability]:
        """
        Check the endpoints needed by Layer 1.

        Streaming endpoints are only opened briefly, and only a small number of
        bytes are read.
        """

        return {
            "snapshot": known_snapshot
            or self._probe_endpoint(
                self.config.snapshot_endpoint,
                read_limit=64,
            ),
            "video": self._probe_endpoint(
                self.config.video_endpoint,
                read_limit=64,
            ),
            "audio": self._probe_endpoint(
                self.config.audio_endpoint,
                read_limit=64,
            ),
            "sensors": self._probe_endpoint(
                self.config.sensors_endpoint,
                read_limit=256,
            ),
            "status": self._probe_endpoint(
                self.config.status_endpoint,
                read_limit=256,
            ),
        }

    @staticmethod
    def _features_from_endpoints(
        endpoints: Dict[str, EndpointAvailability],
    ) -> List[str]:
        features: List[str] = []

        mapping = {
            "snapshot": IPWebcamFeature.CAMERA_SNAPSHOT.value,
            "video": IPWebcamFeature.VIDEO_STREAM.value,
            "audio": IPWebcamFeature.AUDIO_STREAM.value,
            "sensors": IPWebcamFeature.SENSORS.value,
            "status": IPWebcamFeature.STATUS.value,
        }

        for endpoint_name, feature_name in mapping.items():
            if endpoints[endpoint_name].available:
                features.append(feature_name)

        # These controls are normally provided by IP Webcam's HTTP command API.
        if endpoints["snapshot"].available:
            features.extend(
                [
                    IPWebcamFeature.TORCH_CONTROL.value,
                    IPWebcamFeature.FOCUS_CONTROL.value,
                ]
            )

        return features

    def _probe_endpoint(
        self,
        endpoint: str,
        *,
        read_limit: int,
    ) -> EndpointAvailability:
        started = time.perf_counter()

        try:
            _, content_type = self._request_bytes(
                endpoint,
                timeout=self.config.connect_timeout,
                read_limit=read_limit,
            )

            latency = (time.perf_counter() - started) * 1000.0

            return EndpointAvailability(
                endpoint=endpoint,
                available=True,
                content_type=content_type,
                latency_ms=round(latency, 2),
            )

        except IPWebcamError as exc:
            return EndpointAvailability(
                endpoint=endpoint,
                available=False,
                error=str(exc),
            )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    def get_camera_frame(self) -> bytes:
        """Download one JPEG frame from /shot.jpg."""

        self._ensure_connected()
        self._require_feature(IPWebcamFeature.CAMERA_SNAPSHOT)

        data, content_type = self._request_bytes(
            self.config.snapshot_endpoint,
            timeout=self.config.request_timeout,
        )

        if not data:
            raise IPWebcamRequestError("IP Webcam returned an empty frame")

        if (
            "image" not in content_type.lower()
            and not data.startswith(b"\xff\xd8")
        ):
            raise IPWebcamRequestError(
                f"Unexpected snapshot content type: {content_type}"
            )

        return data

    def save_camera_frame(
        self,
        filename: Optional[str] = None,
    ) -> Path:
        """Save one camera frame into the Layer 1 output directory."""

        frame = self.get_camera_frame()

        if filename is None:
            filename = f"phone_frame_{int(time.time() * 1000)}.jpg"

        path = self.output_dir / filename
        path.write_bytes(frame)

        self.logger.info("Saved phone camera frame: %s", path)
        return path

    def get_opencv_video_source(self) -> str:
        """
        Return the MJPEG URL that can be passed to cv2.VideoCapture().
        """

        self._ensure_connected()
        self._require_feature(IPWebcamFeature.VIDEO_STREAM)
        return self._url_with_credentials(self.video_url)

    def open_opencv_capture(self) -> Any:
        """
        Open the phone's live MJPEG stream using OpenCV.

        OpenCV is imported only when this method is used.
        """

        self._ensure_connected()
        self._require_feature(IPWebcamFeature.VIDEO_STREAM)

        try:
            import cv2
        except ImportError as exc:
            raise IPWebcamError(
                "OpenCV is required. Install it using: "
                "pip install opencv-python"
            ) from exc

        capture = cv2.VideoCapture(self.get_opencv_video_source())

        if not capture.isOpened():
            capture.release()
            raise IPWebcamConnectionError(
                f"OpenCV could not open {self.video_url}"
            )

        return capture

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    def get_audio_stream_url(self) -> str:
        """Return the IP Webcam microphone stream URL."""

        self._ensure_connected()
        self._require_feature(IPWebcamFeature.AUDIO_STREAM)
        return self._url_with_credentials(self.audio_url)

    def read_audio_bytes(
        self,
        *,
        max_bytes: int = 65536,
    ) -> bytes:
        """
        Read a bounded chunk from /audio.wav.

        This is useful for testing. Continuous speech processing should consume
        `get_audio_stream_url()` using an audio/FFmpeg module.
        """

        self._ensure_connected()
        self._require_feature(IPWebcamFeature.AUDIO_STREAM)

        data, _ = self._request_bytes(
            self.config.audio_endpoint,
            timeout=self.config.request_timeout,
            read_limit=max_bytes,
        )
        return data

    # --------------------------------------------------------
    # STATUS AND SENSORS
    # --------------------------------------------------------

    def get_server_status(
        self,
        *,
        require_connection: bool = True,
    ) -> Dict[str, Any]:
        """Read /status.json from IP Webcam."""

        if require_connection:
            self._ensure_connected()
            self._require_feature(IPWebcamFeature.STATUS)

        return self._request_json(self.config.status_endpoint)

    def get_sensors(self) -> Dict[str, Any]:
        """
        Read enabled phone sensors from /sensors.json.

        The exact fields depend on the sensors enabled inside IP Webcam and the
        sensors physically present on the phone.
        """

        self._ensure_connected()
        self._require_feature(IPWebcamFeature.SENSORS)
        return self._request_json(self.config.sensors_endpoint)

    def collect_hardware_snapshot(
        self,
        *,
        include_camera_file: bool = True,
    ) -> Dict[str, Any]:
        """
        Collect one status/sensor/frame snapshot for Layer 1 testing.
        """

        self._ensure_connected()

        snapshot: Dict[str, Any] = {
            "timestamp": time.time(),
            "source": "IP Webcam Pro",
            "base_url": self.base_url,
            "connection": self.get_status().to_dict(),
            "data": {},
            "errors": {},
        }

        if self.supports(IPWebcamFeature.STATUS):
            try:
                snapshot["data"]["server_status"] = self.get_server_status()
            except IPWebcamError as exc:
                snapshot["errors"]["server_status"] = str(exc)

        if self.supports(IPWebcamFeature.SENSORS):
            try:
                snapshot["data"]["sensors"] = self.get_sensors()
            except IPWebcamError as exc:
                snapshot["errors"]["sensors"] = str(exc)

        if (
            include_camera_file
            and self.supports(IPWebcamFeature.CAMERA_SNAPSHOT)
        ):
            try:
                snapshot["data"]["camera_frame_path"] = str(
                    self.save_camera_frame("latest_phone_frame.jpg")
                )
            except IPWebcamError as exc:
                snapshot["errors"]["camera"] = str(exc)

        output_path = self.output_dir / "latest_hardware_snapshot.json"
        output_path.write_text(
            json.dumps(snapshot, indent=4, default=str),
            encoding="utf-8",
        )

        return snapshot

    # --------------------------------------------------------
    # IP WEBCAM CONTROLS
    # --------------------------------------------------------

    def set_torch(self, enabled: bool) -> bool:
        """
        Turn the phone flashlight on or off.

        IP Webcam commonly exposes:
            /enabletorch
            /disabletorch
        """

        self._ensure_connected()
        endpoint = "/enabletorch" if enabled else "/disabletorch"
        self._request_bytes(endpoint, read_limit=256)
        return True

    def autofocus(self) -> bool:
        """
        Request autofocus.

        IP Webcam commonly exposes the /focus endpoint.
        """

        self._ensure_connected()
        self._request_bytes("/focus", read_limit=256)
        return True

    def send_control_command(
        self,
        endpoint: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Send an additional IP Webcam HTTP command.

        Example:
            manager.send_control_command("/focus")
        """

        self._ensure_connected()

        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        if parameters:
            endpoint = f"{endpoint}?{urlencode(parameters)}"

        data, _ = self._request_bytes(endpoint, read_limit=4096)
        return data

    # --------------------------------------------------------
    # HEARTBEAT
    # --------------------------------------------------------

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="noongil-ip-webcam-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        thread = self._heartbeat_thread
        self._heartbeat_thread = None

        if (
            thread
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=self.config.heartbeat_interval + 1.0)

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.config.heartbeat_interval):
            if not self.is_connected:
                return

            try:
                started = time.perf_counter()

                # A snapshot request reliably checks both server and camera.
                self._request_bytes(
                    self.config.snapshot_endpoint,
                    timeout=self.config.connect_timeout,
                    read_limit=32,
                )

                latency = (time.perf_counter() - started) * 1000.0

                with self._lock:
                    self._status.last_heartbeat_at = time.time()
                    self._status.latency_ms = round(latency, 2)
                    self._status.last_error = None

                self._save_status(self.get_status())

            except IPWebcamError as exc:
                self.logger.warning("IP Webcam heartbeat failed: %s", exc)

                self._set_state(
                    ConnectionState.DISCONNECTED,
                    connected=False,
                    error=str(exc),
                )

                if self.config.auto_reconnect and not self._stop_event.is_set():
                    self._schedule_reconnect()
                return

    def _schedule_reconnect(self) -> None:
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return

        self._reconnect_thread = threading.Thread(
            target=self.reconnect,
            name="noongil-ip-webcam-reconnect",
            daemon=True,
        )
        self._reconnect_thread.start()

    # --------------------------------------------------------
    # HTTP
    # --------------------------------------------------------

    def _request_json(self, endpoint: str) -> Dict[str, Any]:
        data, _ = self._request_bytes(endpoint)

        try:
            result = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IPWebcamRequestError(
                f"Invalid JSON returned by {endpoint}"
            ) from exc

        if not isinstance(result, dict):
            raise IPWebcamRequestError(
                f"Expected a JSON object from {endpoint}"
            )

        return result

    def _request_bytes(
        self,
        endpoint: str,
        *,
        timeout: Optional[float] = None,
        read_limit: Optional[int] = None,
    ) -> Tuple[bytes, str]:
        url = self.get_url(endpoint)

        headers = {
            "Accept": "*/*",
            "User-Agent": "NOONGIL-X-Layer1-IPWebcam/1.0",
        }

        if self.config.username:
            credentials = (
                f"{self.config.username}:{self.config.password or ''}"
            )
            encoded = base64.b64encode(
                credentials.encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {encoded}"

        request = Request(url, headers=headers, method="GET")

        try:
            with urlopen(
                request,
                timeout=timeout or self.config.request_timeout,
            ) as response:
                if read_limit is None:
                    data = response.read()
                else:
                    data = response.read(read_limit)

                content_type = response.headers.get(
                    "Content-Type",
                    "application/octet-stream",
                )

                return data, content_type

        except HTTPError as exc:
            raise IPWebcamRequestError(
                f"HTTP {exc.code} from {url}"
            ) from exc
        except URLError as exc:
            raise IPWebcamConnectionError(
                f"Cannot reach {url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise IPWebcamConnectionError(
                f"Request timed out: {url}"
            ) from exc
        except OSError as exc:
            raise IPWebcamConnectionError(
                f"Network error for {url}: {exc}"
            ) from exc

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def _ensure_connected(self) -> None:
        if not self.is_connected:
            raise IPWebcamNotConnectedError(
                "IP Webcam Pro is not connected"
            )

    def _require_feature(
        self,
        feature: IPWebcamFeature,
    ) -> None:
        if not self.supports(feature):
            raise IPWebcamFeatureUnavailableError(
                f"IP Webcam feature is unavailable: {feature.value}. "
                "Check the app settings and restart its server."
            )

    def _url_with_credentials(self, url: str) -> str:
        """
        Embed credentials for clients such as OpenCV that cannot use the
        urllib Authorization header directly.
        """

        if not self.config.username:
            return url

        username = quote(self.config.username, safe="")
        password = quote(self.config.password or "", safe="")
        return url.replace("://", f"://{username}:{password}@", 1)

    # --------------------------------------------------------
    # CONTEXT MANAGER
    # --------------------------------------------------------

    def __enter__(self) -> "ConnectionManager":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.stop()


# ============================================================
# CONFIG LOADER
# ============================================================


def load_ip_webcam_config(path: str | Path) -> IPWebcamConfig:
    """Load IPWebcamConfig from JSON."""

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"IP Webcam configuration not found: {config_path}"
        )

    raw = json.loads(config_path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("Configuration must contain a JSON object")

    config = IPWebcamConfig(**raw)
    config.validate()
    return config


# ============================================================
# MANUAL TEST
# ============================================================


def main() -> None:
    """
    Manual connection test.

    Before running:
    1. Connect phone and laptop to the same Wi-Fi.
    2. Open IP Webcam Pro.
    3. Scroll down and press "Start server".
    4. Copy the IP address displayed on the phone.
    5. Replace the host below.
    """

    print("\n" + "=" * 68)
    print("NOONGIL-X | IP WEBCAM PRO CONNECTION MANAGER")
    print("=" * 68)

    config = IPWebcamConfig(
        host="192.168.1.8",  # CHANGE THIS to the IP shown by the app.
        port=8080,
        username=None,
        password=None,
        auto_reconnect=True,
    )

    manager = ConnectionManager(config)

    try:
        status = manager.connect()

        print("\n[CONNECTED]")
        print(json.dumps(status.to_dict(), indent=4, default=str))

        print("\n[STREAM URLS]")
        print(f"Snapshot : {manager.snapshot_url}")
        print(f"Video    : {manager.video_url}")

        if manager.supports(IPWebcamFeature.AUDIO_STREAM):
            print(f"Audio    : {manager.audio_url}")

        if manager.supports(IPWebcamFeature.SENSORS):
            print(f"Sensors  : {manager.sensors_url}")

        snapshot = manager.collect_hardware_snapshot()

        print("\n[LAYER 1 HARDWARE SNAPSHOT]")
        print(json.dumps(snapshot, indent=4, default=str))

        print("\nConnection is active. Press Ctrl+C to stop.")

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    except IPWebcamError as exc:
        print(f"\n[ERROR] {exc}")
        print(
            "\nCheck that the IP address matches the one displayed "
            "inside IP Webcam Pro."
        )

    finally:
        manager.stop()


if __name__ == "__main__":
    main()