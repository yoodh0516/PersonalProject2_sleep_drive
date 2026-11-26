

from contextlib import ExitStack
from pathlib import Path
import logging
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template, request
from gpiozero import Button, LED, MCP3008, Servo, TonalBuzzer
from gpiozero.tones import Tone


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
logger = logging.getLogger(__name__)

CAMERA_SOURCE = 0

# 조이스틱 설정
DEAD_ZONE = 0.15
INVERT_X = False
INVERT_Y = False

# 졸음 감지 설정 (팀원 B 설정값 원형 유지)
DROWSY_SECONDS = 2.0  # 2초 이상 눈 감김: 2단계 경보 (부저 880Hz + 조이스틱 햅틱 진동)
STAGE3_SECONDS = 4.0  # 4초 이상 눈 감김: 3단계 경보 (부저 1046Hz + 강력 진동 + LED 점멸)
MAX_OBSERVATION_GAP = 0.5

# 입력 핀: BCM 기준
PIN_ACCELERATOR = 22
PIN_BRAKE = 23

# 출력 핀: BCM 기준
PIN_LED = 17
PIN_SERVO = 18   # MG996R 메탈 기어 서보 (조이스틱 햅틱 진동)
PIN_BUZZER = 27  # 피에조 부저


def normalize_axis(value, invert=False):
    """조이스틱 축 값을 -1.0 ~ 1.0 범위로 정규화"""
    value = max(-1.0, min(1.0, (value - 0.5) * 2.0))

    if invert:
        value = -value

    if abs(value) <= DEAD_ZONE:
        return 0.0

    magnitude = (abs(value) - DEAD_ZONE) / (1.0 - DEAD_ZONE)
    return round(magnitude if value > 0 else -magnitude, 3)


class HardwareController:
    """Flask와 졸음 감지에서 공동으로 사용하는 하드웨어 제어기."""

    def __init__(self, spi_port=0, spi_device=0):
        self._lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._resources = ExitStack()
        self._is_alarm_active = False
        self._alarm_stage = None
        self._closing = False

        self._headlight = False
        self._last_error = None

        # 햅틱 진동 및 부저 백그라운드 스레드 제어
        self._stop_vibe_event = threading.Event()
        self._vibe_thread = None

        self._center_x = 0.5
        self._filtered_steer = 0.0
        self._ema_alpha = 0.35
        self._deadzone = 0.08

        pins = [
            PIN_ACCELERATOR,
            PIN_BRAKE,
            PIN_LED,
            PIN_SERVO,
            PIN_BUZZER,
        ]

        if len(set(pins)) != len(pins):
            raise ValueError("GPIO 핀이 중복되었습니다. PIN_* 설정을 확인하세요.")

        def own(device):
            self._resources.callback(device.close)
            return device

        try:
            self.led = own(LED(PIN_LED))
            self.btn_accel = own(
                Button(PIN_ACCELERATOR, pull_up=True, bounce_time=0.03)
            )
            self.btn_brake = own(
                Button(PIN_BRAKE, pull_up=True, bounce_time=0.03)
            )
            # octaves=3으로 설정하여 155Hz ~ 1245Hz 가청 대역 확보 (범위 초과 에러 방지)
            self.buzzer = own(
                TonalBuzzer(PIN_BUZZER, mid_tone="A4", octaves=3)
            )
            # MG996R 서보모터 펄스 규격
            self.servo = own(
                Servo(
                    PIN_SERVO,
                    initial_value=None,
                    min_pulse_width=0.0005,
                    max_pulse_width=0.0024,
                    frame_width=0.02,
                )
            )
            self.joy_x = own(
                MCP3008(channel=0, port=spi_port, device=spi_device)
            )
            self.joy_y = own(
                MCP3008(channel=1, port=spi_port, device=spi_device)
            )

            print(
                "조이스틱 영점 보정 중: 손을 떼고 중앙에 두세요.",
                flush=True,
            )
            self._calibrate_joystick()

        except BaseException:
            self._resources.close()
            raise

    def _ensure_open(self):
        if self._closing:
            raise RuntimeError("하드웨어가 종료 중이거나 종료되었습니다.")

    def _calibrate_joystick(self, samples=15):
        values = []
        for _ in range(samples):
            values.append(self.joy_x.value)
            time.sleep(0.01)

        center = sum(values) / samples

        if 0.2 <= center <= 0.8:
            self._center_x = center
        else:
            logger.warning(
                "조이스틱 중심값 %.3f 이상: 기본값 0.5 사용",
                center,
            )

    def get_driving_inputs(self):
        with self._lock:
            self._ensure_open()

            raw_x, raw_y = self.joy_x.value, self.joy_y.value
            diff = raw_x - self._center_x
            target = 0.0

            if diff > self._deadzone:
                target = (diff - self._deadzone) / max(
                    0.01,
                    1.0 - self._center_x - self._deadzone,
                )
            elif diff < -self._deadzone:
                target = (diff + self._deadzone) / max(
                    0.01,
                    self._center_x - self._deadzone,
                )

            target = max(-1.0, min(1.0, target))

            # EMA 필터로 조향 변화 완화
            self._filtered_steer = (
                self._ema_alpha * target
                + (1.0 - self._ema_alpha) * self._filtered_steer
            )
            steer = round(
                self._filtered_steer * (-1 if INVERT_X else 1),
                3,
            )

            # 브레이크 우선 처리 (BOS)
            brake = bool(self.btn_brake.is_pressed)
            accel = bool(self.btn_accel.is_pressed) and not brake

            return {
                # 기존 HTML에서 사용하는 항목
                "ok": True,
                "x": steer,
                "y": normalize_axis(raw_y, INVERT_Y),
                "raw_x": round(raw_x, 4),
                "raw_y": round(raw_y, 4),
                "accelerator": accel,
                "brake": brake,

                # HC 기존 이름 호환
                "steer": steer,
                "is_accel": accel,
                "is_brake": brake,

                **self._status_unlocked(),
            }

    def _status_unlocked(self):
        return {
            "is_headlight": self._headlight,
            "led_is_lit": bool(self.led.is_lit),
            "is_alarm_active": self._is_alarm_active,
            "alarm_stage": self._alarm_stage,
            "hardware_error": self._last_error,
        }

    def status(self):
        with self._lock:
            self._ensure_open()
            return self._status_unlocked()

    def set_headlight(self, state):
        if type(state) is not bool:
            raise ValueError("state는 true 또는 false여야 합니다.")

        with self._lock:
            self._ensure_open()
            self._headlight = state

            # 3단계 경보 중이 아닐 때만 직접 LED 상태 반영
            if not (self._is_alarm_active and self._alarm_stage == 3):
                self.led.value = state

    def _play_tone_safe(self, frequency):
        """부저 주파수 한계 범위를 벗어나지 않도록 안전 재생"""
        try:
            frequency = max(
                self.buzzer.min_tone.frequency,
                min(self.buzzer.max_tone.frequency, float(frequency)),
            )
            self.buzzer.play(Tone(frequency=frequency))
        except Exception:
            pass

    def set_drowsy_alarm(self, stage=None):
        """
        졸음 경보 제어:
        - None: 경보 해제
        - 2단계: 880Hz 부저 비프음 + MG996R 조이스틱 햅틱 진동
        - 3단계: 1046Hz 고주파 부저 + 강력 햅틱 진동 + LED 비상등 점멸
        """
        if stage is not None and stage not in (2, 3):
            raise ValueError("stage는 None, 2, 3 중 하나여야 합니다.")

        with self._lock:
            self._ensure_open()

            if stage is None:
                self.stop_alarm()
                return True

            if self._last_error is not None:
                return False

            # 동일 단계 경보가 이미 실행 중이면 중복 실행 방지
            if self._is_alarm_active and self._alarm_stage == stage:
                return True

            # 단계 전환(2 -> 3) 시 기존 워커 스레드 정지 후 재시작
            self._stop_vibe_worker()

            self._is_alarm_active = True
            self._alarm_stage = stage
            self._stop_vibe_event.clear()

            # Flask 웹 루프를 방해하지 않도록 별도 백그라운드 스레드에서 서보 햅틱 진동 실행
            self._vibe_thread = threading.Thread(
                target=self._haptic_alarm_worker,
                args=(stage,),
                name="haptic-alarm-worker",
                daemon=True,
            )
            self._vibe_thread.start()
            logger.info("졸음 경보 %s단계 시작 (부저 + MG996R 조이스틱 햅틱 진동)", stage)
            return True

    def _haptic_alarm_worker(self, stage):
        """
        백그라운드 스레드: 부저 톤 출력, MG996R 메탈 기어 서보 햅틱 진동, LED 점멸 수행
        - 방법 B 적용: 종료 시 0.3초간 위치를 고정한 뒤 신호를 분리하여 정확한 0도 원위치 복귀 보장
        """
        try:
            freq = 880 if stage == 2 else 1046
            vibe_amp = 0.30 if stage == 2 else 0.45
            cycle_delay = 0.055

            self._play_tone_safe(freq)
            step = 0

            while not self._stop_vibe_event.is_set():
                # 3단계 비상시 LED 점멸 (0.11초 주기)
                if stage == 3:
                    self.led.value = (step % 2 == 0)

                # MG996R 서보모터 좌우 왕복 운동 (조이스틱 흔들림 전달)
                self.servo.value = vibe_amp
                time.sleep(cycle_delay)
                self.servo.value = -vibe_amp
                time.sleep(cycle_delay)
                step += 1

        except Exception as exc:
            self._record_error(exc)
        finally:
            # ★ [방법 B]: 부저 끄고 정중앙(0도) 복귀 후 0.3초 위치 고정 유지 -> 그 뒤 detach
            try:
                self.buzzer.stop()
                self.servo.mid()
                time.sleep(0.3)
                self.servo.detach()
            except Exception:
                pass

    def _stop_vibe_worker(self):
        """진동 스레드 안전 중단 신호 전달"""
        self._stop_vibe_event.set()
        if self._vibe_thread and self._vibe_thread.is_alive():
            self._vibe_thread.join(timeout=0.5)
        self._vibe_thread = None

    def _record_error(self, exc):
        self._last_error = str(exc)
        logger.exception("하드웨어 경보 처리 실패")

    def _cleanup_action(self, action):
        try:
            action()
        except Exception as exc:
            self._record_error(exc)

    def _reset_alarm_outputs(self):
        self._stop_vibe_worker()
        self._cleanup_action(self.buzzer.stop)
        self._cleanup_action(lambda: setattr(self.led, "value", self._headlight))
        self._is_alarm_active = False
        self._alarm_stage = None

    def stop_alarm(self):
        """운전자 정상 복귀 시 경보 즉시 해제"""
        with self._lock:
            if self._closing or not self._is_alarm_active:
                return
            self._reset_alarm_outputs()

    def close(self):
        with self._close_lock:
            with self._lock:
                if self._closing:
                    return
                self._closing = True
                self._reset_alarm_outputs()
                self._resources.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==============================================================================
# 전역 변수 및 카메라/Flask 연동 설정
# ==============================================================================
hardware = None
camera = None
face_cascade = None
eye_cascade = None

camera_lock = threading.Lock()
frame_condition = threading.Condition(camera_lock)
shutdown_event = threading.Event()

latest_jpeg = None
frame_sequence = 0
last_frame_time = None
camera_available = False

detection_status = {
    "state": "unknown",
    "message": "카메라 확인 중입니다.",
    "missing_seconds": 0.0,
    "alarm_stage": None,
}

# 영상 작업 스레드 전용 상태 변수
eye_missing_since = None
last_observation_time = None
observation_status = detection_status.copy()


def load_cascade(filename):
    folders = [
        BASE_DIR / "haar-cascade-files-master",
        BASE_DIR / "models",
    ]

    if hasattr(cv2, "data"):
        folders.append(Path(cv2.data.haarcascades))

    for folder in folders:
        path = folder / filename
        if path.is_file():
            classifier = cv2.CascadeClassifier(str(path))
            if not classifier.empty():
                print(f"모델 로딩 성공: {path}", flush=True)
                return classifier

    raise RuntimeError(
        f"모델을 찾거나 읽을 수 없습니다: {filename}\n"
        f"검색 폴더: {folders}"
    )


def json_response(data, status=200):
    response = jsonify(data)
    response.status_code = status
    response.headers["Cache-Control"] = "no-store"
    return response


# ==============================================================================
# Flask 라우팅 엔드포인트
# ==============================================================================
@app.route("/")
@app.route("/car")
@app.route("/sleep_drive.html")
def car():
    response = app.make_response(render_template("sleep_drive.html"))
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/input")
def read_input():
    try:
        if hardware is None:
            raise RuntimeError("GPIO가 초기화되지 않았습니다.")
        return json_response(hardware.get_driving_inputs())
    except Exception:
        app.logger.exception("GPIO 입력 읽기 실패")
        return json_response(
            {
                "ok": False,
                "error": "GPIO 입력을 읽지 못했습니다. Python 터미널을 확인하세요.",
            },
            503,
        )


@app.route("/api/headlight", methods=["GET", "POST"])
def headlight():
    if hardware is None:
        return json_response(
            {"ok": False, "error": "GPIO가 초기화되지 않았습니다."},
            503,
        )

    if request.method == "POST":
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or type(data.get("state")) is not bool:
            return json_response(
                {
                    "ok": False,
                    "error": 'JSON {"state": true} 또는 {"state": false}를 보내세요.',
                },
                400,
            )

    try:
        if request.method == "POST":
            hardware.set_headlight(data["state"])
        return json_response({"ok": True, **hardware.status()})
    except Exception:
        app.logger.exception("LED 제어 실패")
        return json_response(
            {"ok": False, "error": "LED 제어에 실패했습니다."},
            503,
        )


def current_detection():
    with camera_lock:
        if last_frame_time is None:
            return detection_status.copy()

        if time.monotonic() - last_frame_time > MAX_OBSERVATION_GAP:
            return {
                "state": "unknown",
                "missing_seconds": 0.0,
                "alarm_stage": None,
                "message": "영상 수신 대기 중이거나 처리가 지연되고 있습니다.",
            }

        return detection_status.copy()


@app.route("/video")
@app.route("/drowsiness_status")
def drowsiness_status():
    return json_response(current_detection())


def publish_camera_failure(message):
    global latest_jpeg, last_frame_time, camera_available

    with frame_condition:
        latest_jpeg = None
        last_frame_time = None
        camera_available = False

        detection_status.update(
            state="unknown", message=message,
            missing_seconds=0.0, alarm_stage=None,
        )
        frame_condition.notify_all()


# ==============================================================================
# 카메라 및 알람 동기화 워커 스레드
# ==============================================================================
def camera_worker():
    """브라우저 연결 수와 관계없이 한 곳에서만 카메라를 읽고 분석한다."""
    global latest_jpeg, last_frame_time, frame_sequence, camera_available
    global eye_missing_since, last_observation_time

    try:
        while not shutdown_event.is_set():
            success, frame = camera.read()
            captured_at = time.monotonic()

            if not success:
                eye_missing_since = None
                last_observation_time = None
                publish_camera_failure("카메라에서 영상을 읽지 못했습니다.")
                shutdown_event.wait(0.1)
                continue

            frame = detect_face_and_eyes(frame)
            encoded, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )

            with frame_condition:
                detection_status.update(observation_status)
                last_frame_time = captured_at
                camera_available = True
                latest_jpeg = buffer.tobytes() if encoded else None
                frame_sequence += 1
                frame_condition.notify_all()

            shutdown_event.wait(1 / 15)

    except Exception:
        app.logger.exception("영상 처리 실패")
        publish_camera_failure("영상 처리에 실패했습니다. Python 터미널을 확인하세요.")

    finally:
        camera.release()


def sync_drowsiness_alarm():
    """카메라 분석 결과와 하드웨어 경보 상태 동기화"""
    if hardware is None:
        return

    status = current_detection()
    if status["state"] == "drowsy":
        hardware.set_drowsy_alarm(status["alarm_stage"])
    else:
        hardware.stop_alarm()


def alarm_monitor():
    try:
        while not shutdown_event.is_set():
            sync_drowsiness_alarm()
            shutdown_event.wait(0.05)
    except Exception:
        app.logger.exception("졸음 경보 연동 실패")
    finally:
        if hardware is not None:
            hardware.stop_alarm()


def generate_frames():
    seen_sequence = -1
    while not shutdown_event.is_set():
        with frame_condition:
            frame_condition.wait_for(
                lambda: (
                    shutdown_event.is_set()
                    or frame_sequence != seen_sequence
                ),
                timeout=1.0,
            )

            if shutdown_event.is_set():
                return

            if frame_sequence == seen_sequence:
                continue

            seen_sequence = frame_sequence
            jpeg = latest_jpeg

        if jpeg is not None:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + jpeg
                + b"\r\n"
            )


@app.route("/video_feed")
def video_feed():
    with camera_lock:
        available = camera_available

    if not available:
        return Response(
            "카메라 연결과 CAMERA_SOURCE를 확인하세요.",
            status=503,
            content_type="text/plain; charset=utf-8",
        )

    return Response(
        generate_frames(),
        content_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


# ==============================================================================
# [요청 반영] 팀원 B의 얼굴 및 눈 검출 알고리즘 100% 원형 유지 (수정사항 제외)
# ==============================================================================
def detect_face_and_eyes(frame):
    global eye_missing_since, last_observation_time

    now = time.monotonic()

    if (
        last_observation_time is not None
        and now - last_observation_time > MAX_OBSERVATION_GAP
    ):
        eye_missing_since = None

    last_observation_time = now
    observation_status.update(missing_seconds=0.0, alarm_stage=None)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(60, 60),
    )

    # 얼굴이 없으면 눈 감김 여부를 판단하지 않음
    if len(faces) == 0:
        eye_missing_since = None
        observation_status.update(
            state="unknown",
            message="얼굴이 보이지 않습니다. 카메라를 바라봐 주세요.",
        )
        cv2.putText(
            frame,
            "FACE NOT DETECTED",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )
        return frame

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (255, 150, 0),
        2,
    )

    eye_top = y + int(h * 0.15)
    eye_bottom = y + int(h * 0.60)
    eye_left = x + int(w * 0.08)
    eye_right = x + int(w * 0.92)

    eye_region = gray[eye_top:eye_bottom, eye_left:eye_right]

    eyes = eye_cascade.detectMultiScale(
        eye_region,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(12, w // 10), max(12, h // 12)),
        maxSize=(int(w * 0.45), int(h * 0.35)),
    )

    for ex, ey, ew, eh in eyes:
        cv2.rectangle(
            frame,
            (eye_left + ex, eye_top + ey),
            (eye_left + ex + ew, eye_top + ey + eh),
            (0, 220, 0),
            2,
        )

    cv2.putText(
        frame,
        f"FACE DETECTED | EYE REGIONS: {len(eyes)}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 0),
        2,
    )

    if len(eyes) == 0:
        if eye_missing_since is None:
            eye_missing_since = now

        missing_seconds = now - eye_missing_since
        observation_status["missing_seconds"] = missing_seconds

        if missing_seconds >= DROWSY_SECONDS:
            stage = 3 if missing_seconds >= STAGE3_SECONDS else 2
            observation_status.update(
                state="drowsy",
                alarm_stage=stage,
                message=(
                    f"⚠ 졸음 감지됨 [{stage}단계] — 눈이 {missing_seconds:.1f}초 동안 "
                    "검출되지 않았습니다."
                ),
            )
            message = f"STAGE {stage} | NO EYES: {missing_seconds:.1f}s"
            color = (0, 0, 255)

        else:
            observation_status.update(
                state="checking",
                message=(
                    f"눈 미검출 {missing_seconds:.1f}초 — "
                    "상태를 확인하고 있습니다."
                ),
            )
            message = f"EYES NOT DETECTED: {missing_seconds:.1f}s"
            color = (0, 165, 255)

    else:
        eye_missing_since = None
        observation_status.update(
            state="detected",
            message="눈이 검출되고 있습니다.",
        )
        message = "EYES DETECTED"
        color = (0, 220, 0)

    cv2.putText(
        frame,
        message,
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )

    return frame


# ==============================================================================
# 메인 함수
# ==============================================================================
def main():
    global hardware, camera, face_cascade, eye_cascade, camera_available

    logging.basicConfig(level=logging.INFO)

    html_path = TEMPLATE_DIR / "sleep_drive.html"
    if not html_path.is_file():
        raise SystemExit(f"기존 HTML 파일을 배치하세요:\n{html_path}")

    face_cascade = load_cascade("haarcascade_frontalface_default.xml")
    eye_cascade = load_cascade("haarcascade_eye.xml")

    shutdown_event.clear()

    with HardwareController() as hw:
        hardware = hw
        workers = []
        camera_thread = None

        try:
            camera = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_V4L2)

            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                camera_available = True
                camera_thread = threading.Thread(
                    target=camera_worker,
                    name="camera",
                    daemon=True,
                )
                camera_thread.start()
                workers.append(camera_thread)

            else:
                publish_camera_failure("카메라 연결을 확인해 주세요.")
                print(
                    "카메라 연결 실패. 주행 입력은 사용할 수 있습니다.",
                    flush=True,
                )

            monitor = threading.Thread(
                target=alarm_monitor,
                name="alarm-monitor",
                daemon=True,
            )
            monitor.start()
            workers.append(monitor)

            print(f"실행 파일: {Path(__file__).resolve()}", flush=True)
            print("주행 화면: http://<라즈베리파이 IP>:5000/", flush=True)
            print(
                f"입력: CH0/CH1, 액셀{PIN_ACCELERATOR}, 브레이크{PIN_BRAKE} / "
                f"출력: LED{PIN_LED}, 서보{PIN_SERVO}, 부저{PIN_BUZZER} (BCM)",
                flush=True,
            )

            app.run(
                host="0.0.0.0",
                port=5000,
                debug=False,
                threaded=True,
                use_reloader=False,
            )

        finally:
            shutdown_event.set()

            with frame_condition:
                frame_condition.notify_all()

            hw.stop_alarm()

            for worker in workers:
                worker.join(timeout=3.0)
                if worker.is_alive():
                    logger.warning(
                        "%s 스레드 종료 대기 시간 초과",
                        worker.name,
                    )

            if camera is not None and (
                camera_thread is None or camera_thread.ident is None
            ):
                camera.release()

            hardware = None


if __name__ == "__main__":
    main()
