import threading
import time
from datetime import datetime

import cv2

from app import config
from app.event_logger import EventLogger
from app.face_recog import create_face_app
from app.posture import classify_posture, is_fall_transition
from app.tracker import SimpleTracker, iou_xyxy

COCO_PERSON = 0
COCO_BOTTLE = 39


class CameraService:
    def __init__(self):
        self.detector = None
        self.pose_model = None
        self.logger = EventLogger.from_config()

        # GIU FACE NHUNG KHONG DUNG TRONG BAN CHAY CHINH
        self.face_app = create_face_app() if config.ENABLE_FACE else None

        self.cap = None
        self.tracker = SimpleTracker(
            iou_thresh=config.TRACK_IOU_THRESH,
            max_age=config.TRACK_MAX_AGE,
            smooth_alpha=config.TRACK_SMOOTH_ALPHA,
            min_hits=config.TRACK_MIN_HITS,
        )
        self.bottle_tracker = SimpleTracker(
            iou_thresh=config.BOTTLE_TRACK_IOU_THRESH,
            max_age=config.BOTTLE_TRACK_MAX_AGE,
            smooth_alpha=config.TRACK_SMOOTH_ALPHA,
            min_hits=config.BOTTLE_TRACK_MIN_HITS,
        )

        self.running = False
        self.thread = None
        self.latest_jpeg = None
        self.lock = threading.Lock()

        self.frame_idx = 0
        self.last_occ_ts = 0.0
        self.cooldowns = {}
        self.track_state = {}
        self.status = {
            "cam_id": config.CAMERA_ID,
            "online": False,
            "fps": 0.0,
            "people_count": 0,
            "unknown_count": 0,
            "sitting_count": 0,
            "lying_count": 0,
            "fall_count": 0,
            "overload": False,
            "last_frame_ts": None,
            "error": None,
            "backend": config.CV_BACKEND,
            "camera_source": config.CAMERA_SOURCE,
        }

    def _open_camera(self):
        src = config.CAMERA_SOURCE
        if isinstance(src, str) and src.startswith("gst:"):
            pipeline = src[4:]
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap

        if str(src).isdigit():
            idx = int(src)
            for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    return cap
        else:
            for backend in (cv2.CAP_GSTREAMER, cv2.CAP_ANY):
                cap = cv2.VideoCapture(src, backend)
                if cap.isOpened():
                    return cap
        return None

    def _build_runtime_inside_thread(self):
        if config.CV_BACKEND == "trt":
            from app.runtime_trt import DetectorTRT, PoseTRT

            self.detector = DetectorTRT()
            self.pose_model = PoseTRT() if config.ENABLE_POSE else None
        else:
            from app.runtime_ultra import DetectorUltra, PoseUltra

            self.detector = DetectorUltra()
            self.pose_model = PoseUltra() if config.ENABLE_POSE else None

    def _destroy_runtime(self):
        for obj in (self.pose_model, self.detector):
            if obj is not None and hasattr(obj, "destroy"):
                try:
                    obj.destroy()
                except Exception:
                    pass
        self.pose_model = None
        self.detector = None

    def _event_ready(self, key):
        now = time.time()
        prev = self.cooldowns.get(key, 0.0)
        if now - prev >= config.EVENT_COOLDOWN_SEC:
            self.cooldowns[key] = now
            return True
        return False

    def _draw_box(self, frame, bbox, label, color):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    def _log_event(
        self,
        event_type,
        track_id=None,
        bbox=None,
        posture=None,
        people_count=None,
        person_name=None,
        confidence=None,
        extra=None,
    ):
        self.logger.log_event(
            cam_id=config.CAMERA_ID,
            event_type=event_type,
            track_id=str(track_id) if track_id is not None else None,
            person_id=None,
            person_name=person_name,
            bbox=bbox,
            posture=posture,
            people_count=people_count,
            confidence=confidence,
            snapshot_path=None,
            extra=extra or {},
        )

    def _posture_confirm_frames(self, posture: str) -> int:
        if posture == "lying":
            return max(1, config.LYING_CONFIRM_FRAMES)
        if posture == "sitting":
            return max(1, config.SITTING_CONFIRM_FRAMES)
        if posture == "unknown":
            return max(1, config.POSTURE_CONFIRM_UNKNOWN_FRAMES)
        return 1

    def _run_loop(self):
        last_fps_ts = time.time()
        fps_count = 0
        last_person_dets = []
        last_bottle_dets = []
        last_poses = []

        try:
            self.cap = self._open_camera()
            if self.cap is None:
                self.status["error"] = (
                    "Khong mo duoc camera. Dat CAMERA_SOURCE=0 cho webcam USB, "
                    "hoac CAMERA_SOURCE='gst:<pipeline>' cho pipeline GStreamer."
                )
                return

            self._build_runtime_inside_thread()
            if self.detector is None:
                self.status["error"] = "Khong tao duoc detector runtime"
                return

            self.status["error"] = None

            while self.running:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    self.status["online"] = False
                    self.status["error"] = "Doc frame that bai tu camera"
                    time.sleep(0.1)
                    continue

                self.status["online"] = True
                self.status["last_frame_ts"] = datetime.now().isoformat()
                self.status["error"] = None
                self.frame_idx += 1
                fps_count += 1

                if time.time() - last_fps_ts >= 1.0:
                    self.status["fps"] = fps_count / max(time.time() - last_fps_ts, 1e-6)
                    fps_count = 0
                    last_fps_ts = time.time()

                if self.frame_idx % max(1, config.YOLO_EVERY_N) == 0:
                    dets = self.detector.predict(frame)
                    last_person_dets = [d for d in dets if d.get("cls") == COCO_PERSON]
                    last_bottle_dets = [d for d in dets if d.get("cls") == COCO_BOTTLE]

                if self.pose_model is not None and self.frame_idx % max(1, config.POSE_EVERY_N) == 0:
                    last_poses = self.pose_model.predict(frame)

                track_assignments = self.tracker.update([d["bbox"] for d in last_person_dets])
                bottle_assignments = self.bottle_tracker.update([d["bbox"] for d in last_bottle_dets])

                pose_by_track = {}
                for pose in last_poses:
                    best_tid, best_iou = None, 0.0
                    for tid, tb in track_assignments:
                        score = iou_xyxy(pose["bbox"], tb)
                        if score > best_iou:
                            best_tid, best_iou = tid, score
                    if best_tid is not None and best_iou >= 0.25:
                        pose_by_track[best_tid] = pose

                people_count = len(track_assignments)
                unknown_count = 0
                sitting_count = 0
                lying_count = 0
                fall_count = 0

                overload = people_count >= config.OVERLOAD_THRESHOLD
                self.status["overload"] = overload

                if people_count >= config.CROWD_THRESHOLD and self._event_ready(("CROWD", config.CAMERA_ID)):
                    self._log_event(
                        "CROWD",
                        people_count=people_count,
                        extra={"threshold": config.CROWD_THRESHOLD},
                    )

                if overload and self._event_ready(("OVERLOAD", config.CAMERA_ID)):
                    self._log_event(
                        "OVERLOAD",
                        people_count=people_count,
                        extra={"threshold": config.OVERLOAD_THRESHOLD},
                    )

                if len(bottle_assignments) > 0 and self._event_ready(("BOTTLE", config.CAMERA_ID)):
                    self._log_event(
                        "BOTTLE",
                        people_count=people_count,
                        extra={"count": len(bottle_assignments)},
                    )

                now_ts = time.time()

                for tid, bbox in track_assignments:
                    state = self.track_state.setdefault(
                        tid,
                        {
                            "posture": "unknown",
                            "posture_candidate": "unknown",
                            "candidate_streak": 0,
                            "last_upright_ts": now_ts,
                            "danger_until": 0.0,
                            "person_name": None,
                            "last_center": None,
                            "last_motion_ts": now_ts,
                            "posture_meta": {},
                        },
                    )

                    candidate_posture = "unknown"
                    posture_meta = {"ok": False, "reason": "no_pose"}
                    if tid in pose_by_track:
                        candidate_posture, posture_meta = classify_posture(
                            pose_by_track[tid]["keypoints"], bbox, return_meta=True
                        )

                    if candidate_posture == state["posture_candidate"]:
                        state["candidate_streak"] += 1
                    else:
                        state["posture_candidate"] = candidate_posture
                        state["candidate_streak"] = 1

                    confirmed_posture = state["posture"]
                    if candidate_posture != "unknown" and state["candidate_streak"] >= self._posture_confirm_frames(candidate_posture):
                        confirmed_posture = candidate_posture
                    elif candidate_posture == "unknown" and state["candidate_streak"] >= self._posture_confirm_frames("unknown"):
                        confirmed_posture = state["posture"]

                    prev_posture = state["posture"]
                    state["posture"] = confirmed_posture
                    state["posture_meta"] = posture_meta

                    x1, y1, x2, y2 = bbox
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    prev_center = state.get("last_center")
                    prev_motion_ts = state.get("last_motion_ts", now_ts)
                    dt = max(now_ts - prev_motion_ts, 1e-3)
                    center_drop_ratio = 0.0
                    vertical_speed = 0.0
                    if prev_center is not None:
                        center_drop = max(center[1] - prev_center[1], 0.0)
                        center_drop_ratio = center_drop / max(float(y2 - y1), 1.0)
                        vertical_speed = center_drop / dt

                    state["last_center"] = center
                    state["last_motion_ts"] = now_ts

                    if confirmed_posture in {"standing", "sitting"}:
                        state["last_upright_ts"] = now_ts

                    is_danger = False
                    event_extra = {
                        "posture_metrics": posture_meta,
                        "center_drop_ratio": round(center_drop_ratio, 3),
                        "vertical_speed": round(vertical_speed, 2),
                    }

                    if confirmed_posture == "unknown":
                        unknown_count += 1
                    elif confirmed_posture == "sitting":
                        sitting_count += 1
                    elif confirmed_posture == "lying":
                        lying_count += 1
                        is_danger = True
                        if self._event_ready(("LYING", tid)):
                            self._log_event(
                                "LYING",
                                track_id=tid,
                                bbox=bbox,
                                posture=confirmed_posture,
                                people_count=people_count,
                                confidence=posture_meta.get("posture_confidence"),
                                extra=event_extra,
                            )

                    fall_gate = (
                        confirmed_posture == "lying"
                        and is_fall_transition(prev_posture, confirmed_posture)
                        and (
                            center_drop_ratio >= config.FALL_MIN_CENTER_DROP_RATIO
                            or vertical_speed >= config.FALL_MIN_VERTICAL_SPEED
                            or now_ts - state["last_upright_ts"] <= config.FALL_MAX_TRANSITION_SEC
                        )
                    )

                    if fall_gate:
                        fall_count += 1
                        is_danger = True
                        state["danger_until"] = now_ts + config.DANGER_HOLD_SEC
                        if self._event_ready(("FALL", tid)):
                            self._log_event(
                                "FALL",
                                track_id=tid,
                                bbox=bbox,
                                posture=confirmed_posture,
                                people_count=people_count,
                                confidence=posture_meta.get("posture_confidence"),
                                extra=event_extra,
                            )

                    if now_ts < state["danger_until"]:
                        is_danger = True

                    person_name = state["person_name"] or ("track:%s" % tid)
                    posture_conf = posture_meta.get("posture_confidence")
                    posture_text = confirmed_posture
                    if posture_conf is not None:
                        posture_text = "%s %.2f" % (confirmed_posture, posture_conf)
                    color = (0, 0, 255) if is_danger else (0, 255, 0)
                    self._draw_box(frame, bbox, "%s | %s" % (person_name, posture_text), color)

                for _, bottle_bbox in bottle_assignments:
                    self._draw_box(frame, bottle_bbox, "bottle", (0, 140, 255))

                if overload:
                    cv2.putText(
                        frame,
                        "OVERLOAD",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )

                self.status["people_count"] = people_count
                self.status["unknown_count"] = unknown_count
                self.status["sitting_count"] = sitting_count
                self.status["lying_count"] = lying_count
                self.status["fall_count"] = fall_count

                if time.time() - self.last_occ_ts >= config.OCCUPANCY_SAMPLE_SEC:
                    self.last_occ_ts = time.time()
                    self.logger.log_occupancy(
                        cam_id=config.CAMERA_ID,
                        people_count=people_count,
                        unknown_count=unknown_count,
                        lying_count=lying_count,
                        fall_count=fall_count,
                        extra={
                            "fps": self.status["fps"],
                            "overload": overload,
                            "overload_threshold": config.OVERLOAD_THRESHOLD,
                            "sitting_count": sitting_count,
                        },
                    )

                ok, jpeg = cv2.imencode(".jpg", frame)
                if ok:
                    with self.lock:
                        self.latest_jpeg = jpeg.tobytes()

        except Exception as ex:
            self.status["online"] = False
            self.status["error"] = repr(ex)
            raise
        finally:
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
            self._destroy_runtime()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def mjpeg_generator(self):
        while True:
            with self.lock:
                frame = self.latest_jpeg
            if frame is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)

    def stop(self):
        self.running = False
        try:
            if self.thread is not None and self.thread.is_alive():
                self.thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self._destroy_runtime()

    def get_status(self):
        return self.status
