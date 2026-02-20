import cv2
import mediapipe as mp
import pyautogui
import time
import math
import os

# ================= MediaPipe Setup =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# ================= Variables =================
freeze_cursor = False
scroll_active = False
prev_hand_y = None

thumb_index_times = []
cluster_cooldown = 0

screen_width, screen_height = pyautogui.size()

CURSOR_SPEED = 1.9
SCROLL_SPEED = 1600

os.makedirs("screenshots", exist_ok=True)

print("Hand Control Started")

# ================= Camera =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not open")
    exit()

# ================= Main Loop =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark

            # ---------------- Fingertips ----------------
            tips = [4, 8, 12, 16, 20]
            pts = [lm[i] for i in tips]

            thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip = pts

            # ---------------- Finger state (for fist) ----------------
            finger = []
            finger.append(1 if lm[4].x > lm[3].x else 0)
            for i in [8, 12, 16, 20]:
                finger.append(1 if lm[i].y < lm[i - 2].y else 0)

            is_fist = (sum(finger) == 0)

            now = time.time()

            # ==================================================
            # 📸 SCREENSHOT (Thumb + Index DOUBLE pinch)
            # ==================================================
            thumb_index_dist = math.hypot(
                thumb_tip.x - index_tip.x,
                thumb_tip.y - index_tip.y
            )

            if thumb_index_dist < 0.05:
                if not freeze_cursor:
                    freeze_cursor = True
                    thumb_index_times.append(now)

                    if len(thumb_index_times) >= 2 and (thumb_index_times[-1] - thumb_index_times[-2]) < 0.4:
                        filename = f"screenshots/screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                        pyautogui.screenshot(filename)
                        thumb_index_times.clear()

                        cv2.putText(frame, "SCREENSHOT", (40, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                freeze_cursor = False

            # ==================================================
            # 🤌 OPEN / CLOSE (ALL FINGERTIPS TOGETHER)
            # ==================================================
            cx = sum(p.x for p in pts) / 5
            cy = sum(p.y for p in pts) / 5

            distances = [
                math.hypot(p.x - cx, p.y - cy)
                for p in pts
            ]

            if max(distances) < 0.045:
                if now - cluster_cooldown > 1.2:
                    pyautogui.doubleClick()
                    cluster_cooldown = now

                    cv2.putText(frame, "OPEN / CLOSE", (40, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ==================================================
            # ✊ FAST GRAB SCROLL
            # ==================================================
            if is_fist:
                if not scroll_active:
                    scroll_active = True
                    prev_hand_y = index_tip.y
                else:
                    dy = index_tip.y - prev_hand_y
                    pyautogui.scroll(int(dy * -SCROLL_SPEED))
                    prev_hand_y = index_tip.y

                    cv2.putText(frame, "SCROLLING", (40, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                scroll_active = False
                prev_hand_y = None

            # ==================================================
            # 🖱️ FAST CURSOR MOVE
            # ==================================================
            if not freeze_cursor and not scroll_active:
                center_x = screen_width // 2
                center_y = screen_height // 2

                dx = (index_tip.x * screen_width - center_x) * CURSOR_SPEED
                dy = (index_tip.y * screen_height - center_y) * CURSOR_SPEED

                pyautogui.moveTo(
                    int(center_x + dx),
                    int(center_y + dy),
                    duration=0.01
                )

    cv2.imshow("Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= Cleanup =================
cap.release()
cv2.destroyAllWindows()




