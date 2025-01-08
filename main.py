import cv2
import numpy as np
import mss
import pyautogui
import torch
from pynput.mouse import Listener
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QCheckBox, QPushButton, QLabel, QSlider, QHBoxLayout
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QPoint, QThread, pyqtSignal
import sys
import warnings
import time
import ctypes

pyautogui.FAILSAFE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# Параметры
global FOV, SMOOTH, triggerBot, aimBot, visualize, showFPS
FOV = 250  # Радиус действия (Field of View)
SMOOTH = 1  # Скорость реакции (чем больше, тем медленнее)
triggerBot = True
aimBot = True  # Включение аимбота
visualize = False
showFPS = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.hub.load(
    './yolov5',
    'custom',
    path='best.pt',
    source='local',
    device=device
)

WIDTH = 1920
HEIGHT = 1080
right_button_pressed = False

# Отключение задержек в pyautogui
pyautogui.PAUSE = 0

def on_mouse_button_press(x, y, button, pressed):
    global right_button_pressed
    if button == button.right:
        right_button_pressed = pressed

mouse_listener = Listener(on_click=on_mouse_button_press)
mouse_listener.start()



PUL = ctypes.POINTER(ctypes.c_ulong)

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("mi", MouseInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def send_input(dx, dy):
    """Используем SendInput для перемещения мыши."""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(dx, dy, 0, 0x0001, 0, ctypes.pointer(extra))
    command = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))



class DetectionThread(QThread):
    detections_updated = pyqtSignal(list, int)  # Передаем два отдельных аргумента: список и FPS

    def move_to_target(self, target):
        """Перемещаем мышь плавно к цели."""
        current_x, current_y = self.region['width'] // 2, self.region['height'] // 2
        target_x, target_y = target

        dx = int((target_x - current_x) / SMOOTH)
        dy = int((target_y - current_y) / SMOOTH)
        send_input(dx, dy)  # Используем низкоуровневое перемещение мыши

    def __init__(self, region):
        super().__init__()
        self.region = region
        self.running = True

    def run(self):
        global aimBot, triggerBot, right_button_pressed

        with mss.mss() as sct:
            while self.running:
                start_time = time.time()

                screen_capture = np.array(sct.grab(self.region))
                img = cv2.cvtColor(screen_capture, cv2.COLOR_BGRA2BGR)

                results = model(img)
                detections = results.xyxy[0].cpu().numpy()
                filtered_detections = []
                for det in detections:
                    x_min, y_min, x_max, y_max = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                    
                    height_ = y_max - y_min
                    # Фильтруем по ширине
                    if 1 <= height_ <= 500:
                        filtered_detections.append((x_min, y_min, x_max, y_max))


                # TriggerBot logic
                if triggerBot and right_button_pressed:
                    for det in filtered_detections:
                        x_min, y_min, x_max, y_max = det
                        if self.is_target_in_center(x_min, y_min, x_max, y_max):
                            pyautogui.click()
                            break

                # AimBot logic
                if aimBot and right_button_pressed:
                    best_target = self.find_best_target(filtered_detections)
                    if best_target:
                        self.move_to_target(best_target)

                fps = int(1 / (time.time() - start_time))
                self.detections_updated.emit(filtered_detections, fps)

    def stop(self):
        self.running = False
        self.wait()

    def is_target_in_center(self, x_min, y_min, x_max, y_max):
        center_x = self.region['width'] // 2
        center_y = self.region['height'] // 2
        return x_min <= center_x <= x_max and y_min <= center_y <= y_max

    def find_best_target(self, detections):
        """Находим лучшую цель для прицеливания."""
        center_x = self.region['width'] // 2
        center_y = self.region['height'] // 2
        best_target = None
        min_distance = FOV

        for x_min, y_min, x_max, y_max in detections:
            target_x = (x_min + x_max) // 2
            target_y = y_min + int((y_max - y_min) * 0.4)  # Целимся на 40% ниже верхней линии

            # Вычисляем расстояние до центра экрана
            distance = np.sqrt((target_x - center_x) ** 2 + (target_y - center_y) ** 2)
            if distance < min_distance:
                best_target = (target_x, target_y)
                min_distance = distance

        return best_target


class OverlayWindow(QMainWindow):
    def __init__(self, region):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.menu = CheatMenu(self)
        self.menu.setGeometry(600, 300, 400, 350)

        self.is_menu_visible = True
        self.enable_wallhack = False
        self.detections = []  # Для хранения объектов
        self.fps = 0

        self.detection_thread = DetectionThread(region)
        self.detection_thread.detections_updated.connect(self.update_detections)
        self.detection_thread.start()

        self.resize(1920, 1080)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Home:
            self.is_menu_visible = not self.is_menu_visible
            self.menu.setVisible(self.is_menu_visible)
        super().keyPressEvent(event)

    def update_detections(self, detections, fps):
        if visualize:
            self.detections = detections
        else:
            self.detections.clear()
        self.fps = fps
        self.update()

    def closeEvent(self, event):
        self.detection_thread.stop()
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(Qt.red if self.enable_wallhack else Qt.green)
        pen.setWidth(2)
        painter.setPen(pen)
        center_x, center_y = self.width() // 2, self.height() // 2

        # Рисуем FOV
        if visualize:
            painter.drawEllipse(center_x - FOV, center_y - FOV, FOV * 2, FOV * 2)

        # Рисуем найденные объекты
        for x_min, y_min, x_max, y_max in self.detections:
            painter.drawRect(x_min, y_min, x_max - x_min, y_max - y_min)

        # Рисуем FPS
        if showFPS:
            painter.setPen(QPen(Qt.white))
            painter.drawText(10, 30, f"FPS: {self.fps}")

class CheatMenu(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            background-color: #3A1A5E;
            border: 2px solid #7043A1;
            border-radius: 15px;
        """)

        self.parent = parent
        self.is_dragging = False
        self.drag_position = QPoint()

        self.header = QLabel("LostSouls | v 1.0 public build", self)
        self.header.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: bold;
            background-color: #5A2E85;
            padding: 10px;
            border-top-left-radius: 15px;
            border-top-right-radius: 15px;
        """)
        self.header.setAlignment(Qt.AlignCenter)

        fov_layout = QHBoxLayout()
        self.fov_label = QLabel(f"FOV: {FOV}", self)
        self.fov_label.setStyleSheet("color: white; font-size: 14px;")
        self.fov_slider = self.create_slider(FOV, 50, 500, self.update_fov)
        fov_layout.addWidget(self.fov_label)
        fov_layout.addWidget(self.fov_slider)

        smooth_layout = QHBoxLayout()
        self.smooth_label = QLabel(f"SMOOTH: {SMOOTH}", self)
        self.smooth_label.setStyleSheet("color: white; font-size: 14px;")
        self.smooth_slider = self.create_slider(SMOOTH, 1, 20, self.update_smooth)
        smooth_layout.addWidget(self.smooth_label)
        smooth_layout.addWidget(self.smooth_slider)

        self.triggerbot_checkbox = self.create_checkbox("TriggerBot", triggerBot, self.toggle_triggerbot)
        self.aimbot_checkbox = self.create_checkbox("AimBot", aimBot, self.toggle_aimbot)
        self.visualize_checkbox = self.create_checkbox("Visualize", visualize, self.toggle_visualize)
        self.fps_checkbox = self.create_checkbox("Show FPS", showFPS, self.toggle_fps)

        self.exit_button = QPushButton("Close", self)
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: #AA4444;
                color: white;
                font-size: 14px;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #CC5555;
            }
        """)
        self.exit_button.clicked.connect(self.exit_application)

        layout = QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addLayout(fov_layout)
        layout.addLayout(smooth_layout)
        layout.addWidget(self.triggerbot_checkbox)
        layout.addWidget(self.aimbot_checkbox)
        layout.addWidget(self.visualize_checkbox)
        layout.addWidget(self.fps_checkbox)
        layout.addWidget(self.exit_button)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(3)
        self.setLayout(layout)
        


    def create_slider(self, value, min_value, max_value, callback):
        slider = QSlider(Qt.Horizontal, self)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(value)
        slider.valueChanged.connect(callback)
        slider.setStyleSheet("""
            QSlider::handle:horizontal {
                background-color: #7043A1;
                border: 1px solid #5A2E85;
                width: 10px;
            }
            QSlider::groove:horizontal {
                background-color: #4A306D;
                height: 8px;
            }
        """)
        return slider

    def create_checkbox(self, name, checked, callback):
        checkbox = QCheckBox(name, self)
        checkbox.setChecked(checked)
        checkbox.setStyleSheet("color: white; font-size: 14px;")
        checkbox.stateChanged.connect(callback)
        return checkbox

    def update_fov(self, value):
        global FOV
        FOV = value
        self.fov_label.setText(f"FOV: {FOV}")

    def update_smooth(self, value):
        global SMOOTH
        SMOOTH = value
        self.smooth_label.setText(f"SMOOTH: {SMOOTH}")

    def toggle_triggerbot(self, state):
        global triggerBot
        triggerBot = state == Qt.Checked

    def toggle_aimbot(self, state):
        global aimBot
        aimBot = state == Qt.Checked

    def toggle_visualize(self, state):
        global visualize
        visualize = state == Qt.Checked

    def toggle_fps(self, state):
        global showFPS
        showFPS = state == Qt.Checked

    def exit_application(self):
        sys.exit()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            self.move(event.globalPos() - self.drag_position)

    def mouseReleaseEvent(self, event):
        self.is_dragging = False

def get_center_region(monitor_width, monitor_height):
    center_x = monitor_width // 2
    center_y = monitor_height // 2
    left = center_x - WIDTH // 2
    top = center_y - HEIGHT // 2
    return {"left": left, "top": top, "width": WIDTH, "height": HEIGHT}

def main():
    global right_button_pressed

    app = QApplication(sys.argv)
    monitor_width, monitor_height = pyautogui.size()
    region = get_center_region(monitor_width, monitor_height)

    overlay_window = OverlayWindow(region)
    overlay_window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
