"""
AutoBot 控制系统 - 游戏风格 PyQt5 图形界面
"""

import sys
import time
from pathlib import Path

from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

# 类型辅助：消除 Pylance 对 QProcess 的误报
import typing
if typing.TYPE_CHECKING:
    from PyQt5.QtCore import QProcess as QProcessType
else:
    QProcessType = QProcess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_SCRIPT = PROJECT_ROOT / "main.py"

# 游戏风格样式 - 深色背景 + 霓虹发光
STYLE = """
QMainWindow {
    background-color: #0a0e17;
}
QGroupBox {
    font-size: 13px;
    font-weight: bold;
    color: #00f0ff;
    border: 1px solid #00f0ff;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 14px;
    background-color: rgba(0, 240, 255, 0.03);
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #00f0ff;
}
QPushButton {
    border: 1px solid #00f0ff;
    border-radius: 6px;
    padding: 10px 28px;
    font-size: 14px;
    font-weight: bold;
    color: #00f0ff;
    background: rgba(0, 240, 255, 0.08);
}
QPushButton:hover {
    background: rgba(0, 240, 255, 0.2);
    border-color: #00ff88;
    color: #00ff88;
}
QPushButton:pressed {
    background: rgba(0, 240, 255, 0.3);
}
QPushButton:disabled {
    border-color: #1a2a4a;
    color: #1a2a4a;
    background: transparent;
}
QComboBox {
    border: 1px solid #00f0ff;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 22px;
    background: #0d1520;
    color: #e0e0e0;
}
QComboBox:hover {
    border-color: #00ff88;
}
QComboBox::drop-down {
    border: none;
    background: #0d1520;
}
QComboBox QAbstractItemView {
    background: #0d1520;
    color: #e0e0e0;
    selection-background-color: #00f0ff;
    selection-color: #0a0e17;
}
QDoubleSpinBox {
    border: 1px solid #00f0ff;
    border-radius: 4px;
    padding: 4px;
    min-height: 22px;
    background: #0d1520;
    color: #e0e0e0;
}
QDoubleSpinBox:hover {
    border-color: #00ff88;
}
QPlainTextEdit {
    background: #050810;
    color: #e0e0e0;
    border: 1px solid #00f0ff;
    border-radius: 6px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    padding: 6px;
}
QCheckBox {
    color: #c0c0c0;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #00f0ff;
    border-radius: 3px;
    background: transparent;
}
QCheckBox::indicator:checked {
    background: #00f0ff;
    border-color: #00f0ff;
}
QCheckBox::indicator:hover {
    border-color: #00ff88;
}
QRadioButton {
    color: #c0c0c0;
    spacing: 8px;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #00f0ff;
    border-radius: 8px;
    background: transparent;
}
QRadioButton::indicator:checked {
    background: #00f0ff;
    border-color: #00f0ff;
}
QRadioButton::indicator:hover {
    border-color: #00ff88;
}
QLabel {
    color: #c0c0c0;
}
"""


class ConsoleOutput(QPlainTextEdit):
    """只读控制台输出。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)

    def append_message(self, text: str):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text + "\n")
        self.setTextCursor(cursor)
        self.ensureCursorVisible()


class AutoBotGUI(QMainWindow):
    """AutoBot 主窗口 - 游戏风格。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoBot 控制系统")
        self.setMinimumSize(750, 550)
        self.resize(850, 650)
        self.setStyleSheet(STYLE)

        self.process = None
        self.process_running = False

        self._setup_ui()
        self._log("系统初始化完成，等待指令...")

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # 标题 - 霓虹发光效果
        title = QLabel("AUTOBOT 控制系统")
        title.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #00f0ff;"
            "padding: 8px 0; letter-spacing: 3px;"
        )
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # 副标题
        subtitle = QLabel("CS:GO 自动导航与战斗系统")
        subtitle.setStyleSheet(
            "font-size: 12px; color: #00ff88; padding-bottom: 4px; letter-spacing: 1px;"
        )
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)

        # 参数区
        params_group = QGroupBox("参数配置")
        params_layout = QHBoxLayout(params_group)
        params_layout.setSpacing(24)

        # 左侧
        left_layout = QVBoxLayout()
        form = QFormLayout()
        form.setSpacing(8)

        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.addItems(["COM3", "COM4", "COM5", "COM6", "COM7", "COM8"])
        self.port_combo.setCurrentText("COM6")
        form.addRow("Arduino 端口:", self.port_combo)

        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(["9600", "115200", "57600"])
        self.baudrate_combo.setCurrentText("9600")
        form.addRow("波特率:", self.baudrate_combo)

        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(20, 500)
        self.spin_speed.setValue(120)
        self.spin_speed.setSuffix(" px/s")
        form.addRow("移动速度:", self.spin_speed)

        left_layout.addLayout(form)

        self.chk_arduino = QCheckBox("启用 Arduino 控制（无 Arduino 请取消勾选）")
        self.chk_arduino.setChecked(False)
        left_layout.addWidget(self.chk_arduino)

        self.chk_verbose = QCheckBox("详细输出模式")
        self.chk_verbose.setChecked(True)
        left_layout.addWidget(self.chk_verbose)

        params_layout.addLayout(left_layout)

        # 右侧
        right_layout = QVBoxLayout()

        right_layout.addWidget(QLabel("阵营选择:"))
        self.radio_auto = QRadioButton("自动识别")
        self.radio_ct = QRadioButton("CT - 反恐精英")
        self.radio_t = QRadioButton("T - 恐怖分子")
        self.radio_auto.setChecked(True)
        right_layout.addWidget(self.radio_auto)
        right_layout.addWidget(self.radio_ct)
        right_layout.addWidget(self.radio_t)

        right_layout.addSpacing(10)
        self.chk_combat = QCheckBox("启用战斗模式")
        self.chk_combat.setChecked(True)
        right_layout.addWidget(self.chk_combat)

        self.chk_site_guard = QCheckBox("到达 A 点后守点")
        self.chk_site_guard.setChecked(True)
        right_layout.addWidget(self.chk_site_guard)

        right_layout.addStretch()
        params_layout.addLayout(right_layout)

        main_layout.addWidget(params_group)

        # 按钮区
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(16)

        self.btn_start = QPushButton("▶ 启动系统")
        self.btn_start.setStyleSheet(
            "QPushButton { border: 2px solid #00ff88; color: #00ff88;"
            " background: rgba(0, 255, 136, 0.1); padding: 12px 40px;"
            " font-size: 15px; }"
            "QPushButton:hover { background: rgba(0, 255, 136, 0.25);"
            " border-color: #00ff88; }"
            "QPushButton:disabled { border-color: #1a3a2a; color: #1a3a2a;"
            " background: transparent; }"
        )
        self.btn_start.clicked.connect(self._start_bot)

        self.btn_stop = QPushButton("■ 停止系统")
        self.btn_stop.setStyleSheet(
            "QPushButton { border: 2px solid #ff4466; color: #ff4466;"
            " background: rgba(255, 68, 102, 0.1); padding: 12px 40px;"
            " font-size: 15px; }"
            "QPushButton:hover { background: rgba(255, 68, 102, 0.25);"
            " border-color: #ff4466; }"
            "QPushButton:disabled { border-color: #3a1a22; color: #3a1a22;"
            " background: transparent; }"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_bot)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # 控制台
        console_group = QGroupBox("系统日志")
        console_layout = QVBoxLayout(console_group)
        self.console = ConsoleOutput()
        console_layout.addWidget(self.console)
        main_layout.addWidget(console_group, 1)

    def _log(self, text: str):
        timestamp = time.strftime("%H:%M:%S")
        self.console.append_message(f"[{timestamp}] {text}")

    def _build_args(self):
        args = [sys.executable, str(MAIN_SCRIPT)]

        if self.chk_arduino.isChecked():
            args.append("--arduino")
            args.append("--arduino-port")
            args.append(self.port_combo.currentText())
            args.append("--baudrate")
            args.append(self.baudrate_combo.currentText())

        if self.chk_verbose.isChecked():
            args.append("--verbose")

        args.append("--movement-speed")
        args.append(str(self.spin_speed.value()))

        if self.radio_ct.isChecked():
            args.append("--role")
            args.append("ct")
        elif self.radio_t.isChecked():
            args.append("--role")
            args.append("t")

        if not self.chk_combat.isChecked():
            args.append("--disable-combat")

        if not self.chk_site_guard.isChecked():
            args.append("--disable-site-guard")

        return args

    def _start_bot(self):
        if self.process_running:
            return

        args = self._build_args()
        self._log(f"启动命令: {' '.join(args)}")

        self.process = QProcess()
        from PyQt5.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)
        self.process.start(args[0], args[1:])

        self.process_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._log("▶ 系统已启动")

    def _stop_bot(self):
        if self.process and self.process_running:
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()
            self._log("■ 系统已停止")

    def _on_stdout(self):
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line.strip():
                self._log(line)

    def _on_stderr(self):
        data = self.process.readAllStandardError().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line.strip():
                self._log(f"ERR: {line}")

    def _on_finished(self):
        self.process_running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._log("系统进程已退出")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # 设置全局字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    window = AutoBotGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
