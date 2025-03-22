import sys
import os
import csv
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QFileDialog, QLabel
from PySide6.QtCore import QTimer
# from PySide6.QtCore import QMetaObject, Qt
# from PySide6.QtCore import QTimer

from socketPot import socketPot, globalMemory
from ui_form import Ui_MainWindow  # UI 파일을 사용 (미리 Qt Designer에서 생성해야 함)

# from socketPot import socket_pot_signals

#  pyside6-uic ui_form.ui -o ui_form.py

gv = globalMemory.GlobalVariableT24
dc = globalMemory.GlobalDataContainer
aa = socketPot.socket_pot_signals

def parse_file(filename):
    data = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("["):  # 빈 줄과 헤더 무시
                continue
            if '=' in line:  # '='이 포함된 줄만 처리
                key, _, value = line.partition('=')  
                data[int(key.strip())] = int(value.strip())  
    return data

def extract_ascii_values(data, start_key, count=5):
    keys = sorted(data.keys())
    start_index = keys.index(start_key) if start_key in keys else -1
    if start_index == -1 or start_index + count > len(keys):
        return "Invalid key range"
    
    ascii_chars = [chr(data[keys[i]]) for i in range(start_index, start_index + count)]
    return ''.join(ascii_chars)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        aa.log_signal.connect(self.log_message)


        # self.ui.log_output = QTextEdit(self)
        # self.ui.log_output.setGeometry(120, 350, 440, 130)
        # # self.ui.log_output.setReadOnly(True)

        self.pot = socketPot.PotConnection()
        
        # if self.pot.isConnected() == False: 
        #    self.pot.connect()

        self.connection_timer = QTimer(self)  
        self.connection_timer.timeout.connect(self.check_connection) 

        self.init_connection()
        
        # self.ui.PID = QTextEdit(self) 
        # self.ui.PID.setGeometry(500, 40, 100, 30)
        # self.ui.PID.setReadOnly(True)  

        # self.ui.PID_label = QLabel(self)

        self.load_ascii_data()
    
    def load_ascii_data(self):
        filename = "output/[L Parmeter] rework.ini"  
        start_key = 10         
        
        data = parse_file(filename)
        ascii_text = extract_ascii_values(data, start_key)
        self.ui.PID.setText(ascii_text)
        # self.ui.PID_label.setText(ascii_text)

    def init_connection(self):

        if not self.pot.isConnected():
            self.connection_timer.start(2000) 
        else:
            self.update_button_status()

    def check_connection(self):
        if not self.pot.isConnected():
            print("연결 시도 중...")
            self.pot.connect()
        else:
            print("연결 완료!")
            self.connection_timer.stop()  # 타이머 정지
            self.update_button_status()  # 버튼 색상 변경

    def update_button_status(self):
         self.ui.read_1_button.setStyleSheet("background-color: green; color: white;")            
            

    # def log_message(self, msg):       
    #     QTimer.singleShot(0, lambda: self.update_log(msg))

    # def update_log(self, msg):       
    #     self.ui.log_output.append(msg)

    def log_message(self, message):
        self.ui.log_output.append(message)

            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
