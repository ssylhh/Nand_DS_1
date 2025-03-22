import time
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QWidget
from serial import Serial
from PyQt5.QtSerialPort import QSerialPortInfo

class Receiver(QThread):

    def __init__(self, parent=None):
        super(Receiver, self).__init__(parent)
        self.serial_monitor = parent
        self.is_running = False
        print("쓰레드 초기화 완료!")

    def run(self):
        while self.serial_monitor.dev is None:
            time.sleep(0.1)  # CPU 점유율 방지

        while not self.serial_monitor.dev.isOpen():
            time.sleep(0.1)

        print("수신 쓰레드 시작!")
        self.is_running = True

        while self.is_running:
            rx_msg = self.serial_monitor.dev.readline()
            if rx_msg:
                decoded_msg = rx_msg.decode('ascii', errors='ignore').strip()
                self.serial_monitor.textEdit.append(decoded_msg)
                print(decoded_msg)

class SerialMonitor(QWidget):
    def __init__(self, parent=None):
        super(SerialMonitor, self).__init__(parent)
        self.setupUi(self)

        port_list = QSerialPortInfo().availablePorts()
        for i, port_info in enumerate(port_list):
            self.comboBox.insertItem(i, port_info.portName())

        baudrate_list = ['9600', '57600', '115200', '128000']
        for i, baudrate_info in enumerate(baudrate_list):
            self.comboBox_2.insertItem(i, baudrate_info)

        self.dev = None
        self.receiver = Receiver(self)

    def port_open(self):
        current_port = self.comboBox.currentText()
        current_baudrate = int(self.comboBox_2.currentText())

        if self.dev and self.dev.isOpen():
            print("이미 포트가 열려 있습니다.")
            return

        try:
            self.dev = Serial(
                port=current_port,
                baudrate=current_baudrate,
                parity='N',
                stopbits=1,
                bytesize=8,
                timeout=8
            )
            print("포트가 성공적으로 열렸습니다.")

            if not self.receiver.isRunning():
                self.receiver.start()

        except Exception as e:
            print(f"포트 열기 실패: {e}")

    def port_close(self):
        if self.dev and self.dev.isOpen():
            self.receiver.is_running = False
            self.dev.close()
            self.dev = None
            print("포트가 닫혔습니다.")
        else:
            print("포트가 이미 닫혀 있습니다.")

    def serial_write(self):
        if self.dev and self.dev.isOpen():
            tx_msg = self.lineEdit.text().strip() + "\r\n"
            self.dev.write(tx_msg.encode("ascii"))
            self.textEdit.append(f"TX >> {tx_msg.strip()}")

    def clear(self):
        self.textEdit.clear()

    def text_copy(self):
        self.text = self.textEdit.toPlainText()
        self.textEdit_2.setText(self.text)
        extracted_lines = [line for line in self.text.splitlines() if "K24 FW Start" in line]
        for line in extracted_lines:
            print(line)