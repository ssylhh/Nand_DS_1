# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_form.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGroupBox, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QStatusBar, QTextEdit,
    QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(751, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(120, 70, 211, 261))
        self.verticalLayout = QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.read_1_button = QPushButton(self.groupBox)
        self.read_1_button.setObjectName(u"read_1_button")

        self.verticalLayout.addWidget(self.read_1_button)

        self.read_2_button = QPushButton(self.groupBox)
        self.read_2_button.setObjectName(u"read_2_button")

        self.verticalLayout.addWidget(self.read_2_button)

        self.read_3_button = QPushButton(self.groupBox)
        self.read_3_button.setObjectName(u"read_3_button")

        self.verticalLayout.addWidget(self.read_3_button)

        self.read_4_button = QPushButton(self.groupBox)
        self.read_4_button.setObjectName(u"read_4_button")

        self.verticalLayout.addWidget(self.read_4_button)

        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(350, 70, 211, 261))
        self.verticalLayout_2 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.write_1_button = QPushButton(self.groupBox_2)
        self.write_1_button.setObjectName(u"write_1_button")

        self.verticalLayout_2.addWidget(self.write_1_button)

        self.write_2_button = QPushButton(self.groupBox_2)
        self.write_2_button.setObjectName(u"write_2_button")

        self.verticalLayout_2.addWidget(self.write_2_button)

        self.write_3_button = QPushButton(self.groupBox_2)
        self.write_3_button.setObjectName(u"write_3_button")

        self.verticalLayout_2.addWidget(self.write_3_button)

        self.write_4_button = QPushButton(self.groupBox_2)
        self.write_4_button.setObjectName(u"write_4_button")

        self.verticalLayout_2.addWidget(self.write_4_button)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(130, 20, 371, 31))
        font = QFont()
        font.setFamilies([u"\uad81\uc11c"])
        font.setPointSize(16)
        font.setBold(False)
        self.label.setFont(font)
        self.PID = QTextEdit(self.centralwidget)
        self.PID.setObjectName(u"PID")
        self.PID.setGeometry(QRect(500, 20, 111, 31))
        font1 = QFont()
        font1.setFamilies([u"Arial"])
        font1.setPointSize(9)
        font1.setBold(True)
        self.PID.setFont(font1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Reading ", None))
        self.read_1_button.setText(QCoreApplication.translate("MainWindow", u"(V / L) Parameter", None))
        self.read_2_button.setText(QCoreApplication.translate("MainWindow", u"S Parameter", None))
        self.read_3_button.setText(QCoreApplication.translate("MainWindow", u"G Parameter", None))
        self.read_4_button.setText(QCoreApplication.translate("MainWindow", u"Others ", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Writing", None))
        self.write_1_button.setText(QCoreApplication.translate("MainWindow", u"(V / L) Parameter", None))
        self.write_2_button.setText(QCoreApplication.translate("MainWindow", u"S Parameter", None))
        self.write_3_button.setText(QCoreApplication.translate("MainWindow", u"G Parameter", None))
        self.write_4_button.setText(QCoreApplication.translate("MainWindow", u"Others ", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Rework Program for T24", None))
    # retranslateUi

