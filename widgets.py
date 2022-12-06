import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QLabel,
    QProgressDialog,
)


class CustomDialog(QDialog):
    def __init__(self, title="Hello", content="Dialog box", showCancel=False):
        super().__init__()

        # initilaze the parameters
        self.title = title
        self.content = content

        self.setWindowTitle(self.title)

        QBtn = QDialogButtonBox.Ok
        if showCancel:
            QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel(self.content)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class CustomProgress(QProgressDialog):
    def __init__(self, title, bodyText):
        super().__init__()

        # initlialize the parameters
        self.title = title
        self.bodyText = bodyText
