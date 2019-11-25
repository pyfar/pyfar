from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QGridLayout, QLineEdit, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator


class AxisDialog(QDialog):
    """Qt GUI to update the axix limits"""

    def __init__(self, parent, axes):
        super().__init__(parent)

        # Get axes info
        self.axes = axes
        self.xlim = self.axes.get_xlim()
        self.ylim = self.axes.get_ylim()

        self.setWindowTitle('Update Axis Limits')
        self.grid = QGridLayout(self)

        # Create widgets
        self.label_x = QLabel("x-axis")
        self.label_y = QLabel("y-axis")
        self.label_start = QLabel("Start")
        self.label_stop = QLabel("Stop")
        # self.label_delta = QLabel("Delta")
        self.edit_xmin = QLineEdit("{:0.5f}".format(self.xlim[0]))
        self.edit_xmax = QLineEdit("{:0.5f}".format(self.xlim[1]))
        # self.edit_xdelta = QLineEdit()
        self.edit_ymin = QLineEdit("{:0.5f}".format(self.ylim[0]))
        self.edit_ymax = QLineEdit("{:0.5f}".format(self.ylim[1]))
        # self.edit_ydelta = QLineEdit()

        self.edit_xmin.setValidator(QDoubleValidator())
        self.edit_xmax.setValidator(QDoubleValidator())
        self.edit_ymin.setValidator(QDoubleValidator())
        self.edit_ymax.setValidator(QDoubleValidator())

        # OK and Cancel buttons
        self.buttonbox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)

        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        # Add widgets to grid
        self.grid.addWidget(self.label_start, 0, 1, 1, 1)
        self.grid.addWidget(self.label_stop, 0, 2, 1, 1)
        # self.grid.addWidget(self.label_delta, 0, 3, 1, 1)
        self.grid.addWidget(self.label_x, 1, 0, 1, 1)
        self.grid.addWidget(self.label_y, 2, 0, 1, 1)
        self.grid.addWidget(self.edit_xmin, 1, 1, 1, 1)
        self.grid.addWidget(self.edit_xmax, 1, 2, 1, 1)
        # self.grid.addWidget(self.edit_xdelta, 1, 3, 1, 1)
        self.grid.addWidget(self.edit_ymin, 2, 1, 1, 1)
        self.grid.addWidget(self.edit_ymax, 2, 2, 1, 1)
        # self.grid.addWidget(self.edit_ydelta, 2, 3, 1, 1)
        self.grid.addWidget(self.buttonbox, 3, 1, 1, 2)

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def update_axis(axes, parent=None):
        dialog = AxisDialog(parent, axes)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            xlim = (float(dialog.edit_xmin.text()),
                    float(dialog.edit_xmax.text()))
            ylim = (float(dialog.edit_ymin.text()),
                    float(dialog.edit_ymax.text()))
        else:
            xlim = dialog.axes.get_xlim()
            ylim = dialog.axes.get_ylim()

        return (xlim, ylim, result == QDialog.Accepted)
