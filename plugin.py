from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon


class VisMatchGeorefPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dlg = None

    def initGui(self):
        self.action = QAction(QIcon(), "VisMatch Georeferencer", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToRasterMenu("&VisMatch Georeferencer", self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        self.iface.removePluginRasterMenu("&VisMatch Georeferencer", self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.dlg:
            self.dlg.close()
            self.dlg = None

    def run(self):
        from .dialog import VisMatchDialog
        if self.dlg is None:
            self.dlg = VisMatchDialog(self.iface, self.iface.mainWindow())
        self.dlg.show()
        self.dlg.raise_()
        self.dlg.activateWindow()
