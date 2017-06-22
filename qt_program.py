import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from mainwindow import Ui_MainWindow
from main2 import Run

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Flower Recognition using Deep Learning")
        
        # 변수 선언(초기값 VGG16)
        self.model = 'vgg16'
        
        # 액션 연결
        self.ui.actionOpen.triggered.connect(self.actionOpen)
        self.ui.actionExit.triggered.connect(self.actionExit)
        self.ui.actionRun.triggered.connect(self.actionRun)
        self.ui.actionVGG16.triggered.connect(self.actionVGG16)
        self.ui.actionVGG19.triggered.connect(self.actionVGG19)
        self.ui.actionResNet.triggered.connect(self.actionResNet)
        self.ui.actionInception_V3.triggered.connect(self.actionInception_V3)
        self.ui.actionXception.triggered.connect(self.actionXception)
        self.ui.actionNormal_Size.triggered.connect(self.actionNormalSize)
        self.ui.actionFit_to_Window.triggered.connect(self.actionFit_to_Window)
        
        # UI 초기화
        self.ui.actionNormal_Size.setEnabled(False)
        self.ui.actionFit_to_Window.setEnabled(False)
        self.ui.actionRun.setEnabled(False)
        
        self.ui.label.setBackgroundRole(QPalette.Base)
        self.ui.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.ui.label.setScaledContents(True)
        
        # 꽃 이름 
        self.class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",
                       "iris", "tigerlily", "tulip", "fritillary", "sunflower", 
                       "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
                       "windflower", "pansy", "Enchinacea", "Frangipani", "Ipomoea_pandurata", "mugunghwa",
                       "Nymphaea_odorata"]
    
    def actionOpen(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        self.fileName = fileName
        if fileName:
            self.image = QImage(fileName)
            if self.image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return

            self.ui.label.setPixmap(QPixmap.fromImage(self.image))
            self.ui.label.adjustSize()
            #self.ui.label.scaleFactor = 1.0

            self.ui.actionNormal_Size.setEnabled(True)
            self.ui.actionFit_to_Window.setEnabled(True)
            self.ui.actionRun.setEnabled(True)
            #self.detectAct.setEnabled(True)
            #self.updateActions()

            #if not self.fitToWindowAct.isChecked():
            #    self.imageLabel.adjustSize()
                
    def actionExit(self):
        sys.exit()
        
    def actionNormalSize(self):
        self.ui.label.adjustSize()
        self.ui.label.scaleFactor = 1.0
        
    def actionFit_to_Window(self):
        myPixmap = QPixmap.fromImage(self.image)
        myScaledPixmap = myPixmap.scaled(800, 600, Qt.KeepAspectRatio)
        self.ui.label.setPixmap(myScaledPixmap)
    
    def actionRun(self):
        pred, prob = Run(self, self.fileName, self.model)
        
        flower_name = self.class_names[pred[0]] + ' ' + str(prob)
        QMessageBox.information(self, "Result", flower_name)
        print(flower_name)
    
    def actionVGG16(self):
        self.model = 'vgg16'
    
    def actionVGG19(self):
        self.model = 'vgg19'
    
    def actionResNet(self):
        self.model = 'resnet50'
    
    def actionInception_V3(self):
        self.model = 'inceptionv3'
    
    def actionXception(self):
        self.model = 'xception'
    
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = UI()
    window.show()
    sys.exit(app.exec_())