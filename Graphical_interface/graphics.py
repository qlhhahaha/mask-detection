import sys
#PyQt5中使用的基本控件都在PyQt5.QtWidgets模块中
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox,QInputDialog,QFileDialog
#导入designer工具生成的login模块
from interface import Ui_Interface
import torch
import os
import shutil
import cv2

class MyMainForm(QMainWindow, Ui_Interface):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.file_directory = os.path.abspath(os.path.dirname(os.getcwd()))
        self.output_size = 640
        self.model_load()
        self.control()
    def control(self):
        self.img_up.clicked.connect(self.img_upload)
        self.img_dec.clicked.connect(self.img_detect)
    def model_load(self):
        pathtoyolov5 = self.file_directory+'/yolov5'
        pathtobest = self.file_directory+'/yolov5/best.pt'
        self.model = torch.hub.load(pathtoyolov5, 'custom', path=pathtobest, source='local')
    def img_upload(self):
        get_filename_path, ok = QFileDialog.getOpenFileName(self,
                                                            "选取单个文件",
                                                            "D:/",
                                                            "Image Files (*.jpg *.png *.tif *.jpeg)")
        if ok:
            suffix = get_filename_path.split(".")[-1]
            save_path = os.path.join("images/temp", "init." + suffix)
            shutil.copy(get_filename_path, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            init_img = cv2.imread(save_path)
            resize_scale = self.output_size / init_img.shape[0]
            processed_img = cv2.resize(init_img, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/temp/processed.jpg", processed_img)
            self.ini_img.setPixmap(QtGui.QPixmap("images/temp/processed.jpg"))
            self.ini_img.setScaledContents(True)
            b = 0
    def img_detect(self):
        model = self.model
        img = cv2.imread('images/temp/processed.jpg')[..., ::-1]
        results = model(img, size=640)
        results.save(save_dir='images/temp')
        self.pro_img.setPixmap(QtGui.QPixmap("images/temp/image0.jpg"))
        self.pro_img.setScaledContents(True)




if __name__ == "__main__":
    #固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    #初始化
    myWin = MyMainForm()
    #将窗口控件显示在屏幕上
    myWin.show()
    #程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
