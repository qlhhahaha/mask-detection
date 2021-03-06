import sys
#PyQt5中使用的基本控件都在PyQt5.QtWidgets模块中
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox,QInputDialog,QFileDialog
#导入designer工具生成的login模块
from interface import Ui_Interface
import torch
import os
import shutil
import cv2
import threading
import sys
from pathlib import Path
import torch.backends.cudnn as cudnn
# sys.path.append('D://program//project//python//Mask-detection//yolov5//models')
# sys.path.append('D://program//project//python//Mask-detection//yolov5//utils')

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

class MyMainForm(QMainWindow, Ui_Interface):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        #初始化界面中的图片
        self.ui_img(par=1)
        self.ui_img(par=2)
        self.ui_img(par=3)
        self.file_directory = os.path.abspath(os.path.dirname(os.getcwd()))
        self.output_size = 360
        self.device = 'cpu'
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        self.model_load(device=self.device)
        self.control()
        b=0

    def control(self):
        self.img_up.clicked.connect(self.img_upload)
        self.img_dec.clicked.connect(self.img_detect)
        self.vid_up.clicked.connect(self.vid_upload)
        self.vid_off.clicked.connect(self.video_off)
        self.cam_on.clicked.connect(self.camera_on)
        self.cam_off.clicked.connect(self.camara_off)


    def ui_img(self,par=1):
        if par == 1:
            self.ini_img.setPixmap(QtGui.QPixmap("images/cover/image0.jpg"))
            self.pro_img.setPixmap(QtGui.QPixmap("images/cover/image1.jpg"))
            self.ini_img.setAlignment(Qt.AlignCenter)
            self.pro_img.setAlignment(Qt.AlignCenter)
            # self.ini_img.setScaledContents(True)
            # self.pro_img.setScaledContents(True)
        elif par == 2:
            self.vid.setPixmap(QtGui.QPixmap("images/cover/video.png"))
            self.vid.setAlignment(Qt.AlignCenter)
        elif par == 3:
            self.cam.setPixmap(QtGui.QPixmap("images/cover/camera.png"))
            self.cam.setAlignment(Qt.AlignCenter)


    def model_load(self,
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):
        pathtoyolov5 = self.file_directory+'/yolov5'
        pathtobest = self.file_directory+'/yolov5/best.pt'
        self.model1 = torch.hub.load(pathtoyolov5, 'custom', path=pathtobest, source='local')
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        device = select_device(device)
        model2 = DetectMultiBackend(pathtobest, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model2.stride, model2.names, model2.pt, model2.jit, model2.onnx
        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model2.model.half() if half else model2.model.float()
        self.model2 =model2

    def img_upload(self):
        self.ui_img(par=1)
        get_filename_path, ok = QFileDialog.getOpenFileName(self,
                                                            "选取单个文件",
                                                            "D:/",
                                                            "Image Files (*.jpg *.png *.tif *.jpeg)")
        if ok:
            suffix = get_filename_path.split(".")[-1]
            save_path = os.path.join("images/temp", "init." + suffix)
            shutil.copy(get_filename_path, save_path)
            # 应该调整一下图片的大小，然后统一放在一起
            init_img = cv2.imread(save_path)
            resize_scale = self.output_size / init_img.shape[0]
            processed_img = cv2.resize(init_img, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/temp/processed.jpg", processed_img)
            self.ini_img.setPixmap(QtGui.QPixmap("images/temp/processed.jpg"))
            # self.ini_img.setScaledContents(True)
            self.ini_img.setAlignment(Qt.AlignCenter)

    def img_detect(self):
        model = self.model1
        img = cv2.imread('images/temp/processed.jpg')[..., ::-1]
        results = model(img, size=640)
        results.save(save_dir='images/temp')
        self.pro_img.setPixmap(QtGui.QPixmap("images/temp/image0.jpg"))
        # self.pro_img.setScaledContents(True)
        self.pro_img.setAlignment(Qt.AlignCenter)


    def vid_upload(self):
        get_filename_path, ok = QFileDialog.getOpenFileName(self,
                                                            "选取单个文件",
                                                            "D:/",
                                                            "Image Files (*.mp4 *.avi)")
        if ok:
            # self.webcam_detection_btn.setEnabled(False)
            # self.mp4_detection_btn.setEnabled(False)
            # self.vid_stop_btn.setEnabled(True)
            self.vid_source = get_filename_path
            self.webcam = False
            th = threading.Thread(target=self.vid_detect(vid_source = 'video'))
            th.start()

    def video_off(self):
        self.stopEvent.set()
        self.ui_img(par=2)
        self.vid_source = '0'
        self.webcam = True

    def camera_on(self):
        self.vid_source = '0'
        self.webcam = True
        th = threading.Thread(target=self.vid_detect(vid_source = 'camera'))
        th.start()

    def camara_off(self):
        self.stopEvent.set()
        self.ui_img(par=3)
        self.vid_source = '0'
        self.webcam = True


    def vid_detect(self, vid_source = ''):
        # pass
        model = self.model2
        output_size = self.output_size
        # source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = [640, 640]  # inference size (pixels)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        # device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        source = str(self.vid_source)
        webcam = self.webcam
        device = select_device(self.device)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # im.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + (
                #     '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} ')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            # if save_crop:
                            #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                            #                  BGR=True)
                # Print time (inference-only)
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                # Stream results
                # Save results (image with detections)
                im0 = annotator.result()
                frame = im0
                resize_scale = output_size / frame.shape[0]
                frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                cv2.imwrite("images/temp/single_result_vid.jpg", frame_resized)
                if vid_source == 'video':
                    self.vid.setPixmap(QtGui.QPixmap("images/temp/single_result_vid.jpg"))
                    self.vid.setAlignment(Qt.AlignCenter)
                elif vid_source == 'camera':
                    self.cam.setPixmap(QtGui.QPixmap("images/temp/single_result_vid.jpg"))
                    self.cam.setAlignment(Qt.AlignCenter)
                # self.vid_img
                # if view_img:
                # cv2.imshow(str(p), im0)
                # self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                # cv2.waitKey(1)  # 1 millisecond
            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                if vid_source == 'video':
                    self.ui_img(par=2)
                elif vid_source == 'camera':
                    self.ui_img(par=3)
                break

        '''
        ### 界面关闭事件 ### 
        '''

        def closeEvent(self, event):
            reply = QMessageBox.question(self,
                                         'quit',
                                         "Are you sure?",
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.close()
                event.accept()
            else:
                event.ignore()


if __name__ == "__main__":
    #固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    #初始化
    myWin = MyMainForm()
    #将窗口控件显示在屏幕上
    myWin.show()
    #程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
