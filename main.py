from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
import argparse
import cv2
import torch
from torch.autograd import Variable

from pprint import pprint
from operator import itemgetter

import itertools
from itertools import compress
from random import randrange

def Convertir_RGB(img):
    # Convertir Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img

def Convertir_BGR(img):
    # Convertir red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img

def calculate_centr(coord):
  return (coord[0]+(coord[2]/2), coord[1]+(coord[3]/2))

def calculate_centr_distances(centroid_1, centroid_2):
  return  math.sqrt((centroid_2[0]-centroid_1[0])**2 + (centroid_2[1]-centroid_1[1])**2)

# Pixel per meters
width = 1080
average_px_meter = (width-150) / 0.07

model_def = 'config/yolov3-custom.cfg'
weights_path = 'checkpoints/yolov3_ckpt_99.pth'
class_path = 'data/custom/classes.names'
conf_thres = 0.9
batch_size = 2
# pathVideo = '/home/alan/Videos/dataset_completo_parte1.mp4'
pathVideo = '/home/alan/Documents/data-files/classe1_(fio-solto).mp4'
checkpoint_model = 'checkpoints/yolov3_ckpt_99.pth'
n_cpu = 8

frame_width = 0
frame_height = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default=model_def, help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default=weights_path, help="path to weights file")
    parser.add_argument("--class_path", type=str, default=class_path, help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=conf_thres, help="object confidence threshold")
    parser.add_argument("--webcam", type=int, default=0, help="Is the video processed video? 1 = Yes, 0 == no")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=batch_size, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=n_cpu, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--directorio_video", type=str, default=pathVideo, help="Directorio al video")
    parser.add_argument("--checkpoint_model", type=str, default=checkpoint_model, help="path to checkpoint model")

    opt = parser.parse_args()

    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if opt.webcam == 1:
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    else:
        cap = cv2.VideoCapture(opt.directorio_video)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # out = cv2.VideoWriter('outp.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a = []
    while cap:
        ret, frame = cap.read()
        if ret is False:
            break
        #frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)
        # A imagem vem em Blue, Green, Red, logo nós convertemos para RGB que é a entrada que o modelo chama
        RGBimg = Convertir_RGB(frame)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))

        with torch.no_grad():
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        listItens = []
        listCentros = []
        listCoordenadas = []

        for detection in detections:
            if detection is not None:
                detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = [int(c) for c in colors[int(cls_pred)]]

                    # frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 3)


                    cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)  # Nome da clase detectada

                    cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 3)  # Certeza de precisão da classe


                    dict1 = {'Location': int(x1), 'Class': classes[int(cls_pred)]}
                    listItens.append(dict1)
                    # print(dict1)
                # print(listItens)
                sorted_list = sorted(listItens, key=itemgetter('Location'))
                print('Transformadores:', sorted_list)
                num = 0
                for d in list(sorted_list):
                    d['Id'] = num
                    # print('d:', d)
                    num+=1
                    if d['Class']=='defeito':
                        sorted_list.pop()
                        print('retirar:',int(d['Id']))
                print('result',sorted_list)

                # pprint(list(enumerate(sorted_list)))
                # pprint(sorted_list)
                print('----')

        # Convertimos de vuelta a BGR para que cv2 pueda desplegarlo en los Convertir_RGBcolores correctos
        if opt.webcam == 1:
            cv2.imshow('frame', Convertir_BGR(RGBimg))
            # out.write(RGBimg)
        else:
            # out.write(Convertir_BGR(RGBimg))
            cv2.imshow('frame', Convertir_BGR(RGBimg))
        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # out.release()
    cap.release()
    cv2.destroyAllWindows()