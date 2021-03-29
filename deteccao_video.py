from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
import argparse
import cv2
import torch
from torch.autograd import Variable
import json

from pprint import pprint
from operator import itemgetter

import itertools
from itertools import compress
from random import randrange

def converter_rgb(img):
    # Converter Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def converter_bgr(img):
    # Converter red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img

# def calculate_centr(coord):
#   return (coord[0]+(coord[2]/2), coord[1]+(coord[3]/2))
#
# def calculate_centr_distances(centroid_1, centroid_2):
#   return  math.sqrt((centroid_2[0]-centroid_1[0])*2 + (centroid_2[1]-centroid_1[1])*2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_99.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.85, help="object confidence threshold")
    parser.add_argument("--webcam", type=int, default=0, help="Is the video processed video? 1 = Yes, 0 == no")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--directorio_video", type=str, default="/home/cesarhcq/projeto_creathus_ws/deteccao_projeto_springer/videos/fhd_c_ilum", help="Directorio al video")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
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
        # cap.set(10, 180)
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1920, 1080)) #(1280, 960)
    else:
        cap = cv2.VideoCapture(opt.directorio_video)
        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))
        # fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
        # out = cv2.VideoWriter('outp.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1920, 1080))
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a = []
    while cap:
        ret, frame = cap.read()
        if ret is False:
            break
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        # A imagem vem em BGR e é convertida em RGB que é o que o modelo requer
        RGBimg = converter_rgb(frame)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))

        with torch.no_grad():
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        lista_mola = []
        lista_pino = []

        for detection in detections:
            if detection is not None:
                detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = [int(c) for c in colors[int(cls_pred)]]
                    # print("Detectou-se {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2,
                    #                                                                 y2))
                    frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 3)
                    cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                                3)  # Nome da classe detectada
                    cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 3)  # Certeza de predição da classe

                    # classes_dict = {'medidas 1': int(x1), 'classe 1': classes[int(cls_pred)], 'medidas 2': int(y1),
                    #                 'classe 2': classes[int(cls_pred)], 'medidas 3': int(x2), 'classe 3': classes[int(cls_pred)],
                    #                 'medidas 4': int(y2), 'classe 4': classes[int(cls_pred)]}


                    if classes[int(cls_pred)] == 'mola':
                        classes_dict = {'dist': int(x1), 'classe': classes[int(cls_pred)]}
                        lista_mola.append(classes_dict)

                    if classes[int(cls_pred)] == 'pino':
                        classes_dict = {'dist': int(x1), 'classe': classes[int(cls_pred)]}
                        lista_pino.append(classes_dict)

                # lista ordenada para mola
                sorted_lista_mola = sorted(lista_mola, key=itemgetter('dist'))
                json_convert_mola = json.dumps(sorted_lista_mola,indent=4)
                # print(json_convert_mola)

                # lista ordenada para pino

                sorted_lista_pino = sorted(lista_pino, key=itemgetter('dist'))
                json_convert_pino = json.dumps(sorted_lista_pino, indent=4)
                # print(json_convert_pino)

                # print(name_list)
                print('------------')

                if len(sorted_lista_mola) == 2:
                    # print('status mola ---- OK')
                    dict_mola = {'tipo': 'mola','status': 'APROVADO','QTD':len(sorted_lista_mola)}
                    print(json.dumps(dict_mola, indent=4))
                else:
                    dict_mola = {'tipo': 'mola','status': 'NG','QTD':len(sorted_lista_mola)}
                    print(json.dumps(dict_mola, indent=4))

                if len(sorted_lista_pino) == 4:
                    # print('status pino ---- OK')
                    dict_pino = {'tipo': 'pino','status': 'APROVADO','QTD':len(sorted_lista_pino)}
                    print(json.dumps(dict_pino, indent=4))
                else:
                    dict_pino = {'tipo': 'pino','status': 'NG','QTD':len(sorted_lista_pino)}
                    print(json.dumps(dict_pino, indent=4))

        if opt.webcam == 1:
            cv2.imshow('frame', converter_bgr(RGBimg))
            out.write(RGBimg)
        else:
            # out.write(converter_bgr(RGBimg))
            cv2.imshow('Salcomp Processo', RGBimg)
        # cv2.waitKey(0)
        # Pressione Q no teclado para terminar o processo de execução do algoritmo (quit)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # out.release()
    cap.release()
    cv2.destroyAllWindows()