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

def calculate_perm(centroids):
  permutations = []
  for current_permutation in itertools.permutations(centroids, 2):
    if current_permutation[::-1] not in permutations:
      permutations.append(current_permutation)
  return permutations

def midpoint(p1, p2):
    return ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)

def calculate_slope(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    return m

# Pixel per meters
width = 1080
average_px_meter = (width-150) / 0.07

model_def = 'config/yolov3-custom.cfg'
weights_path = 'checkpoints/yolov3_ckpt_99.pth'
class_path = 'data/custom/classes.names'
conf_thres = 0.9
batch_size = 2
pathVideo = '/home/alan/Videos/dataset_completo_parte1.mp4'
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

                    coord = [int(x1), int(y1), int(x2), int(y2)]
                    listCoordenadas.append(coord)

                    centros = calculate_centr(coord)
                    # print("centros:",centros)
                    listCentros.append(centros)

                    permutations = calculate_perm(centros)
                    # print("permutations:", permutations)

                    # print(
                    #     "Resultado: {} - posição X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1,
                    #                                                                     x2, y2))

                    frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 3)

                    cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                                3)  # Nome da clase detectada

                    cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 3)  # Certeza de precisão da classe


                    dict1 = {'Location': int(x1), 'Class': classes[int(cls_pred)]}
                    listItens.append(dict1)
                    # print(dict1)
                # print(listItens)
                sorted_list = sorted(listItens, key=itemgetter('Location'))
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

                # for id in list(enumerate(sorted_list)):
                #     # print('out:',list(id))
                #     if dict1['Class'] == 'defeito':
                #         print('achou:',list(id))
                print('----')

                # print('listaCentros',listCentros)
                # dist01 = calculate_centr_distances(listCentros[0],listCentros[1])
                # dist12 = calculate_centr_distances(listCentros[1],listCentros[2])
                #
                # print("dist1:", dist01)
                # print("dist2:", dist12)

                permutations = calculate_perm(listCentros)
                print("permutations:", permutations)
                print('----')

                # # Display boxes and centroids
                # fig, ax = plt.subplots(figsize=(20, 12), dpi=90)
                # ax.imshow(frame, interpolation='nearest')
                # for coord, centr in zip(listCoordenadas, listCentros):
                #     ax.add_patch(patches.Rectangle((coord[0], coord[1]), coord[2], coord[3], linewidth=2, edgecolor='y',
                #                                    facecolor='none', zorder=10))
                #     ax.add_patch(patches.Circle((centr[0], centr[1]), 3, color='yellow', zorder=20))
                #
                # # Display lines between centroids
                # for perm in permutations:
                #     dist = calculate_centr_distances(perm[0], perm[1])
                #     dist_m = dist / average_px_meter
                #
                #     print("M meters: ", dist_m)
                #     middle = midpoint(perm[0], perm[1])
                #     print("Middle point", middle)
                #
                #     x1 = perm[0][0]
                #     x2 = perm[1][0]
                #     y1 = perm[0][1]
                #     y2 = perm[1][1]
                #
                #     slope = calculate_slope(x1, y1, x2, y2)
                #     dy = math.sqrt(3 ** 2 / (slope ** 2 + 1))
                #     dx = -slope * dy
                #
                #     # Display randomly the position of our distance text
                #     if randrange(10) % 2 == 0:
                #         Dx = middle[0] - dx * 10
                #         Dy = middle[1] - dy * 10
                #     else:
                #         Dx = middle[0] + dx * 10
                #         Dy = middle[1] + dy * 10
                #
                #     if dist_m < 1.5:
                #         ax.annotate("{}m".format(round(dist_m, 2)), xy=middle, color='white', xytext=(Dx, Dy),
                #                     fontsize=10, arrowprops=dict(arrowstyle='->', lw=1.5, color='yellow'),
                #                     bbox=dict(facecolor='red', edgecolor='white', boxstyle='round', pad=0.2), zorder=30)
                #
                #         ax.plot((perm[0][0], perm[1][0]), (perm[0][1], perm[1][1]), linewidth=2, color='yellow',
                #                 zorder=15)
                #     elif 1.5 < dist_m < 3.5:
                #         ax.annotate("{}m".format(round(dist_m, 2)), xy=middle, color='black', xytext=(Dx, Dy),
                #                     fontsize=8, arrowprops=dict(arrowstyle='->', lw=1.5, color='skyblue'),
                #                     bbox=dict(facecolor='y', edgecolor='white', boxstyle='round', pad=0.2), zorder=30)
                #         ax.plot((perm[0][0], perm[1][0]), (perm[0][1], perm[1][1]), linewidth=2, color='skyblue',
                #                 zorder=15)
                #     else:
                #         pass

                # Display boxes and centroids
                fig, ax = plt.subplots(figsize=(20, 12), dpi=90, frameon=False)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis('off')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                for coord, centr in zip(listCoordenadas, listCentros):
                    ax.add_patch(patches.Rectangle((coord[0], coord[1]), coord[2], coord[3], linewidth=2, edgecolor='y',
                                                   facecolor='none', zorder=10))
                    ax.add_patch(patches.Circle((centr[0], centr[1]), 3, color='yellow', zorder=20))

                # Display lines between centroids
                for perm in permutations:
                    dist = calculate_centr_distances(perm[0], perm[1])
                    dist_m = dist / average_px_meter

                    x1 = perm[0][0]
                    y1 = perm[0][1]
                    x2 = perm[1][0]
                    y2 = perm[1][1]

                    # Calculate middle point
                    middle = midpoint(perm[0], perm[1])

                    # Calculate slope
                    slope = calculate_slope(x1, y1, x2, y2)
                    dy = math.sqrt(3 ** 2 / (slope ** 2 + 1))  # Display boxes and centroids
                fig, ax = plt.subplots(figsize=(20, 12), dpi=90, frameon=False)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis('off')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                for coord, centr in zip(listCoordenadas, listCentros):
                    ax.add_patch(patches.Rectangle((coord[0], coord[1]), coord[2], coord[3], linewidth=2, edgecolor='y',
                                                   facecolor='none', zorder=10))
                    ax.add_patch(patches.Circle((centr[0], centr[1]), 3, color='yellow', zorder=20))

                # Display lines between centroids
                for perm in permutations:
                    dist = calculate_centr_distances(perm[0], perm[0])
                    dx = -slope * dy

                    # Set random location
                    if randrange(10) % 2 == 0:
                        Dx = middle[0] - dx * 10
                        Dy = middle[1] - dy * 10
                    else:
                        Dx = middle[0] + dx * 10
                        Dy = middle[1] + dy * 10

                    if dist_m < 1.5:
                        ax.annotate("{}m".format(round(dist_m, 2)), xy=middle, color='white', xytext=(Dx, Dy),
                                    fontsize=10, arrowprops=dict(arrowstyle='->', lw=1.5, color='yellow'),
                                    bbox=dict(facecolor='red', edgecolor='white', boxstyle='round', pad=0.2), zorder=35)
                        ax.plot((perm[0][0], perm[1][0]), (perm[0][1], perm[1][1]), linewidth=2, color='yellow',
                                zorder=15)
                    elif 1.5 < dist_m < 3.5:
                        ax.annotate("{}m".format(round(dist_m, 2)), xy=middle, color='black', xytext=(Dx, Dy),
                                    fontsize=8, arrowprops=dict(arrowstyle='->', lw=1.5, color='skyblue'),
                                    bbox=dict(facecolor='y', edgecolor='white', boxstyle='round', pad=0.2), zorder=35)
                        ax.plot((perm[0][0], perm[1][0]), (perm[0][1], perm[1][1]), linewidth=2, color='skyblue',
                                zorder=15)
                    else:
                        pass

                    ax.imshow(frame, interpolation='nearest')

                    # Convert figure to numpy
                    fig.canvas.draw()

                    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    img = np.array(fig.canvas.get_renderer()._renderer)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imshow('distance', img)


        #
        # Convertimos de vuelta a BGR para que cv2 pueda desplegarlo en los Convertir_RGBcolores correctos
        # ax.imshow(frame, interpolation='nearest')

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
