import cv2

cap1 = cv2.VideoCapture("dataset_completo_parte1.mp4")
cap2 = cv2.VideoCapture("dataset_completo_parte2.mp4")

width = 800
height = 480
dim = (width, height)

i = 0
while True:
    success1, frame1 = cap1.read()
    success2, frame2 = cap2.read()

    if success1 and success2:
        # resize image
        resized1 = cv2.resize(frame1, dim, interpolation = cv2.INTER_AREA)
        resized2 = cv2.resize(frame2, dim, interpolation = cv2.INTER_AREA)

        #print('Resized Dimensions : ',resized1.shape)

        if i % 50 == 0:
            print(i)
            cv2.imwrite("frame1/"+str(i)+"_frame.jpg", resized1)
            cv2.imwrite("frame2/"+str(i)+"_frame.jpg", resized2)

        i = i + 1
        print(i)