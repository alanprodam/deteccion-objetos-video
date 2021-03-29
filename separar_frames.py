import cv2

cap1 = cv2.VideoCapture("/home/cesarhcq/Desktop/data_2/videos/fhd_c_ilum")


i = 0
while True:
    success1, frame1 = cap1.read()

    if success1:

        if i % 1 == 0:
            print(i)
            cv2.imwrite("/home/cesarhcq/Desktop/data_2/images/" + str(i) + "_frame.jpg", frame1)
        i = i + 1
        print(i)
