import cv2
import numpy as np

background = cv2.imread("C:\\Users\\Nishant\\Desktop\\background.jpg")
cap = cv2.VideoCapture(0)
background = cv2.resize(background,(640,480))
k = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(640,480))

    Z = frame.reshape((-1,3))
    Z = np.float32(Z)

    _, labels, centers = cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    labels = labels.reshape((480,640))

    unique, counts = np.unique(labels, return_counts=True)
    bg_cluster = unique[np.argmax(counts)]
    mask = labels == bg_cluster
    result = frame.copy()
    result[mask] = background[mask]
    cv2.imshow("Virtual Background", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()