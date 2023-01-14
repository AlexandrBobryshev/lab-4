from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def printFunc(filename, outputFilename, booleanFinal):


    f = open(filename, 'r')
    x = []
    y = []

    while True:

        line = f.readline() 
        if not line:
            break

        x.append(float(line.split()[0]))
        y.append(float(line.split()[1]))

    plt.rcParams['toolbar'] = 'None'
    plt.figure(figsize=(960/plt.rcParams['figure.dpi'], 540/plt.rcParams['figure.dpi']))

    if booleanFinal == False:
        plt.scatter(x, y)

    else:
        plt.scatter(x, y, c = "black", alpha = 0.1)

    plt.axis('off')
    plt.savefig(outputFilename)
    f.close()
    plt.clf()

printFunc("DS3.txt", "image.jpg", False)
printFunc("DS3.txt", "resultImage.jpg", True)

resultImage = cv2.imread("resultImage.jpg")
img = cv2.imread("image.jpg")
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray_image,129,255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
height, width, _ = img.shape
os.remove("image.jpg")

points = []

for c in contours:

    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(resultImage, (cX, cY), 2.5, (255, 0, 0), -1)
    points.append([cX, height - cY])

cv2.imwrite("resultImage.jpg", resultImage)

X = np.array(points)
vor = Voronoi(points)
fig = voronoi_plot_2d(vor)
plt.savefig("Voronoi_Diagram.jpg")

