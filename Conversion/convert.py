import cv2
import sys

if __name__ == '__main__':

    path = sys.argv[1]

    new_path = ".".join(path.split(".")[:-1]) + ".png"

    cv2.imwrite(new_path, cv2.imread(path, -1))

