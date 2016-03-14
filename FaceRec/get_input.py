import cv2

def input_image():

    # Load an color image in grayscale
    img = cv2.imread('fish-bike.jpg')

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # print img.shape
    res = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)

    # build the VGG16 network with our input_img as input

    return res