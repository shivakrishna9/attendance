import numpy as np
import cv2
import time
import glob
import re
import random
from collections import Counter


def get_images():
    lst = []
    lw = []
    for image in glob.glob("../extras/newtest/test_faces/*.jpg"):
        image = re.sub('\.\./', '', image)
        person = '_'.join(image.split('/')[3].split('\.')[0].split('_')[:-1])
        lst.append((image, person))
    
    up = [x for (a, x) in lst]
    x = Counter(up)
    

    with open('../traintest/incorrect.txt', 'w') as f:
        for i in lst:
            print i[0] +'\t'+ i[1]
            f.write(i[0]+'\t'+ i[1]+'\n')

    print len(up)
    print x
    

def image_ext():
    i = []
    for image in glob.glob("../extras/newtest/myclass/*/*.jpg"):
        person = image.split('/')[4]
        image = re.sub('\.\./', '', image)
        img = image
        if 'not' in person:
            i.append((person, 'none', img))
        else:
            i.append(('face', person, img))

    up = [x for (a, x, y) in i if x != 'none']
    upx = up
    x = Counter(up)
    up = list(set(up))
    up = sorted(up)

    m = []
    for image in glob.glob("../newtest/*/*.jpg"):
        person = image.split('/')[2].lower()
        image = re.sub('\.\./', '', image)
        img = image
        if 'not' in person:
            m.append((person, 'none', img))
        else:
            m.append(('face', person, img))

    random.shuffle(m)
    random.shuffle(i)

    with open("../traintest/class66_train.txt", 'w') as f:
        for k in i:
            if k[1] != 'none':
                # pass
                print k[1] + '\t' + str(up.index(k[1])) + '\t' + k[2]
                f.write(k[1] + '\t' + str(up.index(k[1])) + '\t' + k[2] + '\n')
            # else:
            #     print k[0]+'\t'+k[1]+'\t'+'none'+'\t'+k[2]
            #     f.write(k[0]+'\t'+k[1]+'\t'+'none'+'\t'+k[2]+'\n')

    print up
    print len(upx)
    print x


def pre_process():
    i = []
    for image in glob.glob("extras/newtest/myclass/*/*.jpg"):
        person = image.split('/')[3]
        image = re.sub('\.\./', '', image)
        img = image
        if 'not' in person:
            i.append((person, 'none', img))
        else:
            i.append(('face', person, img))

    up = [x for (a, x, y) in i if x != 'none']
    upx = up
    x = Counter(up)
    up = list(set(up))
    up = sorted(up)

    # m= []
    # for image in glob.glob("../newtest/*/*.jpg"):
    #     person = image.split('/')[2].lower()
    #     image = re.sub('\.\./', '', image)
    #     img = image
    #     if 'not' in person:
    #         m.append((person,'none',img))
    #     else:
    #         m.append(('face',person,img))

    # random.shuffle(m)
    # random.shuffle(i)

    # with open("../traintest/class20_test.txt", 'w') as f:
    #     for k in m:
    #         if k[1]!='none':
    #             # pass
    #             print k[1]+'\t'+str(up.index(k[1]))+'\t'+k[2]
    #             f.write(k[1]+'\t'+str(up.index(k[1]))+'\t'+k[2]+'\n')
    #         # else:
    #         #     print k[0]+'\t'+k[1]+'\t'+'none'+'\t'+k[2]
    #         #     f.write(k[0]+'\t'+k[1]+'\t'+'none'+'\t'+k[2]+'\n')

    # print up
    # print len(upx)
    # print x
    return up


def encode():
    lst = []
    l1 = []
    with open("../traintest/classtrain.txt", 'r') as f:
        for i in f:
            if i.split(',')[1].split('\n')[0] not in l1:
                l1.append(i.split(',')[1].split('\n')[0])
            lst.append([i.split(',')[1].split('\n')[0], i.split(',')[0]])

    print l1

    with open("train.txt", 'w') as f:
        for i in lst:
            print str(i[1]) + "," + str(l1.index(i[0]))
            f.write(i[1] + "," + str(l1.index(i[0])) + '\n')


def image_load():
    # Load an color image in grayscale

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # print img.shape
    for image in glob.glob("../extras/faceScrub/download/*/*.jpg"):
        start = time.time()
        img = cv2.imread(image)
        # res = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # construct a list of bounding boxes from the detection
        # rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]

        # update the data dictionary with the faces detected
        # data.update({"num_faces": len(rects), "faces": rects, "success": True})

        print "time", time.time() - start
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_color = img[y:y + h, x:x + w]
            cv2.imshow('image', img)
            if cv2.waitKey(0) & 0xFF == ord('y'):
                cv2.destroyAllWindows()
                with open("../traintest/detrain.txt", 'a') as f:
                    print image.split('/')[5] + "," + image.split('/')[4] + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h)
                    f.write(image.split('/')[5] + "," + image.split('/')[4] + ',' + str(
                        x) + ',' + str(y) + ',' + str(w) + ',' + str(h) + '\n')

            elif cv2.waitKey(0) & 0xFF == ord('n'):
                cv2.destroyAllWindows()


if __name__ == '__main__':
    # image_load()
    get_images()
    # pre_process()
    # image_ext()
