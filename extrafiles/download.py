#!/usr/bin/env python2.7
# coding: utf-8
import os
from os.path import join, exists
import multiprocessing
import hashlib
import cv2


files = ['../traintest/actors.txt', '../traintest/actresses.txt']
RESULT_ROOT = '../extras/download'
if not exists(RESULT_ROOT):
    os.mkdir(RESULT_ROOT)


# def remove(names, urls, fnames):

#     for image in glob(RESULT_ROOT+'*/*.jpg'):
        


def download((names, urls, fnames)):
    """
        download from urls into folder names using wget
    """

    assert(len(names) == len(urls))
    assert(len(names) == len(fnames))

    # download using external wget
    CMD = 'wget -c -t 1 -T 3 "%s" -O "%s"'
    for i in range(len(names)):
        directory = join(RESULT_ROOT, names[i])
        if not exists(directory):
            os.mkdir(directory)
        fname = fnames[i] + '.jpg'
        dst = join(directory, fname)
        print "downloading", fname
        if exists(dst):
            print "already downloaded, skipping..."
            continue
        else:
            res = os.system(CMD % (urls[i], dst))
        # # get face
        # face_directory = join(directory, 'face')
        # if not exists(face_directory):
        #     os.mkdir(face_directory)
        # img = cv2.imread(dst)
        # if img is None:
        #     # no image data
        #     os.remove(dst)
        # else:
        #     face_path = join(face_directory, fname)
        #     face = img[bboxes[i][1]:bboxes[i][3], bboxes[i][0]:bboxes[i][2]]
        #     cv2.imwrite(face_path, face)
        #     # write bbox to file
        #     with open(join(directory,'_bboxes.txt'), 'a') as fd:
        #         bbox_str = ','.join([str(_) for _ in bboxes[i]])
        #         fd.write('%s %s\n' % (fname, bbox_str))


if __name__ == '__main__':
    for f in files:
        with open(f, 'r') as fd:
            # strip first line
            fd.readline()
            names = []
            fnames = []
            urls = []
            for line in fd.readlines():
                components = line.split('\t')
                # print components
                # assert(len(components) == )
                name = '_'.join(components[0].split(' '))
                # print name
                url = components[3]
                fname = name+'_'+components[1]
                fnames.append(fname)
                names.append(name)
                urls.append(url)
        # every name gets a task

        # names, urls, fnames = remove(names, urls, fnames)

        last_name = names[0]
        task_names = []
        task_urls = []
        task_fnames = []
        tasks = []
        for i in range(len(names)):
            if names[i] == last_name:
                task_names.append(names[i])
                task_fnames.append(fnames[i])
                task_urls.append(urls[i])
            else:
                tasks.append((task_names, task_urls, task_fnames))
                task_names = [names[i]]
                task_urls = [urls[i]]
                task_fnames = [fnames[i]]
                last_name = names[i]
        tasks.append((task_names, task_urls, task_fnames))

        pool_size = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=pool_size, maxtasksperchild=2)
        pool.map(download, tasks)
        pool.close()
        pool.join()