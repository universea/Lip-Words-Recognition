import os
import numpy as np
import pandas as pd
import cv2
import sys
import glob
import pickle
import codecs
from multiprocessing import Pool

def create_dictionary(source_dir):
    dictionary_txt = codecs.open(source_dir+'dictionary.txt', 'w') # label txt
    with open(source_dir+'lip_train.txt', 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = [line.strip().split('\t') for line in lines]
            dictionary = sorted(np.unique([line[1] for line in lines])) 
            #print(dictionary)
            print(len(dictionary))
            for class_id,class_name in enumerate(dictionary):
                dictionary_txt.write("{0},{1}\n".format(class_id,class_name))
    print("make dictionary succeed")

def create_train_and_val_pkl(source_dir):

    label_dict = {}

    with open(source_dir+'dictionary.txt', 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = [line.strip().split(',') for line in lines]
            for i in range(len(lines)):
                label_dict[lines[i][1]] = int(lines[i][0])
            #print(label_dict)



    source_train_dir = source_dir+'lip_train/'
    target_train_dir = source_dir+'train/'
    target_val_dir = source_dir+'val/'

    if not os.path.exists(target_train_dir):
        os.mkdir(target_train_dir)

    if not os.path.exists(target_val_dir):
        os.mkdir(target_val_dir)

    with open(source_dir+'lip_train.txt', 'r', encoding='utf8') as f1:
            lines = f1.readlines()
            lines = [line.strip().split('\t') for line in lines]
            for j,line in enumerate(lines):
                
                #print(line,label_dict[line[1]])
                image_file = os.listdir(os.path.join(source_train_dir, line[0]))
                image_file.sort()
                image_name = line[0]
                label_id = label_dict[line[1]]
                image_num = len(image_file)
                #print(image_file)
                #print('image_name=',line[0])
                #print('image_num=',image_num)
                frame = []
                vid = image_name
                for i in range(image_num):
                    image_path = os.path.join(os.path.join(source_train_dir, line[0]), image_file[i])
                    #print('path=',image_path)
                    frame.append(image_path)
                output_pkl = vid + '.pkl'
                if j % 20 == 0:
                    output_pkl = vid + '.pkl'
                    output_pkl = os.path.join(target_val_dir, output_pkl) 
                    f = open(output_pkl, 'wb')
                    pickle.dump((vid, label_id, frame), f, -1)
                    f.close()
                    output_pkl = vid + '.pkl'
                    output_pkl = os.path.join(target_train_dir, output_pkl)
                    f = open(output_pkl, 'wb')
                    pickle.dump((vid, label_id, frame), f, -1)
                    f.close()
                else:
                    output_pkl = os.path.join(target_train_dir, output_pkl)
                    f = open(output_pkl, 'wb')
                    pickle.dump((vid, label_id, frame), f, -1)
                    f.close()

    print("make train and val dataset succeed")

def create_test_pkl(source_dir):

    label_dict = {}

    with open(source_dir+'dictionary.txt', 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = [line.strip().split(',') for line in lines]
            for i in range(len(lines)):
                label_dict[lines[i][1]] = int(lines[i][0])
            #print(label_dict)



    source_test_dir = source_dir+'lip_test/'
    target_test_dir = source_dir+'test/'

    if not os.path.exists(target_test_dir):
        os.mkdir(target_test_dir)

    test_dirs = os.listdir(source_test_dir)
    #print(test_dirs)

    for each_test_dir in test_dirs:
        image_file = os.listdir(os.path.join(source_test_dir, each_test_dir))
        image_file.sort()
        image_name = each_test_dir
        image_num = len(image_file)
        label_id = 0

        frame = []
        vid = image_name
        for i in range(image_num):
            image_path = os.path.join(os.path.join(source_test_dir, image_name), image_file[i])
            #print('path=',image_path)
            frame.append(image_path)
        output_pkl = vid + '.pkl'

        output_pkl = os.path.join(target_test_dir, output_pkl)

        f = open(output_pkl, 'wb')
        pickle.dump((vid, label_id, frame), f, -1)
        f.close()
    print("make test dataset succeed")

def create_txt(source_dir):

    train_data = os.listdir(source_dir + 'train')
    train_data = [x for x in train_data if not x.startswith('.')]
    print(len(train_data))

    test_data = os.listdir(source_dir + 'test')
    test_data = [x for x in test_data if not x.startswith('.')]
    print(len(test_data))

    val_data = os.listdir(source_dir + 'val')
    val_data = [x for x in val_data if not x.startswith('.')]
    print(len(val_data))

    f = open(source_dir+'train.list', 'w')
    for line in train_data:
        f.write(source_dir + 'train/' + line + '\n')
    f.close()


    f = open(source_dir+'test.list', 'w')
    for line in test_data:
        f.write(source_dir + 'test/' + line + '\n')
    f.close()


    f = open(source_dir+'val.list', 'w')
    for line in val_data:
        f.write(source_dir + 'val/' + line + '\n')
    f.close()

    print("make all txt succeed")

if __name__ == '__main__':
    
    data_dir = '/home/ubuntu/disk2/lip_data/data/'
    create_dictionary(data_dir)
    create_train_and_val_pkl(data_dir)
    create_test_pkl(data_dir)
    create_txt(data_dir)



    

   