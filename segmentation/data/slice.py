#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:26:56 2021

@author: mibook
"""
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon, box
from rasterio.features import rasterize
from shapely.ops import cascaded_union
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os, shutil
import pdb

def read_label(filename):
    img = plt.imread(filename)#mpimg.imread(filename)
    if len(img.shape) == 3: img = np.mean(img, axis=2)
    img = (img==1).astype(int)#
    img = np.expand_dims(img, axis=2)
    #if len(img.shape)==3: img = img[:,:,0]
    return img

def read_img(filename):
    img = plt.imread(filename)#mpimg.imread(filename)#
    #x = np.mean(img, axis=2)
    x = img
    return x

def verify_slice_size(slice, conf):
    if slice.shape[0] != conf["window_size"][0] or slice.shape[1] != conf["window_size"][1]:
        if len(slice.shape)==3: 
            temp = np.zeros((conf["window_size"][0], conf["window_size"][1], slice.shape[2]))
            temp[0:slice.shape[0], 0:slice.shape[1],:] = slice
        elif len(slice.shape)==2:
            temp = np.zeros((conf["window_size"][0], conf["window_size"][1]))
            temp[0:slice.shape[0], 0:slice.shape[1]] = slice
        slice = temp
    return slice

def save_slice(arr, filename):
        np.save(filename, arr)

def folder_setup(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

'''def save_slices(filename, img, label, split, bias, **conf):
    def filter_percentage(slice, percentage, freq):
        labelled_pixels = np.sum(slice)
        total_pixels = slice.shape[0] * slice.shape[1]
        if labelled_pixels/total_pixels < percentage:
            if np.random.rand() <= freq:
                return False
        return True

    if not os.path.exists(conf["out_dir"]+split):
        os.makedirs(conf["out_dir"]+split)

    slicenum = 0
    for row in range(0, img.shape[0], conf["window_size"][0]-conf["overlap"]):
        for column in range(0, img.shape[1], conf["window_size"][1]-conf["overlap"]):
            label_slice = label[row:row+conf["window_size"][0], column:column+conf["window_size"][1]]
            label_slice = verify_slice_size(label_slice, conf)
            if bias is None:
                bias_slice = label_slice
            else:
                bias_slice = bias[row:row+conf["window_size"][0], column:column+conf["window_size"][1]]
                bias_slice = verify_slice_size(bias_slice, conf)
            if filter_percentage(bias_slice, 1-conf["filter"], 0.97):
                img_slice = img[row:row+conf["window_size"][0], column:column+conf["window_size"][1], :]
                img_slice = verify_slice_size(img_slice, conf)
                save_slice(label_slice, conf["out_dir"]+split+"/label_"+str(filename)+"_slice_"+str(slicenum))
                save_slice(img_slice, conf["out_dir"]+split+"/img_"+str(filename)+"_slice_"+str(slicenum))
                print(f"Saved image {filename} slice {slicenum}")
            if filter_percentage(label_slice, conf["filter"], 0.999):
                img_slice = img[row:row+conf["window_size"][0], column:column+conf["window_size"][1], :]
                img_slice = verify_slice_size(img_slice, conf)
                save_slice(label_slice, conf["out_dir"]+split+"/label_"+str(filename)+"_slice_"+str(slicenum))
                save_slice(img_slice, conf["out_dir"]+split+"/img_"+str(filename)+"_slice_"+str(slicenum))
                print(f"Saved image {filename} slice {slicenum}")
            slicenum += 1
'''
def save_test_slices(filename, img, label, split, **conf):
    
    if not os.path.exists(conf["out_dir"]+split):
        os.makedirs(conf["out_dir"]+split)

    slicenum = 0
    for row in range(0, img.shape[0], conf["window_size"][0]):
        for column in range(0, img.shape[1], conf["window_size"][1]):
            label_slice = label[row:row+conf["window_size"][0], column:column+conf["window_size"][1]]
            label_slice = verify_slice_size(label_slice, conf)
            img_slice = img[row:row+conf["window_size"][0], column:column+conf["window_size"][1], :]
            img_slice = verify_slice_size(img_slice, conf)
            save_slice(label_slice, conf["out_dir"]+split+"/label_"+str(filename)+"_slice_"+str(slicenum)+"_"+str(row)+"_"+str(column))
            save_slice(img_slice, conf["out_dir"]+split+"/img_"+str(filename)+"_slice_"+str(slicenum)+"_"+str(row)+"_"+str(column))
            print(f"Saved image {filename} slice {slicenum}")
            slicenum += 1

def remove_and_create(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)

'''def save_slices(filename, img, label, split, bias, **conf):
    def verify_slice_size(slice, conf):
        if slice.shape[0] != conf["window_size"][0] or slice.shape[1] != conf["window_size"][1]:
            temp = np.zeros((conf["window_size"][0], conf["window_size"][1], slice.shape[2]))
            temp[0:slice.shape[0], 0:slice.shape[1],:] = slice
            slice = temp
        return slice

    def filter_percentage(slice, percentage, freq):
        labelled_pixels = np.sum(slice)
        total_pixels = slice.shape[0] * slice.shape[1]
        if labelled_pixels/total_pixels < percentage:
            if np.random.rand() <= freq:
                return False
        return True

    def save_slice(arr, filename):
        np.save(filename, arr)

    if not os.path.exists(conf["out_dir"]+split):
        os.makedirs(conf["out_dir"]+split)

    slicenum = 0
    for row in range(0, img.shape[0], conf["window_size"][0]-conf["overlap"]):
        for column in range(0, img.shape[1], conf["window_size"][1]-conf["overlap"]):
            label_slice = label[row:row+conf["window_size"][0], column:column+conf["window_size"][1]]
            label_slice = verify_slice_size(label_slice, conf)
            if bias is None:
                bias_slice = label_slice
            else:
                bias_slice = bias[row:row+conf["window_size"][0], column:column+conf["window_size"][1]]
                bias_slice = verify_slice_size(bias_slice, conf)
            if filter_percentage(bias_slice, 1-conf["filter"], 0.97):
                img_slice = img[row:row+conf["window_size"][0], column:column+conf["window_size"][1], :]
                img_slice = verify_slice_size(img_slice, conf)
                save_slice(label_slice, conf["out_dir"]+split+"/label_"+str(filename)+"_slice_"+str(slicenum))
                save_slice(img_slice, conf["out_dir"]+split+"/img_"+str(filename)+"_slice_"+str(slicenum))
                print(f"Saved image {filename} slice {slicenum}")
            if filter_percentage(label_slice, conf["filter"], 0.999):
                img_slice = img[row:row+conf["window_size"][0], column:column+conf["window_size"][1], :]
                img_slice = verify_slice_size(img_slice, conf)
                save_slice(label_slice, conf["out_dir"]+split+"/label_"+str(filename)+"_slice_"+str(slicenum))
                save_slice(img_slice, conf["out_dir"]+split+"/img_"+str(filename)+"_slice_"+str(slicenum))
                print(f"Saved image {filename} slice {slicenum}")
            slicenum += 1
'''
def save_slices_v2(filename, img, label, split, **conf):
    def getGaussianSamplingPoints(gt, windowSize, n):
        center = np.mean(np.argwhere(gt==1), axis=0)[0:2] - (windowSize//2) #upper left corner of sample
        lBound = np.min(np.argwhere(gt==1), axis=0)[0:2] #in row,col or y,x
        uBound = np.max(np.argwhere(gt==1), axis=0)[0:2]
        print(center, lBound, uBound)
        height, width = (uBound-lBound)/1.5
        cov = [[height**2, 0], [0, width**2]]
        X,Y=[],[]
        while True:
            y,x = np.random.multivariate_normal(center, cov, 1).astype(int).T
            valid = ((y[0]<(gt.shape[0]-windowSize)) and (x[0]<(gt.shape[1]-windowSize)) 
                    and (y[0]>0) and (x[0]>0))
            if not valid: continue
            print(x,y)
            X.append(x[0])
            Y.append(y[0])
            if len(X) == n: break
        return np.array(X), np.array(Y)

    folder_setup(conf["out_dir"]+split)

    columns,rows = getGaussianSamplingPoints(label,conf["window_size"][0],conf["n_samples"])
    
    for i in range(len(columns)):
        row, column = rows[i], columns[i]
        slicenum = i
        label_slice = label[row:row+conf["window_size"][0], column:column+conf["window_size"][1]]
        label_slice = verify_slice_size(label_slice, conf)
        img_slice = img[row:row+conf["window_size"][0], column:column+conf["window_size"][1], :]
        img_slice = verify_slice_size(img_slice, conf)
        save_slice(label_slice, conf["out_dir"]+split+"/label_"+str(filename)+"_slice_"+str(slicenum))
        save_slice(img_slice, conf["out_dir"]+split+"/img_"+str(filename)+"_slice_"+str(slicenum))
        print(f"Saved image {filename} slice {slicenum}: {np.max(label_slice)}")
        

def train_test_shuffle(out_dir, train_split, val_split, test_split):
    train_path = out_dir + "train/"
    remove_and_create(train_path)
    val_path = out_dir + "val/"
    remove_and_create(val_path)
    test_path = out_dir + "test/"
    remove_and_create(test_path)

    slices = [x for x in os.listdir(out_dir) if (x.endswith('.npy') and "tiff" in x )]
    n_tiffs = len(slices)
    random_index = np.random.permutation(n_tiffs)
    savepath = train_path
    for count, index in enumerate(random_index):
        if count > int(n_tiffs*train_split):
            savepath = val_path
        if count > int(n_tiffs*(train_split+val_split)):
            savepath = test_path
        tiff_filename = slices[index]
        mask_filename = tiff_filename.replace("tiff","mask")
        shutil.move(out_dir+tiff_filename, savepath+tiff_filename)
        shutil.move(out_dir+mask_filename, savepath+mask_filename)

def addPositionChannel(im):
    r = np.repeat(range(0,im.shape[0]), im.shape[1])
    c = np.tile(range(0,im.shape[1]), im.shape[0])
    m = r*c
    position = np.expand_dims(np.reshape(m,im.shape[0:2])/np.max(m), axis=-1)
    return np.dstack((im, position))