import torch
import torch.nn as nn
import os
import numpy as np
import random


def retrievalParamSP():
    def get_label(root):
        with open(root,'r') as f:
            shape_data = f.readlines()
        def label_split(x):
            label = []
            label.append(x.split(',')[1][:-1])
            return label
        p = list(map(label_split,shape_data))
        q = np.array([int(v[0]) for v in p ])

        return q

    def retrievalParamSPs(shape_label,sketch_test_label):
        shapeLabels = np.array(shape_label)  ### cast all the labels as array
        sketchTestLabel = np.array(sketch_test_label)  ### cast sketch test label as array
        C_depths = np.zeros(sketchTestLabel.shape)
        unique_labels = np.unique(sketchTestLabel)
        for i in range(unique_labels.shape[0]):  ### find the numbers
            tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0]  ## for sketch index
            tmp_index_shape = np.where(shapeLabels == unique_labels[i])[0]  ## for shape index
            C_depths[tmp_index_sketch] = tmp_index_shape.shape[0]
        return C_depths

    shape_label = get_label('/test/data2/xurui/3d_shape/sketchShapeDataAndResult-master/dataset/shrec13sketchShape/shape_list.txt')
    shape_data = [shape_label[i:i + 12] for i in range(0, len(shape_label), 12)]
    def shape_label_new(x):
        return x[0]
    shape_label_view = np.array(list(map(shape_label_new,shape_data)))
    sketch_test_label = get_label('/test/data2/xurui/3d_shape/sketchShapeDataAndResult-master/dataset/shrec13sketchShape/test_sketch_list.txt')
    C_depths = retrievalParamSPs(list(shape_label_view),list(sketch_test_label))

    return shape_label_view,sketch_test_label,C_depths

def retrievalParamSP_v2(shape_label,sketch_test_label):

    def retrievalParamSPs(shape_label,sketch_test_label):
        shapeLabels = np.array(shape_label)  ### cast all the labels as array
        sketchTestLabel = np.array(sketch_test_label)  ### cast sketch test label as array
        C_depths = np.zeros(sketchTestLabel.shape)
        unique_labels = np.unique(sketchTestLabel)
        for i in range(unique_labels.shape[0]):  ### find the numbers
            tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0]  ## for sketch index
            tmp_index_shape = np.where(shapeLabels == unique_labels[i])[0]  ## for shape index
            C_depths[tmp_index_sketch] = tmp_index_shape.shape[0]
        return C_depths

    C_depths = retrievalParamSPs(shape_label,sketch_test_label)

    return C_depths

