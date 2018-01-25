# import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from collections import OrderedDict
# from cfg import *
# from darknet import MaxPoolStride1
# from region_loss import RegionLoss
# from PIL import Image, ImageDraw, ImageFont
# import matplotlib.image as mpimg
from scipy.misc import imresize, imsave


# from torch.autograd import Variable

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)
        return x


class TinyYoloNet(nn.Module):
    def __init__(self, weight_file = None):
        super(TinyYoloNet, self).__init__()
        # The pretrained model we use has 20 classes
        self.num_classes = 20
        self.box_priors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
        self.num_box_priors = int(len(self.box_priors) / 2)
        self.num_output = int((5 + self.num_classes) * self.num_box_priors)
        # definition of the network
        self.cnn = nn.Sequential(OrderedDict([
            # conv1
            ('conv1', nn.Conv2d(3, 16, 3, 1, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(16)),
            ('leaky1', nn.LeakyReLU(0.1, inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),

            # conv2
            ('conv2', nn.Conv2d(16, 32, 3, 1, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(32)),
            ('leaky2', nn.LeakyReLU(0.1, inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2)),

            # conv3
            ('conv3', nn.Conv2d(32, 64, 3, 1, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('leaky3', nn.LeakyReLU(0.1, inplace=True)),
            ('pool3', nn.MaxPool2d(2, 2)),

            # conv4
            ('conv4', nn.Conv2d(64, 128, 3, 1, 1, bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('leaky4', nn.LeakyReLU(0.1, inplace=True)),
            ('pool4', nn.MaxPool2d(2, 2)),

            # conv5
            ('conv5', nn.Conv2d(128, 256, 3, 1, 1, bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('leaky5', nn.LeakyReLU(0.1, inplace=True)),
            ('pool5', nn.MaxPool2d(2, 2)),

            # conv6
            ('conv6', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            ('bn6', nn.BatchNorm2d(512)),
            ('leaky6', nn.LeakyReLU(0.1, inplace=True)),
            ('pool6', MaxPoolStride1()),

            # conv7
            ('conv7', nn.Conv2d(512, 1024, 3, 1, 1, bias=False)),
            ('bn7', nn.BatchNorm2d(1024)),
            ('leaky7', nn.LeakyReLU(0.1, inplace=True)),

            # conv8
            ('conv8', nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)),
            ('bn8', nn.BatchNorm2d(1024)),
            ('leaky8', nn.LeakyReLU(0.1, inplace=True)),

            # output
            ('output', nn.Conv2d(1024, self.num_output, 1, 1, 0)),
        ]))
        # cast all parameters to float datatype
        self.cnn.float()
        # set the network in evaluation mode
        self.cnn.eval()
        self.weight_file = weight_file
        if weight_file:
            # buffer to the weight file
            self.buf = np.fromfile(self.weight_file, dtype=np.float32)
            # first four float number in weight file are not weights
            self.pos = 4
            self.load_weights()

    def forward(self, x):
        x = self.cnn(x)
        return x

    def load_weights(self):
        # Conv1
        self.load_conv_bn(self.cnn[0], self.cnn[1])
        # Conv2
        self.load_conv_bn(self.cnn[4], self.cnn[5])
        # Conv3
        self.load_conv_bn(self.cnn[8], self.cnn[9])
        # Conv4
        self.load_conv_bn(self.cnn[12], self.cnn[13])
        # Conv5
        self.load_conv_bn(self.cnn[16], self.cnn[17])
        # Conv6
        self.load_conv_bn(self.cnn[20], self.cnn[21])
        # Conv7
        self.load_conv_bn(self.cnn[24], self.cnn[25])
        # Conv8
        self.load_conv_bn(self.cnn[27], self.cnn[28])
        # output
        self.load_conv(self.cnn[30])

        return

    def load_conv_bn(self, conv_model, bn_model):
        num_w = conv_model.weight.numel()
        num_b = bn_model.bias.numel()
        bn_model.bias.data.copy_(torch.from_numpy(self.buf[self.pos:self.pos + num_b]))
        self.pos += num_b
        bn_model.weight.data.copy_(torch.from_numpy(self.buf[self.pos:self.pos + num_b]))
        self.pos += num_b
        bn_model.running_mean.copy_(torch.from_numpy(self.buf[self.pos:self.pos + num_b]))
        self.pos += num_b
        bn_model.running_var.copy_(torch.from_numpy(self.buf[self.pos:self.pos + num_b]))
        self.pos += num_b
        conv_model.weight.data.copy_(torch.from_numpy(self.buf[self.pos:self.pos + num_w]).view_as(conv_model.weight.data))
        self.pos += num_w

        return

    def load_conv(self, conv_model):
        num_w = conv_model.weight.numel()
        num_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(torch.from_numpy(self.buf[self.pos:self.pos + num_b]))
        self.pos += num_b
        conv_model.weight.data.copy_(torch.from_numpy(self.buf[self.pos:self.pos + num_w]).view_as(conv_model.weight.data))
        self.pos += num_w

        return

    def detect(self,img):
        # convert image to torch tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        img = torch.autograd.Variable(img)
        # Make forward pass for image, and keep values as output
        output = self.cnn(img).data



        return output

def sigmoid(x):
    # compute sigmoid of float input
    return 1 / (1 + math.exp(-x))

def get_region_boxes(output, conf_thresh, box_priors, num_box_priors):

    batch = output.size(0)
    h = output.size(2)
    w = output.size(3)

    all_boxes = []
    for b in range(batch):
        for d in range(num_box_priors):
            conf_map = output[b, d * 25 + 4, :, :]
            conf_map = torch.sigmoid(conf_map)  # convert to [0,1] score
            detect = (conf_map > conf_thresh)  # binary tensors
            positive_loc = torch.nonzero(detect)
            for i in range(len(positive_loc)):
                cx = positive_loc[i][0]  # NOT SURE OF THE INDEXING ORDER
                cy = positive_loc[i][1]
                logits = output[b, d * 25 + 5:d * 25 + 25, cx, cy]
                # compute softmax function
                cls_confs = torch.exp(logits) / torch.sum(torch.exp(logits))
                # define most likely class
                cls_conf, cls_idx = torch.max(cls_confs, 0)
                if int(cls_idx[0])==6:  # we're only interested in cars
                    # extract box parameters prediction
                    tx = output[b, d * 25, cx, cy] # this is no longer a tensor. Float.
                    ty = output[b, d * 25+1 , cx, cy]
                    tw = output[b, d * 25 + 2, cx, cy]
                    th = output[b, d * 25 + 3, cx, cy]
                    detect_conf = output[b,d*25+4, cx, cy]
                    # transform parameter into box centers and dimensions. See YOLO9000 for these formulas.
                    # NOTE : we use cx to get y coord and vice-versa. This is due to the orientation of the axes in the
                    # image: x is horizontal, pointing to the right, and y is vertical, pointing to the bottom. In a 2D
                    # matrix, the first axis correspond to the rows, and the second to the columns. Hence, the
                    # representaiton of the image is flipped wrt. the original and we need this to get the center coord. in the image axis.
                    bx = sigmoid(tx) + cy
                    by = sigmoid(ty) + cx
                    bw = box_priors[2*d]*np.exp(tw)
                    bh = box_priors[2*d+1]*np.exp(th)
                    all_boxes.append([bx/w,by/h,bw/w,bh/h,detect_conf,cls_conf])

    return all_boxes

def remove_duplicates(boxes):
    if len(boxes)==0:
        return []
    # sort boxes by order of confidence
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = boxes[i][4]
    _, sort_idx = torch.sort(det_confs,descending=True)
    # loop over the boxes
    sorted = [boxes[sort_idx[i]] for i in range(len(boxes))]
    remove=[]
    for i in range(len(sorted)):
        for j in range(i+1,len(sorted)):
            if boxes_overlap(sorted[i],sorted[j])>0.4:
                remove.append(j)
    unique_boxes = [sorted[i] for i in range(len(sorted)) if i not in remove]
    # loop over subsequent boxes
    # remove any box that significantly overlap the first one
    return unique_boxes

def boxes_overlap(box1,box2):
    # Here we compute a scalar indicating if two bounding boxes overlap significantly.
    # The criteria is area of the intersection of the two boxes over area of the union of the two boxes
    minx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
    maxx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
    miny = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
    maxy = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
    w1 = box1[2]
    h1 = box1[3]
    w2 = box2[2]
    h2 = box2[3]
    union_w = maxx - minx
    union_h = maxy - miny
    inter_w = w1 + w2 - union_w
    inter_h = h1 + h2 - union_h
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    inter_area = inter_w * inter_h
    union_area = area1 + area2 - inter_area
    return inter_area / union_area

def plot_boxes(img, boxes, savename=None):
    width = img.shape[1]
    height = img.shape[0]

    #    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        # box is a tuple : (x_center,y_center,width,height,confidence,class)
        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)
        cv2.rectangle(img, (x2, y2), (x1, y1), (255, 0, 0), 2)
    if savename:
        print("save plot results to %s" % savename)
        # Note the conversion to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(savename, img)

    return img
if __name__ == '__main__':
    from example2 import *

    m = TinyYoloNet(weight_file='tiny-yolo-voc.weights')

    #    img = Image.open('test_images/test1.jpg').convert('RGB')
    img = cv2.imread('test_images/test1.jpg')
    # BGR to RGB conversion
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # TinyYolo expects 46x416 images
    sized = imresize(img, (416, 416))
    # compute output of network
    output = m.detect(sized)
    # convert out put to boxes
    conf_thresh = 0.5
    boxes = get_region_boxes(output, conf_thresh, m.box_priors, m.num_box_priors)
    boxes = remove_duplicates(boxes)
    plot_boxes(img, boxes, 'predict2.jpg')
