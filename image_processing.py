#!/usr/bin/env python

"""
@author: Jafar Jabr <jafaronly@yahoo.com>
=======================================
All what is related to Ai, Pytorch and image processing are from
https://github.com/udacity/deep-learning-v2-pytorch
=> convolutional-neural-networks/conv-visualization/maxpooling_visualization.ipynb
"""
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QDesktopWidget, QFileDialog
import Net


class Main:
    def __init__(self):
        app = QApplication([])
        app.setStyle('WindowsVista')
        app.setLayoutDirection(Qt.RightToLeft)
        window = QWidget()
        window.setWindowTitle("AI + GUI")
        window.setGeometry(50, 50, 500, 300)
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle = window.frameGeometry()
        qtRectangle.moveCenter(centerPoint)
        window.move(qtRectangle.topLeft())
        layout = QVBoxLayout()
        btn0 = QPushButton('Choose Image')
        btn0.clicked.connect(lambda me: self.open_file())
        layout.addWidget(btn0)
        window.setLayout(layout)
        window.show()
        app.exec_()

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        img_path, _ = QFileDialog.getOpenFileName(None, "choose an image", "",
                                                  "Image Files (*.jpg *.png)", options=options)
        if img_path:
            plt.close('all')
            image = plt.imread(img_path)
            fig = plt.figure(figsize=(20, 20))
            fig.canvas.set_window_title("Image processing")
            ax = fig.add_subplot(3, 2, 1)
            ax.imshow(image, cmap='gray')
            ax.set_title('Original Image')
            self.analyze_image(fig, img_path)
            plt.show()

    def analyze_image(self, fig, img_path, layer="convolutional"):
        # load color image
        bgr_img = cv2.imread(img_path)
        # convert to grayscale
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        # normalize, rescale entries to lie in [0,1]
        gray_img = gray_img.astype("float32") / 255
        # plot image
        filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])
        print('Filter shape: ', filter_vals.shape)
        filter_1 = filter_vals
        filter_2 = -filter_1
        filter_3 = filter_1.T
        filter_4 = -filter_3
        filters = np.array([filter_1, filter_2, filter_3, filter_4])
        # For an example, print out the values of filter 1
        print('Filter 1: \n', filter_1)
        weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
        model = Net.Net(weight)
        # print out the layer in the network
        print(model)
        gray_scaled = fig.add_subplot(3, 2, 2)
        gray_scaled.imshow(gray_img, cmap='gray')
        gray_scaled.set_title('Grayscaled')
        for i in range(4):
            _filter = fig.add_subplot(3, 9, i + 11, xticks=[], yticks=[])
            _filter.imshow(filters[i], cmap='gray')
            _filter.set_title('Filter %s' % str(i + 1))
        # convert the image into an input Tensor
        gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)
        # get all the layers
        conv_layer, activated_layer, pooled_layer = model(gray_img_tensor)
        # visualize the output of the given layer
        if layer == "convolutional":
            self.viz_layer(fig, conv_layer)
        elif layer == "activated":
            self.viz_layer(fig, activated_layer)
        elif layer == "pooled":
            self.viz_layer(fig, pooled_layer)

    @staticmethod
    def viz_layer(fig, layer, n_filters=4):
        for i in range(n_filters):
            ax = fig.add_subplot(3, n_filters, i + 9)
            # grab layer outputs
            ax.imshow(np.squeeze(layer[0, i].data.numpy()), cmap='gray')
            ax.set_title('Output %s' % str(i + 1))


if __name__ == '__main__':
    Main()
