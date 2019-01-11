import torch.nn as nn
from typing import List, Tuple
from .pytorch_utils import (BatchNorm1d, BatchNorm2d, BatchNorm3d, Conv1d,
                            Conv2d, Conv3d, FC)


class Seq(nn.Sequential):

    def __init__(self, input_channels):
        super().__init__()
        self.count = 0
        self.current_channels = input_channels

    def conv1d(self,
               out_size: int,
               *,
               kernel_size: int = 1,
               stride: int = 1,
               padding: int = 0,
               dilation: int = 1,
               activation=nn.ReLU(inplace=True),
               bn: bool = False,
               init=nn.init.kaiming_normal_,
               bias: bool = True,
               preact: bool = False,
               name: str = "",
               norm_layer=BatchNorm1d):

        self.add_module(
            str(self.count),
            Conv1d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name,
                norm_layer=norm_layer))
        self.count += 1
        self.current_channels = out_size

        return self

    def conv2d(self,
               out_size: int,
               *,
               kernel_size: Tuple[int, int] = (1, 1),
               stride: Tuple[int, int] = (1, 1),
               padding: Tuple[int, int] = (0, 0),
               dilation: Tuple[int, int] = (1, 1),
               activation=nn.ReLU(inplace=True),
               bn: bool = False,
               init=nn.init.kaiming_normal_,
               bias: bool = True,
               preact: bool = False,
               name: str = "",
               norm_layer=BatchNorm2d):

        self.add_module(
            str(self.count),
            Conv2d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name,
                norm_layer=norm_layer))
        self.count += 1
        self.current_channels = out_size

        return self

    def conv3d(self,
               out_size: int,
               *,
               kernel_size: Tuple[int, int, int] = (1, 1, 1),
               stride: Tuple[int, int, int] = (1, 1, 1),
               padding: Tuple[int, int, int] = (0, 0, 0),
               dilation: Tuple[int, int, int] = (1, 1, 1),
               activation=nn.ReLU(inplace=True),
               bn: bool = False,
               init=nn.init.kaiming_normal_,
               bias: bool = True,
               preact: bool = False,
               name: str = "",
               norm_layer=BatchNorm3d):

        self.add_module(
            str(self.count),
            Conv3d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name,
                norm_layer=norm_layer))
        self.count += 1
        self.current_channels = out_size

        return self

    def fc(self,
           out_size: int,
           *,
           activation=nn.ReLU(inplace=True),
           bn: bool = False,
           init=None,
           preact: bool = False,
           name: str = ""):

        self.add_module(
            str(self.count),
            FC(self.current_channels,
               out_size,
               activation=activation,
               bn=bn,
               init=init,
               preact=preact,
               name=name))
        self.count += 1
        self.current_channels = out_size

        return self

    def dropout(self, p=0.5):

        self.add_module(str(self.count), nn.Dropout(p=0.5))
        self.count += 1

        return self

    def maxpool2d(self,
                  kernel_size,
                  stride=None,
                  padding=0,
                  dilation=1,
                  return_indices=False,
                  ceil_mode=False):
        self.add_module(
            str(self.count),
            nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                return_indices=return_indices,
                ceil_mode=ceil_mode))
        self.count += 1

        return self
