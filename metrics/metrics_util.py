#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import logging

import numpy as np
from metrics.kinetics import accuracy_metrics as kinetics_metrics

logger = logging.getLogger(__name__)


class Metrics(object):
    def __init__(self, name, mode, metrics_args):
        """Not implemented"""
        pass

    def calculate_and_log_out(self, loss, pred, label, info=''):
        """Not implemented"""
        pass

    def accumulate(self, loss, pred, label, info=''):
        """Not implemented"""
        pass

    def finalize_and_log_out(self, info=''):
        """Not implemented"""
        pass

    def reset(self):
        """Not implemented"""
        pass


class Kinetics400Metrics(Metrics):
    def __init__(self, name, mode, metrics_args):
        self.name = name
        self.mode = mode
        self.calculator = kinetics_metrics.MetricsCalculator(name, mode.lower())

    def calculate_and_log_out(self, loss, pred, label, info=''):
        if loss is not None:
            loss = np.mean(np.array(loss))
        else:
            loss = 0.
        acc1, acc5 = self.calculator.calculate_metrics(loss, pred, label)
        logger.info(info + '\tLoss: {},\ttop1_acc: {}, \ttop5_acc: {}'.format('%.6f' % loss, \
                       '%.2f' % acc1, '%.2f' % acc5))
        return loss

    def accumulate(self, loss, pred, label, info=''):
        self.calculator.accumulate(loss, pred, label)

    def finalize_and_log_out(self, info=''):
        self.calculator.finalize_metrics()
        metrics_dict = self.calculator.get_computed_metrics()
        loss = metrics_dict['avg_loss']
        acc1 = metrics_dict['avg_acc1']
        acc5 = metrics_dict['avg_acc5']
        logger.info(info + '\tLoss: {},\ttop1_acc: {}, \ttop5_acc: {}'.format('%.6f' % loss, \
                       '%.2f' % acc1, '%.2f' % acc5))
        return acc1

    def reset(self):
        self.calculator.reset()


class MetricsZoo(object):
    def __init__(self):
        self.metrics_zoo = {}

    def regist(self, name, metrics):
        assert metrics.__base__ == Metrics, "Unknow model type {}".format(
            type(metrics))
        self.metrics_zoo[name] = metrics

    def get(self, name, mode, cfg):
        for k, v in self.metrics_zoo.items():
            if k == name:
                return v(name, mode, cfg)
        raise MetricsNotFoundError(name, self.metrics_zoo.keys())


# singleton metrics_zoo
metrics_zoo = MetricsZoo()


def regist_metrics(name, metrics):
    metrics_zoo.regist(name, metrics)


def get_metrics(name, mode, cfg):
    return metrics_zoo.get(name, mode, cfg)


# sort by alphabet
regist_metrics("TSM", Kinetics400Metrics)
