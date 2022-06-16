import os
from math import exp
from random import uniform

import numpy as np


class Perceptron:

    def __init__(self, name, learning_parameter):
        self.name = name
        self.learning_parameter = learning_parameter
        self.letters = dict.fromkeys(
            ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z'], 0)
        self.weigths = np.array([uniform(0, 1) for _ in range(26)])
        self.error = 0
        self.current_output = 0

    def file_freq(self, file):
        self.letters = dict.fromkeys(
            ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z'], 0)
        size = 0

        with open(file, 'r', encoding="utf-8") as f:
            for line in f:
                line = list(line)
                for char in line:
                    char = char.lower()
                    if char in self.letters:
                        self.letters[char] += 1
                        size += 1
        for k in self.letters:
            self.letters[k] /= size

    def update_weights(self, dir_name):

        d = 1 if dir_name == self.name else 0
        x = np.array(list(self.letters.values()))
        y = self.current_output
        self.weigths += self.learning_parameter * (d - y) * y * (1 - y) * x
        self.calc_error(d, y)

    def calc_error(self, d, y):
        self.error = 0.5 * (d - y) ** 2

    def calc_activation_function(self):
        self.current_output = 0
        x = np.array(list(self.letters.values()))
        dot_product = np.sum(x * self.weigths)
        net_value = dot_product
        y = 1 / (1 + exp(-net_value))
        self.current_output = y

    def process(self, file):
        self.file_freq(file)
        self.calc_activation_function()
        cur_dir = os.path.split(os.path.split(os.path.abspath(file))[0])[1]
        self.update_weights(cur_dir)

    def test(self, file):
        self.file_freq(file)
        self.calc_activation_function()
