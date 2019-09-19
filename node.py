#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This is a node entity class"""


class Node:

    def __init__(self, name, mapping, embedding):
        self.name = name
        self.mapping = mapping
        self.embedding = embedding
