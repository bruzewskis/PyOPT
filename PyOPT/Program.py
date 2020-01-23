#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:34:53 2020

@author: bruzewskis
"""

from PyOPT import Block
from PyOPT import _Container

class Program(_Container._Container):
    
    def __init__(self, name):
        super().__init__(name, Block.Block)