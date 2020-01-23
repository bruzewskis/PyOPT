#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:30:55 2020

@author: bruzewskis
"""
from PyOPT import ScanTypes
from PyOPT import _Container

class Block(_Container._Container):
    
    def __init__(self, name):
        super().__init__(name, ScanTypes.Scan)