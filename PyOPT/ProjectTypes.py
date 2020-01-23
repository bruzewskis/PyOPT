#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:57:54 2020

@author: bruzewskis
"""

from PyOPT import ProgramTypes
from PyOPT import _Container

class Project(_Container._Container):
    
    def __init__(self, name):
        super().__init__(name, ProgramTypes.Program)
        
    
    