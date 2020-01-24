#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:02:02 2020

@author: bruzewskis
"""

class _Container():
    
    def __init__(self, name, _item_class):
        
        self.name = name
        self._items = []
        
        self._class_name = self.__class__.__name__
        
        self._item_class = _item_class
        self._item_class_name = _item_class.__name__
        
    def __str__(self):
        
        num_items = len(self._items)
        outtext = '{}: {}\n'.format(self._class_name, self.name)
        outtext += '='*(len(outtext)-1)
        
        if num_items > 0:
            for item in self._items:
                outstr = '\n\t - {}'
                outtext += outstr.format(item.name)
        else:
            outtext += '\n[No {}s]'.format(self._item_class_name)
            
        return outtext
    
    def __repr__(self):
        
        num_items = len(self._items)
        outtext = '{} <{} length={}>'.format(self.name, self._class_name, 
                                             num_items)
        
        if num_items > 0:
            for i in range(len(self._items)):
                item = self._items[i]
                outstr = '\n ({}) {} <{} length={}>'
                outtext += outstr.format(i, item.name, 
                                         self._item_class_name, len(item))
            
        return outtext
    
    def __len__(self):
        
        return len(self._items)
    
    def __getitem__(self, key):
        
        return self._items[key]
    
    def __setitem__(self, key, value):
        
        self._items[key] = value
        return None
    
    def append(self, item):
        
        if isinstance(item, self._item_class):
            self._items.append(item)
        else:
            raise TypeError('Only items may be appended to a project')
        
    def pprint(self, _prefix=''):
        
        outtext = _prefix + self.name + '\n'
        for item in self._items:
            outtext += item.pprint(_prefix=_prefix+'\t')
        
        return outtext