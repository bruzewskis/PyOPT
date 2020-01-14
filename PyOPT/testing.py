#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Docstring
'''

class Project():
    '''
    class docstring
    '''
    # Define relevant private class variables
    __projectName = 'TestProject'
    
    __configs = []
    
    def __init__(self,name=None):
        '''
        function docstring
        '''
        if not name is None:
            self.setProjectName(name)
        
    def __repr__(self):
        '''
        function docstring
        '''
        
        fmt_str = '{} ({})'
        numberItems = len(self.__configs)
        
        return fmt_str.format(self.__projectName, numberItems)
        
    def setProjectName(self, newname):
        '''
        function docstring
        '''
        self.__projectName = newname
        
    def getProjectName(self):
        '''
        function docstring
        '''
        return self.__projectName
    
    def show(self, style='default'):
        print('-', self.__projectName)
        for config in self.__configs:
            config.show()
        
    def addConfig(self, config=None):
        '''
        function docstring
        '''
        if config is None:
            self.__configs.append( Config() )
        else:
            if isinstance(config, Config):
                self.__configss.append( config )
            else:
                raise TypeError('Must provide a Config object')
    
class Config():
    '''
    class docstring
    '''
    # Define relevant private class variables
    __configName = 'TestConfig'
    
    __blocks = []
    
    def __init__(self, name=None):
        '''
        function docstring
        '''
        
    def __repr__(self):
        '''
        function docstring
        '''
        return self.__configName
        
    def setConfigName(self, newname):
        '''
        function docstring
        '''
        self.__configName = newname
        
    def getConfigName(self):
        '''
        function docstring
        '''
        return self.__configName
    
    def show(self):
        '''
        function docstring
        '''
        print('\t-', self.__configName)
        for block in self.__blocks:
            block.show()
            
class Block():
    '''
    class docstring
    '''
    # Define relevant private class variables
    __blockName = 'TestBlock'
    
    __blocks = []
    
    def __init__(self, name=None):
        '''
        function docstring
        '''
        
    def __repr__(self):
        '''
        function docstring
        '''
        return self.__configName
        
    def setConfigName(self, newname):
        '''
        function docstring
        '''
        self.__configName = newname
        
    def getConfigName(self):
        '''
        function docstring
        '''
        return self.__configName
    
    def show(self):
        '''
        function docstring
        '''
        print('\t-', self.__configName)
    
        
        
    
    
a = Project()
a.addConfig()
a.show()

a.setProjectName('foo')
print(a)