#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimEVLA.py: This module simulates the slewing motion of the EVLA for use in 
determining optimized schedules. Based largely on the equivalent NRAO SSS code. 
"""

__author__ = "Seth Bruzewski"
__email__ = "bruzewskis@unm.edu"
__license__ = "GPL"

# Built in Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord,Angle
from tqdm import tqdm, trange
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from SimEVLA import EquitorialToHorizontal as E2H, PointToPointTime as P2PT


def write_OPT(scheds, ha_offsets):
    '''
    Take in a list of schedules in schedmaster format and output appropriate
    VLA OPT schedule files
    '''
        
    # Templates for scan
    preamble1 = ('VERSION; 4;\n'
                 'SRC-CAT; NCPSources, VLA;\n'
                 'HDWR-CAT; NRAO Defaults, Project 17B-165;\n\n')
    
    preamble2 = ('SCHED-BLOCK; {schedBlockName}; {schedulingType}; '
                 '{iterationCount}; {date}; {timeOfDay}; {shadowLimit}; '
                 '{shadowCalcConfiguration}; {initTeleAz}; {initTeleEl}; '
                 '{avoidSunrise?}; {avoidSunset?}; {windApi}; '
                 '{commentsToOperator};\n\n')
    
    scanline = ('  STD; {scanName}; {sourceName}; {resourceName}; {timeType}; '
                '{time}; {antennaWrap}; {applyRefPtg}; {applyPhase}; '
                '{recordOnMark5}; {allowOverTop}; {use10HzNoise}; '
                '{scanIntents}; {comments};\n')
    
    # Templates for sources
    sourceline = ('{sourceName}; {groupNames}; {coordinateSystem}; {epoch};'
                  '{lonCent}; {latCent}; {velocityRefFrame};'
                  '{velocityConvention}; {velocity}; {calibrator};\n')
    
    # Keep track of any sources we use
    sources = {} # name -> blocklist,RA,DEC
    
    # Loop on each block
    for i in range(len(scheds)):
        
        # Calculate some things we want to know ahead of time
        # LST of first scan
        lst0 = Angle(scheds[i]['start_lst'][0], unit=u.hourangle)
        lst0 = Angle(lst0.hourangle//(5/60)*(5/60), unit=u.hourangle)
        print('='*10,'Writing Block', i+1, '='*10)
        print('Starts at', lst0)
        
        # Figure out LST start range
        lst_min = lst0-10/60*u.hourangle
        lst_min_str = lst_min.to_string('h', sep=':', pad=True)[:5]
        lst_max = lst0+10/60*u.hourangle
        lst_max_str = lst_max.to_string('h', sep=':', pad=True)[:5]
        print('Start range:', lst_min_str+'-'+lst_max_str)
        
        # open block file
        scanfile = 'outputs/block'+str(i+1)+'.optScan'
        bf = open(scanfile, 'w')
        
        # Write preamble 1
        bf.write(preamble1)
        
        # Define preamble 2 params
        p2_parms = {'schedBlockName': 'block'+str(i+1),
                    'schedulingType': 'Dynamic',
                    'iterationCount': 1,
                    'date': Time.now().iso[:-4]+', 2099-12-31 23:59:59',
                    'timeOfDay': lst_min_str+'-'+lst_max_str+',',
                    'shadowLimit': 0.0,
                    'shadowCalcConfiguration': 'A',
                    'initTeleAz': 225.0,
                    'initTeleEl': 35.0,
                    'avoidSunrise?': 'N',
                    'avoidSunset?': 'N',
                    'windApi': 'w=100.0,p=45.0',
                    'commentsToOperator': ''}
        bf.write(preamble2.format(**p2_parms))
        
        # Convenient dictionary format
        sched = scheds[i].to_pandas().to_dict('records')
        
        # Loop on each source
        for scan in sched:
            
            # Decode bytes objects generated during reformation
            decode_scan = {}
            for col in scan:
                if isinstance(scan[col], bytes):
                    decode_scan[col] = scan[col].decode()
                else:
                    decode_scan[col] = scan[col]
                    
            # Write to line
            bf.write(scanline.format(**decode_scan ))
            
            name = decode_scan['sourceName']
            is_scan = decode_scan['scanName'] == ''
            if is_scan and name not in sources:
                sources[name] = {'ra': decode_scan['ra'], 
                                 'dec': decode_scan['dec']}
                
        # close block file
        print('Scan file written to', scanfile)
        bf.close()
        
    
    # Open source list file
    sourcefile = 'outputs/NCP_Sources.pst'
    sf = open(sourcefile, 'w')
    for source in sorted(sources.keys()):
        # Define sourceline for source
        source_parms = {'sourceName': source,
                        'groupNames': '',
                        'coordinateSystem': 'Equatorial',
                        'epoch': 'J2000',
                        'lonCent': sources[source]['ra'],
                        'latCent': sources[source]['dec'],
                        'longRange': '',
                        'latRange': '',
                        'velocityRefFrame': '',
                        'velocityConvention': '',
                        'velocity': '',
                        'calibrator': ''}
        # Write
        sf.write(sourceline.format(**source_parms))
        
    # close source list file
    print('Source file written to', sourcefile)
    sf.close()
    
    return None

if __name__=='__main__':
    test = Table.read('testingblock.fits')
    testing = [test, test]
    test_ha = [0]*len(testing)
    write_OPT(testing, test_ha)
    
    