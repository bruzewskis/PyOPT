#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
schedmaster.py: This code identifies candidates for followup with 12 hours of
further VLA time on FERMI sources, then helps to create a schedule, testing
that it's possible. If everything looks good, we send the schedule to the
to be converted to OPT format and write it to disk. Note that this code serves
also as a prototype for the pythonic OPT project. 
"""

__author__ = "Seth Bruzewski"
__email__ = "bruzewskis@unm.edu"
__license__ = "GPL"

# Third Party Modules
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord,Angle,EarthLocation
from astropy.table import Table
import astropy.units as u
from scipy.optimize import brentq

# Custom Modules
from SimEVLA import EquitorialToHorizontal as E2H, MotionSimulator
from SimEVLA import PointToPointTime as P2P
from optscheduler import write_OPT

def block_argsort(ra, dec):
    '''
    Sort the sources into blocks of RA, moving up and down in declination. 
    With the name 'argsort' borrowed from numpy, this function returns the
    indices in the sorted order, allowing for arrays or tables to be resorted
    easily. 
    
    Args
    =======
    ra (array) - The RA of the sources to be sorted
    dec (array) - The dec of the sources to be sorted
    
    Returns
    =======
    new_order (array) - The sorted indices, to be used in sorting objects
    
    Raises
    =======
    None
    '''
    
    # Divide into hours
    min_block = min(ra)//30*30
    max_block = max(ra)//30*30+30
    num_hours = (max_block - min_block) / 30 # 30 = 2 hr blocks
    div = np.linspace(min_block, max_block, num_hours+1)
    
    # Generate index
    index = np.arange(len(dec))
    
    new_order = np.array([], dtype=int)
    for i in range(len(div)-1):
        # Find indices inside RA range
        inrange = np.logical_and(ra>div[i], ra<=div[i+1])
        ind = index[inrange]
        
        # Sort RA filtered indices by DEC
        inorder = ind[np.argsort(dec[ind])]
        
        # Flip every other one
        if i%2!=0:
            inorder = np.flip(inorder)
        
        # Append
        new_order = np.append(new_order, inorder)
    
    return new_order

def plot_sky(sky, show_tour=False):
    '''
    From input coordinate pairs expressed in degrees, convert to the
    appropriate system and plot in a Mollweide projection friendly way.
    
    Args
    =======
    sky (array) - A (2,N) array of points in RA and DEC to be plotted.
    show_tour (bool) - If True, plot lines connecting points in order.
    
    Returns
    =======
    None
    
    Raises
    =======
    None
    '''
    
    ra = Angle(sky[:,0], unit='deg').wrap_at('180d').rad
    dec = np.deg2rad(sky[:,1])
    
    plt.scatter(ra,dec,s=10)
    
    if show_tour:
        for i in range(len(sky)-1):
            plt.plot([ra[i],ra[i+1]], [dec[i], dec[i+1]])
            
    return None
    
def make_block(data, ramin=None, ramax=None, dcmin=None, dcmax=None):
    '''
    Returns a subset of `data` fitting just inside the bounds provided as
    inputs. Note that because ra is a wrapping coordinate system, some 
    considerations must be made. Namely when we say "angle a is between amin
    and amax," we mean more specifically "angle a is counterclockwise of amin
    and clockwise of amax." The sources are downselected using this method,
    then sorted by their RA and DEC into convenient blocks
    
    Args
    =======
    data (Table) - The data to be filtered down.
    ramin (float) - The minimum RA, specified in degrees.
    ramax (float) - The maximum RA, specified in degrees.
    dcmin (float) - The minimum Dec, specified in degrees.
    dcmax (float) - The maximum Dec, specified in degrees.
    
    Returns
    =======
    outdata (Table) - The input `data` table, filtered and sorted
    
    Raises
    =======
    None
    '''
    
    # Filter for Valid RA
    ra = data['RAJ2000']
    ra_minus = (ra-ramin)%360
    ra_max_minus = (ramax-ramin)%360
    ra_inbounds = np.logical_and(ra_minus>0, ra_minus<ra_max_minus)
    
    # Filter for Valid Dec
    dec = data['DEJ2000']
    dec_inbounds = np.logical_and(dec>dcmin, dec<dcmax)
    
    # All sources in bounds
    inblock = np.logical_and(ra_inbounds, dec_inbounds)
    
    # Find coordinates inside requested area
    inblock_ind = np.nonzero(inblock)[0]
    ra_inblock = data['RAJ2000'][inblock_ind]
    dec_inblock = data['DEJ2000'][inblock_ind]
    
    # Figure out best place to wrap
    radmin = np.deg2rad(ramin)
    radmax = np.deg2rad(ramax)
    dist = np.arctan2(np.sin(radmax-radmin), np.cos(radmax-radmin))
    wrap = radmin + dist%(2*np.pi)/2 + np.pi
    
    # Wrap at best spot then sort into RA-DEC blocks
    ra_wrapped = Angle(ra_inblock).wrap_at(wrap*u.rad).deg
    blocksort_ind = block_argsort(ra_wrapped, dec_inblock)
    master_ind = inblock_ind[blocksort_ind]
    
    # Generate data subset
    outdata = data[master_ind]
    
    return outdata

def find_nearest_cal(sc, calibrators, config):
    '''
    For an input coordinate, returns the nearest high quality VLA calibrator
    for the specified band and array configuration. 
    
    Args
    =======
    sc (SkyCoord) - The sky coordinate to search around for a calibrator.
    band (string) - The observational band.
    config (string) - The observational array configuration.
    
    Returns
    =======
    best_sc (SkyCoord) - The skycoordinate of the nearest calibrator.
    name (string) - The J2000 epoch name for the returned calibrator.
    
    Raises
    =======
    None
    '''
    
    # Narrow down to good quality
    config_qual = calibrators[config+'_qual']
    good_qual = np.logical_or(config_qual=='P', config_qual=='S')
    cals = calibrators[good_qual]
    
    # Convert to SkyCoords
    sc_cals = SkyCoord(cals['ra'], cals['dec'])
    
    # Find minimum distance calibrator
    seps = sc.separation(sc_cals).deg
    near = np.argmin(seps)
    best_sc = sc_cals[near]
    name = cals['name'][near]
    
    return best_sc, name

def worst_setup_slew(az, el):
    '''
    This function finds the worst possible setup slew, assuming the telescope
    begins the observation at the worst possible wrap end. 
    
    Args
    =======
    az (Angle) - The target azimuth of the first source
    el (Angle) - The target elevation of the first source
    
    Returns
    =======
    worst_time (Quantity) - The time the worst possible slew will take
    
    Raises
    =======
    None
    '''
    
    # HARD CODING SLEW SPEEDS
    vel_az = 40/60 * u.Unit('deg/s')
    vel_el = 20/60 * u.Unit('deg/s')
    
    # HARD CODING WRAP LIMITS
    az_ends = np.array([-85, 445])
    el_ends = np.array([0,90])
    
    # Figure out max in each direction
    az_dist = max(abs(az_ends-az.deg)) * u.deg
    el_dist = max(abs(el_ends-el.deg)) * u.deg
    
    time = max([ az_dist/vel_az, el_dist/vel_el])
    
    return time

def to_dur_fmt(t):
    '''
    Convert to format duration likes, input is time object
    '''
    time_hour = t.to('hour').value
    time_str = Angle(time_hour, unit=u.hourangle).to_string('h', pad=True)
    
    return time_str

def skyAz(lst, ra, dec):
    '''
    Helper function to predict_target. Extrapolates the azimuthal position for
    a source as a function of LST
    '''
    
    # Let's just assume we're at the VLA
    vla = EarthLocation.of_site('VLA')
    ha = Angle(lst*u.hourangle).rad - np.deg2rad(ra)
    lat = vla.lat.rad
    dec = np.deg2rad(dec)
    
    # Do Trigonometry
    cosAlt_sinAz = -np.cos(dec)*np.sin(ha)
    cosAlt_cosAz = np.sin(dec)*np.cos(lat) - np.cos(dec)*np.cos(ha)*np.sin(lat)
    
    Az = np.rad2deg(np.arctan2(cosAlt_sinAz,cosAlt_cosAz))%360
    
    return Az    

def skyEl(lst, ra, dec):
    '''
    Helper function to predict_target. Extrapolates the elevation position for
    a source as a function of LST
    '''
    
    # Let's just assume we're at the VLA
    vla = EarthLocation.of_site('VLA')
    ha = Angle(lst*u.hourangle).rad - np.deg2rad(ra)
    lat = vla.lat.rad
    dec = np.deg2rad(dec)
    
    # Do Trigonometry
    sinAlt = np.sin(dec)*np.sin(lat) + np.cos(dec)*np.cos(ha)*np.cos(lat)
    
    Alt = np.rad2deg(np.arcsin(sinAlt))
    
    return Alt

def predict_target(tgt_ra, tgt_dec, tel_az, tel_el, lst0):
    '''
    Predict the location of the target after the slew, such that we intersect
    the objects path of motion. This effect will be most severe near zenith.
    '''
    # Convert to floats
    ra = tgt_ra.deg
    dec = tgt_dec.deg
    tel_az = tel_az.deg
    tel_el = tel_el.deg
    lst0 = lst0.hourangle
    
    # Current telescope position and capabilities
    vel_az = 40/60 * u.Unit('deg/s').to('deg/hour')
    vel_el = 20/60 * u.Unit('deg/s').to('deg/hour')
    
    # Azimuth calculations for intercept time
    az_dir = np.sign( skyAz(lst0, ra, dec) - tel_az)
    azt = lambda lst_t : tel_az + az_dir * vel_az * (lst_t-lst0)
    az_func = lambda lst, rr, dd : azt(lst) - skyAz(lst, rr, dd)
    az_time = brentq(az_func, lst0, lst0+0.5, args=(ra, dec))
    
    # Elevation calculations for intercept time
    el_dir = np.sign( skyEl(lst0, ra, dec) - tel_el)
    elt = lambda lst_t : tel_el + el_dir * vel_el * (lst_t-lst0)
    el_func = lambda lst, rr, dd : elt(lst) - skyEl(lst, rr, dd)
    el_time = brentq(el_func, lst0, lst0+0.5, args=(ra, dec))
    
    # Final coordinates
    worst_time = max([az_time, el_time])
    azf = skyAz(worst_time, ra, dec)
    elf = skyEl(worst_time, ra, dec)
    azf_ang = Angle(azf*u.deg)
    elf_ang = Angle(elf*u.deg)
    
    return azf_ang, elf_ang

def schedule_block(block, mosaics, source_scan=280, cal_scan=300, 
                   ha_offset=0, verbose=True, config='A', wrap_pref=None):
    '''
    For input points and start LST, run a simulated observation to see if 
    the particular schedule is possible. This will produce some figures
    displaying the hour angle and elevation of the telescope over time, 
    highlighting in particular any areas where the telescope would go beyond
    certian limits. The function returns the total time the observation would
    take
    
    Args
    =======
    block (Table) - Target list in the FERMI format. The needed columns are 
        essentially just RAJ2000, DEJ2000, and Source_Name.
    mosaics (dict) - The mosaic dictionary maping each source name to its
        mosaic coordinates
    source_scan (float/Quantity) - The amount of time to spend on each 
        pointing. Default value is 30 and default units are seconds.
    cal_scan (float/Quantity) - The amount of time to spend on phase
        calibrators. Default value is 60 and default units are seconds.
    ha_offset (Quantity) - The hour angle of the first source when the
        observation will begin. For instance, if RA[0]=1h and ha_offset=-1h,
        then the observation will start at LST=0h. Defaults to 0h, such that
        things start when the first source is on the meridian. Default units
        are hourangle
    verbose (bool) - Whether to generate HA and El plots of the projected
        schedule and produce explanatory text.
    band (string) - The band which the observation will be performed in. This
        is used to search for calibrators.
    config (string) - The array configuration when the observation will be
        performed. This is used to search for calibrators.
    
    Returns
    =======
    sched_table (Table) - A table summarizing the observation. If the user
        likes how it went, this table can be passed to OPTSCHEDULER for
        formatting.
    
    Raises
    =======
    None
    '''
    
    ############### SET UP STUFF ###############
    # Check units
    if not isinstance(source_scan, u.Quantity):
        source_scan *= u.s
    if not isinstance(cal_scan, u.Quantity):
        cal_scan *= u.s
    if not isinstance(ha_offset, u.Quantity):
        ha_offset *= u.hourangle
        
    if wrap_pref is None:
        wrap_pref = ''
    
    # Define the coherence time manually
    
    # Convenience function for sc->str
    sc_to_ra = lambda sc:sc.ra.to_string('h', sep=':', pad=True)
    sc_to_dc = lambda sc:sc.dec.to_string('deg', sep=':', pad=True, 
                                          alwayssign=True)
    
    # Convenience function for time->ang
    time_to_ang = lambda t : Angle(t.to('s').value/3600, unit=u.hourangle)
    
    # Convenience function for formatting LST angle
    lst_to_str = lambda lsti : lsti.to_string('h',sep=':',pad=True)[:8]
    
    
    ############### INITIALIZE TIMEKEEPING ###############
    # Figure out when to start
    tgt0 = SkyCoord(block['RAJ2000'][0]*u.deg, block['DEJ2000'][0]*u.deg)
    lst0 = tgt0.ra + ha_offset
    lst0_str = lst0.wrap_at('360d').to_string(u.hourangle, sep=':') #for print
    
    # This will keep a running track of time
    lst = lst0.wrap_at('360d') # initialize
    total_time = 0*u.s
    
    ############### DETERMINE BEST CALIBRATOR ###############
    # Values for best flux cal
    best_flux_name = 'Cygnus A'
    best_flux_sc = SkyCoord('19h59m28.34s', '40d44\'02.12"')
    best_flux_az, best_flux_el = E2H(best_flux_sc.ra, best_flux_sc.dec, lst)
    
    # What is the worst possible slew we might have to do
    worst_slew = worst_setup_slew(best_flux_az, best_flux_el)
    setup_atten = 1*u.min
    setup_req = 30*u.s
    slew_target = P2P(best_flux_sc, tgt0, lst0)
    scan_tgt = 10*u.min
    
    # Calculate length of setup slew
    setup_slew = worst_slew - setup_atten - setup_req
    
    # Figure out delays to apply to lst0 for flux cal
    delay_slew = worst_slew + scan_tgt + slew_target
    delay_atten = setup_atten + setup_req + scan_tgt + slew_target
    delay_req = setup_req + scan_tgt + slew_target
    delay_tgt = scan_tgt + slew_target
    
    ############### SETUP SCANS ###############
    # Define scheduler keeper
    sched = []
    
    # Slew scan
    start_lst = lst0 - time_to_ang(delay_slew)
    end_lst = lst0 - time_to_ang(delay_atten)
    start_az, start_el = E2H(best_flux_sc.ra, best_flux_sc.dec, start_lst)
    end_az, end_el = E2H(best_flux_sc.ra, best_flux_sc.dec, end_lst)
    start_ha = (start_lst-best_flux_sc.ra).wrap_at('180d')
    end_ha = (end_lst-best_flux_sc.ra).wrap_at('180d')
    slew_line = {'scanName': 'slew',
                 'sourceName': best_flux_name,
                 'resourceName': 'X band pointing',
                 'timeType': 'DUR',
                 'time': to_dur_fmt(setup_slew),
                 'antennaWrap': wrap_pref,
                 'applyRefPtg': 'N',
                 'applyPhase': 'N',
                 'recordOnMark5': 'N',
                 'allowOverTop': 'N',
                 'use10HzNoise': 'Y',
                 'scanIntents': 'SetAtnGain,',
                 'comments': '',
                 'ra': sc_to_ra(best_flux_sc),
                 'dec': sc_to_dc(best_flux_sc),
                 'start_lst': lst_to_str(start_lst.wrap_at('360d')),
                 'end_lst': lst_to_str(end_lst.wrap_at('360d')),
                 'start_az': start_az.deg,
                 'end_az': end_az.deg,
                 'start_el': start_el.deg,
                 'end_el': end_el.deg,
                 'start_ha': start_ha.hourangle,
                 'end_ha': end_ha.hourangle,
                 'errors': 0}
    sched.append(slew_line)
    
    # Atten scan
    start_lst = end_lst
    end_lst = lst0 - time_to_ang(delay_req)
    start_az, start_el = end_az, end_el
    end_az, end_el = E2H(best_flux_sc.ra, best_flux_sc.dec, end_lst)
    start_ha = (start_lst-best_flux_sc.ra).wrap_at('180d')
    end_ha = (end_lst-best_flux_sc.ra).wrap_at('180d')
    atten_line = {'scanName': 'atten',
                  'sourceName': best_flux_name,
                  'resourceName': 'L16f3B',
                  'timeType': 'DUR',
                  'time': to_dur_fmt(setup_atten),
                  'antennaWrap': wrap_pref,
                  'applyRefPtg': 'N',
                  'applyPhase': 'N',
                  'recordOnMark5': 'N',
                  'allowOverTop': 'N',
                  'use10HzNoise': 'Y',
                  'scanIntents': 'SetAtnGain,',
                  'comments': '',
                  'ra': sc_to_ra(best_flux_sc),
                  'dec': sc_to_dc(best_flux_sc),
                  'start_lst': lst_to_str(start_lst.wrap_at('360d')),
                  'end_lst': lst_to_str(end_lst.wrap_at('360d')),
                  'start_az': start_az.deg,
                  'end_az': end_az.deg,
                  'start_el': start_el.deg,
                  'end_el': end_el.deg,
                  'start_ha': start_ha.hourangle,
                  'end_ha': end_ha.hourangle,
                  'errors': 0}
    sched.append(atten_line)
    
    # Req scan
    start_lst = end_lst
    end_lst = lst0 - time_to_ang(delay_tgt)
    start_az, start_el = end_az, end_el
    end_az, end_el = E2H(best_flux_sc.ra, best_flux_sc.dec, end_lst)
    start_ha = (start_lst-best_flux_sc.ra).wrap_at('180d')
    end_ha = (end_lst-best_flux_sc.ra).wrap_at('180d')
    req_line = {'scanName': 'req',
                'sourceName': best_flux_name,
                'resourceName': 'L16f3B',
                'timeType': 'DUR',
                'time': to_dur_fmt(setup_req),
                'antennaWrap': wrap_pref,
                'applyRefPtg': 'N',
                'applyPhase': 'N',
                'recordOnMark5': 'N',
                'allowOverTop': 'N',
                'use10HzNoise': 'Y',
                'scanIntents': 'SetAtnGain,',
                'comments': '',
                'ra': sc_to_ra(best_flux_sc),
                'dec': sc_to_dc(best_flux_sc),
                'start_lst': lst_to_str(start_lst.wrap_at('360d')),
                'end_lst': lst_to_str(end_lst.wrap_at('360d')),
                'start_az': start_az.deg,
                'end_az': end_az.deg,
                'start_el': start_el.deg,
                'end_el': end_el.deg,
                'start_ha': start_ha.hourangle,
                'end_ha': end_ha.hourangle,
                'errors': 0}
    sched.append(req_line)
    
    # Flux target scan
    start_lst = end_lst
    end_lst = lst0 - time_to_ang(slew_target)
    start_az, start_el = end_az, end_el
    end_az, end_el = E2H(best_flux_sc.ra, best_flux_sc.dec, end_lst)
    start_ha = (start_lst-best_flux_sc.ra).wrap_at('180d')
    end_ha = (end_lst-best_flux_sc.ra).wrap_at('180d')
    flux_tgt_line = {'scanName': '',
                     'sourceName': best_flux_name,
                     'resourceName': 'L16f3B',
                     'timeType': 'DUR',
                     'time': to_dur_fmt(scan_tgt),
                     'antennaWrap': wrap_pref,
                     'applyRefPtg': 'N',
                     'applyPhase': 'N',
                     'recordOnMark5': 'N',
                     'allowOverTop': 'N',
                     'use10HzNoise': 'Y',
                     'scanIntents': 'CalBP,CalFlux,',
                     'comments': '',
                     'ra': sc_to_ra(best_flux_sc),
                     'dec': sc_to_dc(best_flux_sc),
                     'start_lst': lst_to_str(start_lst.wrap_at('360d')),
                     'end_lst': lst_to_str(end_lst.wrap_at('360d')),
                     'start_az': start_az.deg,
                     'end_az': end_az.deg,
                     'start_el': start_el.deg,
                     'end_el': end_el.deg,
                     'start_ha': start_ha.hourangle,
                     'end_ha': end_ha.hourangle,
                     'errors': 0}
    sched.append(flux_tgt_line)
    
    
    ############### RUN SIMULATION ###############
    # Initialize at defaults
    if wrap_pref == 'CCW':
        w = 'COUNTERCLOCKWISE'
    elif wrap_pref == 'CW':
        w = 'CLOCKWISE'
    else:
        w = None
    sim = MotionSimulator(end_az, end_el, wrap=w)
    
    # Run Simulation
    for i in range(len(block)):
        # Grab mosaic coordinates
        fname = block['Source_Name'][i]
        mosi = mosaics[fname]
        
        # Generate some names and notes
        b26 = lambda n : chr(97+n//26) + chr(97+n%26)
        names = [ fname+b26(n) for n in range(len(mosi)) ]
        intents = ['ObsTgt']*len(mosi)
        dur = [source_scan]*len(mosi)
            
        # Prepend a calibrator
        csc = SkyCoord('21h04m06.937s', '+76d33\'10.41"')
        cname = '3C427.1'
        cal_arr = np.array([[csc.ra.deg, csc.dec.deg]])
        
        # Prepend calibrator
        mosi = np.concatenate((cal_arr, mosi), axis=0)
        names.insert(0, cname)
        intents.insert(0, 'CalGain')
        dur.insert(0, cal_scan)
            
        if i==len(block)-1:
            mosi = np.concatenate((mosi, cal_arr), axis=0)
            names.append(cname)
            intents.append('CalGain')
            dur.append(cal_scan)
        
        for j in range(len(mosi)):
            # Move to a new spot on sky
            scj = SkyCoord(mosi[j][0]*u.deg, mosi[j][1]*u.deg)
            
            # Trying something new
            current_az = sim.getCurrentAntennaAzimuth()
            current_el = sim.getCurrentAntennaElevation()
            azj, elj = predict_target(scj.ra, scj.dec, current_az, 
                                        current_el, lst)
            
            # Move to predicted position
            slew = sim.moveTo(azj, elj, wrap=w)
            
            # Convert slew time to an angle
            slew_ang = time_to_ang(slew)
            scan = dur[j] + 1*u.s
            scan_ang = time_to_ang(scan)
            
            # Record new time
            start_lst = lst
            lst += slew_ang + scan_ang
            total_time += slew + scan
            
            # Telescope tracked to new AltAz
            azj_new, elj_new = E2H(scj.ra, scj.dec, lst)
            sim.moveTo(azj_new, elj_new, wrap=w)
            
            # Num errors
            num_errors = len(sim.getErrors())
            
            # Calculate hour angles for recording
            start_ha = (start_lst - scj.ra).wrap_at('180d')
            end_ha = (lst - scj.ra).wrap_at('180d')
            
            # Keep track of scan
            source_line = {'scanName': '',
                           'sourceName': names[j],
                           'resourceName': 'L16f3B',
                           'timeType': 'DUR',
                           'time': to_dur_fmt(slew+scan),
                           'antennaWrap': wrap_pref,
                           'applyRefPtg': 'N',
                           'applyPhase': 'N',
                           'recordOnMark5': 'N',
                           'allowOverTop': 'N',
                           'use10HzNoise': 'Y',
                           'scanIntents': intents[j],
                           'comments': '',
                           'ra': sc_to_ra(scj),
                           'dec': sc_to_dc(scj),
                           'start_lst': lst_to_str(start_lst.wrap_at('360d')),
                           'end_lst': lst_to_str(lst.wrap_at('360d')),
                           'start_az': azj.deg,
                           'end_az': azj_new.deg,
                           'start_el': elj.deg,
                           'end_el': elj_new.deg,
                           'start_ha': start_ha.hourangle,
                           'end_ha': end_ha.hourangle,
                           'errors': num_errors}
            sched.append(source_line)
            
    sched_tab = Table(sched)
        
    # Display how run went
    if verbose:
        # Skip setup scans
        plot_tab = sched_tab[3:]
        
        # Some nice text
        print('\n'+'='*25, 'SCHEDULING BLOCK', '='*25+'\n')
        verb_str = 'Observing {} sources, starting at LST {}'
        print(verb_str.format(len(block), lst0_str))
        print('Flux cal:', best_flux_name)
        
        #DELETE ME
        print('worst slew', worst_slew)
        
        times = Angle(plot_tab['time'], unit=u.hourangle).hourangle
        start_times = np.cumsum(times)
        
        # Get errors and calibrators
        cc = ['C0' if e==0 else 'C3' for e in plot_tab['errors']]
        calind = ['CalGain' in intn for intn in plot_tab['scanIntents']]
        cal_times = start_times[calind]
        
        # Plot sky
        plt.figure(figsize=(8,8))
        ax1 = plt.subplot2grid((5,5), (0,0), rowspan=3, colspan=5, 
                               projection='mollweide')
        ra = Angle(block['RAJ2000']).wrap_at('180d').rad
        dec = Angle(block['DEJ2000']).rad
        ax1.scatter(ra, dec, s=10)
        for i in range(len(ra)-1):
            p1 = [ra[i], dec[i]]
            p2 = [ra[i+1], dec[i+1]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
        ax1.grid(True)
        
        # Plot az vs time
        ax2 = plt.subplot2grid((5,5), (3,0), colspan=3)
        ax2.scatter(start_times, plot_tab['start_az'], c=cc, s=5)
        ax2.axhline(85, ls='--', c='g')
        ax2.axhline(275, ls='--', c='r')
        for ct in cal_times:
            ax2.axvline(ct, c='C2', ls='--')
        ax2.set_ylabel('Hour Angle [hr]')
        ax2.set_xticklabels([])
        ax2.set_ylim(-85,445)
        ax2.grid(True)
        
        # Plot El vs time
        ax3 = plt.subplot2grid((5,5), (4,0), colspan=3)
        ax3.scatter(start_times, plot_tab['start_el'], c=cc, s=5)
        ax3.axhline(8, ls='--', c='k')
        for ct in cal_times:
            ax3.axvline(ct, c='C2', ls='--')
        ax3.set_xlabel('Time [hr]')
        ax3.set_ylabel('Elevation [deg]')
        ax3.set_ylim(0,90)
        ax3.grid(True)
        
        # Alt-Az map stuff
        a_az = np.deg2rad(plot_tab['start_az'])
        r_el = np.cos(np.deg2rad(plot_tab['start_el']))
        circle = np.linspace(0, 2*np.pi, len(r_el))
        r_min = np.full_like(r_el, np.cos(np.deg2rad(8)))
        
        # Plot alt-az map
        ax4 = plt.subplot2grid((5,5), (3,3), colspan=2, rowspan=2, polar=True)
        ax4.scatter(a_az, r_el, s=2)
        ax4.plot(circle, r_min, 'k--')
        ax4.plot(np.deg2rad([85,85]), [0,1], ls='--', c='g')
        ax4.plot(np.deg2rad([-85,-85]), [0,1], ls='--', c='r')
        ax4.set_ylim(0,1)
        ax4.set_xticklabels(['N', '', 'E', '', 'S', '', 'W'])
        ax4.set_yticklabels([])
        ax4.set_theta_direction(-1)
        ax4.set_theta_zero_location('N')
        ax4.grid(True)
        plt.show()
        
    # Just to be safe, clear the sim from memory
    del sim
    
    # Return scheduler
    return sched_tab

def time_block(block):
    tangs = Angle(block['time'], unit=u.hourangle).hourangle
    diff = sum(tangs)
    return diff
    
def make_best_block(block, mosaics, haoff, wrap, verbose, 
                    doplots=True):
    
    dummy = schedule_block(block, mosaics, ha_offset=haoff, 
                           wrap_pref=wrap, verbose=True)
    
    # Figure out what schedules to test
    width = -1
    steps = 10
    ha_range = np.linspace(haoff-width, haoff+width, steps)
    
    # Generate schedules
    sub_blocks = []
    for ha in ha_range:
        sblock = schedule_block(block, mosaics, ha_offset=ha, 
                                wrap_pref=wrap, verbose=verbose)
        sub_blocks.append(sblock)
        
    # Some helpful plots, for now
    if doplots:
        el_min = [ min(sb['end_el']) for sb in sub_blocks]
        az_min = [ min(sb['end_az']) for sb in sub_blocks]
        az_max = [ max(sb['end_az']) for sb in sub_blocks]
        bl_tme = [ time_block(sb) for sb in sub_blocks]
        
        ax1 = plt.subplot2grid((2,2), (0,0))
        ax1.plot(ha_range, el_min)
        ax1.axhline(8)
        ax1.set_ylabel('Min El')
        
        ax2 = plt.subplot2grid((2,2), (0,1))
        ax2.plot(ha_range, az_max)
        ax2.set_ylabel('Max Az')
        
        ax3 = plt.subplot2grid((2,2), (1,1))
        ax3.plot(ha_range, az_min)
        ax3.set_ylabel('Min Az')
        
        ax4 = plt.subplot2grid((2,2), (1,0))
        ax4.plot(ha_range, bl_tme)
        ax4.set_ylabel('Run Time')
        plt.tight_layout()
        plt.show()
    
    # Copy middle block
    mid_ind = len(sub_blocks)//2
    master_block = sub_blocks[mid_ind].copy()
    
    # Check that all schedules are similar
    same_lengths = np.all([ len(s)==len(sub_blocks[0]) for s in sub_blocks])
    if same_lengths:
        # Extract max times
        print('Building out worst time scenario')
        tangs = Angle( [ sb['time'] for sb in sub_blocks ], unit=u.hourangle)
        tmaxs = np.max(tangs, axis=0).to_string('h', pad=True)
        master_block['time'] = tmaxs
    else:
        # Warn user if schedules are dissimilar
        print('ERROR: Could not build worst time scenario')
    
    # Time and print for user
    master_time = time_block(master_block)
    print('Master block time:', master_time)
    
    return master_block
    
def mosaic_new(radius, beamsize=0.0585):
    '''
    From input parameters of a FERMI positional uncertianty ellipse, generate
    a mosaic of beams to completely cover it. The coordinates of the mosaic
    pointing centers are returned.
    
    Args
    =======
    ra (float) - The right ascension of the FERMI source
    dec (float) - The declination of the FERMI source
    semimaj (float) - The semimajor axis of the FERMI source
    semimin (float) - The semiminor axis of the FERMI source
    angle (float) - The angle of the FERMI source, note that by default this
        is measured clockwise of North, and for our purposes we need it
        counterclockwise of East.
    
    Returns
    =======
    mosaic_coords (array) - An numpy array of coordinate points with shape
        (n,2). Note that the order of these points should amount to a fairly
        optimized slew path, alternating up and down rows.
        
    Raises
    =======
    None
    '''
    
    # Figure out how many rows/columns we need
    ver_bm = np.sqrt(3)/2 * beamsize # scaled to spacing
    nh = np.ceil(radius/beamsize)
    nv = np.ceil(radius/ver_bm)//2*2+1 # oddify
    
    # Build ranges
    rangev = np.linspace(-(nv-1)/2, (nv-1)/2, int(nv))
    
    coords = []
    for i in range(int(nv)):
        # For any row the y will be the same
        y = np.sqrt(3) * beamsize * rangev[i]
        
        # Flips center-like rows
        # Non-center-like rows shift left, add point on right
        tnh = nh
        if i%2==((nv-1)/2)%2:
            tnh = 1-nh
        rangeh = np.linspace(-tnh/2, tnh/2, int(abs(tnh)+1))
            
        # Loop through points in row
        for j in range(len(rangeh)):
            # Calculated x for each point
            x = 2 * beamsize * rangeh[j]
            
            # Record coordinate pair
            coords.append([x,y])
            
    # Arrayify
    coords = np.array(coords)
            
    # Remove any points not inside ellipse + fraction of beam radius
    fr = 0.8 # Derived -> (arccos(fr) - fr*sqrt(1-fr**2)) = 0.05*pi
    ewp = radius + fr*beamsize
    ehp = radius + fr*beamsize
    inside = (coords[:,0]/ewp)**2+(coords[:,1]/ehp)**2 <= 1
    coords_in = coords[inside]
    
    return coords_in

def rotate(sc_pnt, sc_shift):
    if sc_pnt.ndim == 0:
        sc_pnt = SkyCoord([sc_pnt.ra], [sc_pnt.dec])
    
    new_ra = np.zeros(len(sc_pnt))
    new_dec = np.zeros(len(sc_pnt))
    for i in range(len(sc_pnt)):
        pnt_ra = sc_pnt[i].ra
        pnt_dec = sc_pnt[i].dec
        
        ra_shift = sc_shift.ra
        dec_shift = sc_shift.dec
        
        v = np.array([[np.cos(pnt_dec)*np.cos(pnt_ra+ra_shift)],
                      [np.cos(pnt_dec)*np.sin(pnt_ra+ra_shift)],
                      [np.sin(pnt_dec)]])
        
        uv = np.array([[-np.sin(ra_shift)],
                      [np.cos(ra_shift)],
                      [0]])
        
        cross_u = np.array([[0, -uv[2][0], uv[1][0]],
                            [uv[2][0], 0, -uv[0][0]],
                            [-uv[1][0], uv[0][0], 0]])
        
        t1 = np.cos(dec_shift)*np.identity(3)
        t2 = -np.sin(dec_shift)*cross_u
        t3 = (1-np.cos(dec_shift))*np.outer(uv,uv)
        R = t1+t2+t3
        
        v_pr = R.dot(v)
        
        nra = np.arctan2(v_pr[1][0], v_pr[0][0])
        ndc = np.arcsin(v_pr[2][0])
        
        new_ra[i] = nra.to('deg').value
        new_dec[i] = ndc.to('deg').value
    
    new_sc = SkyCoord(new_ra*u.deg, new_dec*u.deg)
    
    if len(new_sc)==1:
        return new_sc[0]
    else:
        return new_sc

def main():
    beamsize = 1.1
    moc = mosaic_new(6, beamsize)
    scm = SkyCoord(moc[:,0]*u.deg, moc[:,1]*u.deg)
    scmr = rotate(scm, SkyCoord('0d', '90d'))
    moc = {'NCP': np.array([scmr.ra.deg, scmr.dec.deg]).T}
    
    tgt = Table({'Source_Name': ['NCP']*2, 'RAJ2000':[0]*2 *u.deg, 
                 'DEJ2000':[89.99]*2 *u.deg})
    
    
    plt.scatter(scmr.ra.deg, scmr.dec.deg)
    plt.xlabel('RA [deg]')
    plt.ylabel('DEC [deg]')
    plt.grid()
    plt.show()
    
    
    # Block list
    blocks = []
    has = []
    
    # Make block
    ha1 = -2
    bestblock1 = make_best_block(tgt, moc, haoff=ha1, wrap='CCW', verbose=False)
    blocks.append(bestblock1)
    has.append(ha1)
    
    # Write
    write_OPT(blocks, has)
    
main()