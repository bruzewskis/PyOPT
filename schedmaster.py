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
from mosaicer import mosaic, make_mosaic_friendly, diagnostic_plot
from findmissed import find_targets
from optscheduler import write_OPT

def checkflags(bitflag, goodflags, verbose=False):
    '''
    Determines whether flags on a particular source are acceptable based on
    the input list of flags the user is okay with. Note that `bitflag` is a
    base-10 number, which when represented as a 12-bit base-2 number
    represents the active flags on a particular source. This is then compared
    to a binary representation of the allowed flags via a bitwise and. If the
    output of this operations is 000000000000, we return True, as all flags
    are acceptable
    
    Args
    =======
    bitflag (int) - The base-10 representation of the flags. As stated in the
        FERMI documentation, this can be thought of as sum(2^(flag_n-1)), so
        if a source has flags 3 and 8, then bitflag=2^(3-1)+2^(8-1)=132.
    goodflags (list) - The list of acceptable flags. For example, if flags 3 
        and 8 are allowed, then this input should simply be [8,3]. Note that 
        the order does not matter.
    verbose (bool) - For diagnostic purposes, whether to display the bitwise
        operation that is occuring.
    
    Returns
    =======
    is_allowed (bool) - A boolean determining if the flags on a particular
        source match those provided in goodflags.
    
    Raises
    =======
    None
    '''
    
    # Create gate number
    checkb = (2**12-1) ^ np.sum(np.power(2, np.array(goodflags)-1))
    
    # Turn off appropriate flags
    output = bitflag & checkb
    
    # Show operation if requested
    if verbose:
        print(np.binary_repr(bitflag,12), '=', bitflag)
        print(np.binary_repr(checkb,12), '=', checkb)
        print('-'*12)
        print(np.binary_repr(output,12), '=', output)
        
    is_allowed = output==0
    
    return is_allowed

def filter_data(data, keepflags, do_assoc=True, do_dec=True, do_nan=True):
    '''
    Filters the data around certian criteria to narrow our search to unassoc
    sources that we can observe with the VLA. Further, we filter for only 
    certian flags in the FERMI data, picking out ideal candidates.
    
    Args
    =======
    data (Table) - The raw fermi data table
    keepflags (list<int>) - A list of integers representing the allowed flags
    do_assoc (bool) - Whether to filter for only unassociated sources
    do_dec (bool) - Whether to filter for VLA observable declinations
    do_nan (bool) - Whether to filter out sources that have NaN values
    
    Returns
    =======
    filtered (Table) - A subset of the data input table which has only sources
        matching all input criteria
    
    Raises
    =======
    ValueError - Raised if keepflags is not properly specified
    '''
    
    # Keep track of filters
    filters = []
    
    # Only keep certian flags
    if isinstance(keepflags, list) and len(keepflags)>0:
        has_good_flags = checkflags(data['Flags'], keepflags)
        filters.append(has_good_flags)
    elif len(keepflags)==0:
        raise ValueError('keepflags=[] implies you want no data')
    else:
        raise ValueError('keepflags not properly specified')
        
    # We only care about unassociated sources
    if do_assoc:
        unassoc1 = np.array([ st.strip() == '' for st in data['ASSOC1'] ])
        unassoc2 = np.array([ st.strip() == '' for st in data['ASSOC2'] ])
        unassoc = np.logical_and(unassoc1, unassoc2)
        filters.append(unassoc)
    
    # Sources in VLA Sky
    if do_dec:
        above40 = data['DEJ2000'] > -40
        filters.append(above40)
    
    # Keep rows that aren't all NaN
    if do_nan:
        nonnan = ~np.isnan(data['Conf_95_SemiMajor'])
        filters.append(nonnan)
    
    # Apply filters
    filt = np.all(np.stack(filters), axis=0)
    filtered = data[filt]
    
    return filtered

def narrow_targets(fdata, vdata, beam, missed_frac=0.5, mosaic_cutoff=7, 
                   writename=None, verbose=False):
    '''
    From input tables, generates a list of targets for followup based on
    certian hard-coded critera, namely the fraction of the missed area and
    the number of pointings needed to mosaic the object. 
    '''
    
    # Find missed areas
    missed_areas = find_targets(fdata, vdata)
    
    # How much of the area was missed
    smaj = fdata['Conf_95_SemiMajor'].quantity
    smin = fdata['Conf_95_SemiMinor'].quantity
    fdata['MissedFrac'] = (missed_areas/(np.pi*smaj*smin)).value
    
    # Generate mosaics, linked to names
    friendly = make_mosaic_friendly(fdata)
    mo_coords = {}
    numtomosaic = []
    for i in range(len(fdata)):
        coords = mosaic(beamsize=beam, **friendly[i])
        mo_coords[fdata['Source_Name'][i]] = coords
        numtomosaic.append(len(coords))
    fdata['NumToMosaic'] = numtomosaic
    if verbose:
        print('Mosaiced all the sources')
    
    # Pretty example
    ri = np.random.randint(0,len(fdata))
    need = fdata['NumToMosaic'][ri]
    name = fdata['Source_Name'][ri]
    if verbose:
        print('Here\'s an example ('+name+') that needs', need, 'pointing(s):')
        diagnostic_plot(friendly[ri], mo_coords[fdata['Source_Name'][ri]],beam)
    
    # Filter down
    check = np.logical_and(fdata['MissedFrac']>missed_frac, 
                           fdata['NumToMosaic']<mosaic_cutoff)
    targets = fdata[check]
    iden_str = 'Identified {0} candidates (missed>{1}, tiles<{2})'
    if verbose:
        print(iden_str.format(len(targets), missed_frac, mosaic_cutoff))
    
    # Write those out
    if writename is not None:
        targets.write(writename, overwrite=True)
        print('Wrote candidate list to', writename)
    
    return targets, mo_coords

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

def schedule_block(block, mosaics, calibrators, source_scan=35, cal_scan=35, 
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
    COHERENCE_TIME = 30*u.min
    
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
    last_cal = -1*COHERENCE_TIME # Make sure we get cal at start
    
    ############### DETERMINE BEST CALIBRATOR ###############
    # Find best available flux calibrator
    flux_cal_names = ['0521+166=3C138', '0542+498=3C147', '1331+305=3C286']
    flux_cals = SkyCoord(['05h21m09.886021s  16d38\'22.051220"', # 3C138
                          '05h42m36.137916s  49d51\'07.233560"', # 3C147
                          '13h31m08.287984s  30d30\'32.958850"']) # 3C286
    flux_az, flux_el = E2H(flux_cals.ra, flux_cals.dec, lst)
    best_ind = np.argmax(flux_el)
    
    # Values for best flux cal
    best_flux_name = flux_cal_names[best_ind]
    best_flux_sc = flux_cals[best_ind]
    
    # What is the worst possible slew we might have to do
    worst_slew = worst_setup_slew(flux_az[best_ind], flux_el[best_ind])
    setup_atten = 1*u.min
    setup_req = 30*u.s
    scan_tgt = 2*u.min
    slew_target = P2P(best_flux_sc, tgt0, lst0)
    
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
                  'resourceName': 'C32f2',
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
                'resourceName': 'C32f2',
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
                     'resourceName': 'C32f2',
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
    old_cal_arr, old_cname = None, None
    for i in range(len(block)):
        # Grab mosaic coordinates
        fname = block['Source_Name'][i]
        mosaic = mosaics[fname]
        
        # Generate some names and notes
        b26 = lambda n : chr(97+n//26) + chr(97+n%26)
        names = [ fname[5:-1]+b26(n) for n in range(len(mosaic)) ]
        intents = ['ObsTgt']*len(mosaic)
        dur = [source_scan]*len(mosaic)
        
        # If its been a while, then we should calibrate
        if total_time - last_cal > COHERENCE_TIME/2:
            
            # Find nearest calibrator to next source
            next_src = SkyCoord(mosaic[0][0]*u.deg, mosaic[0][1]*u.deg)
            csc, cname = find_nearest_cal(next_src, calibrators, config)
            cal_arr = np.array([[csc.ra.deg, csc.dec.deg]])
            
            # Prepend calibrator
            mosaic = np.concatenate((cal_arr, mosaic), axis=0)
            names.insert(0, cname)
            intents.insert(0, 'CalGain')
            dur.insert(0, cal_scan)
            
            if i!=0:
                mosaic = np.concatenate((old_cal_arr, mosaic), axis=0)
                names.insert(0, old_cname)
                intents.insert(0, 'CalGain')
                dur.insert(0, cal_scan)
                
            old_cal_arr = cal_arr
            old_cname = cname
            
            # Record that we calibrated
            last_cal = total_time.copy()
        
        for j in range(len(mosaic)):
            # Move to a new spot on sky
            scj = SkyCoord(mosaic[j][0]*u.deg, mosaic[j][1]*u.deg)
            
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
                           'resourceName': 'C32f2',
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
    
def make_best_block(block, mosaics, calibrators, haoff, wrap, verbose, 
                    doplots=True):
    
    dummy = schedule_block(block, mosaics, calibrators, ha_offset=haoff, 
                           wrap_pref=wrap, verbose=True)
    
    # Figure out what schedules to test
    width = 10/60
    steps = 13
    ha_range = np.linspace(haoff-width, haoff+width, steps)
    
    # Generate schedules
    sub_blocks = []
    for ha in ha_range:
        sblock = schedule_block(block, mosaics, calibrators, ha_offset=ha, 
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
    
def main():
    
    # 4FGL Data
    fermi = Table.read('inputs/FERMI_4FGL.fit', format='fits', hdu=1)
    print('Read in', len(fermi), 'FERMI sources')
    
    # Filter for flags
    fermi = filter_data(fermi, [12,10,1])
    print('Found', len(fermi), 'sources with good flags')
    
    # VLA Data
    vla = Table.read('inputs/pointings.csv', format='csv')
    print('Read in', len(vla), 'VLA pointings')
    
    # VLA Beamsize (C-band, EVLA Memo 154)
    beamsize = (21.07/6)/60 # deg
    
    # VLA Max El
    vla_lat = 34.0784 # deg
    min_dec = -30
    
    # Load in calibrators
    cal_file = 'inputs/calcat_C.fits'
    cals = Table.read(cal_file)
    northern_cals = cals[Angle(cals['dec']).deg>vla_lat]
    southern_cals = cals[Angle(cals['dec']).deg<vla_lat]
    print('Found', len(northern_cals), 'calibrators in the North')
    print('Found', len(southern_cals), 'calibrators in the South')

    # Get target list and pointings for each source
    tgt, moc = narrow_targets(fermi, vla, beamsize, missed_frac=0.5, 
                              mosaic_cutoff=5)
    
    # Block list
    blocks = []
    has = []
    
    # SouthWest
    ha1 = -0.5
    block1 = make_block(tgt, ramin=-150, ramax=0, dcmin=min_dec, dcmax=vla_lat)
    bestblock1 = make_best_block(block1, moc, southern_cals, ha1, 'CCW', False)
    blocks.append(bestblock1)
    has.append(ha1)
    
    # SouthEast 1
    ha2 = 1.25
    block2 = make_block(tgt, ramin=0, ramax=120, dcmin=min_dec, dcmax=vla_lat)
    bestblock2 = make_best_block(block2, moc, southern_cals, ha2, 'CCW', False)
    blocks.append(bestblock2)
    has.append(ha2)
    
    # SouthEast 2
    ha3 = 1
    block3 = make_block(tgt, ramin=120, ramax=210, dcmin=min_dec, dcmax=vla_lat)
    bestblock3 = make_best_block(block3, moc, southern_cals, ha3, 'CCW', False)
    blocks.append(bestblock3)
    has.append(ha3)
    
    # NorthEast
    ha4 = 4
    block4 = make_block(tgt, ramin=0, ramax=180, dcmin=vla_lat, dcmax=90)
    bestblock4 = make_best_block(block4, moc, northern_cals, ha4, '', False)
    blocks.append(bestblock4)
    has.append(ha4)
    
    # NorthWest
    ha5 = 4.5
    block5 = make_block(tgt, ramin=180, ramax=360, dcmin=vla_lat, dcmax=90)
    bestblock5 = make_best_block(block5, moc, northern_cals, ha5, '', False)
    blocks.append(bestblock5)
    has.append(ha5)
    
    # Organize
    total_time = 0
    for block in blocks:
        total_time += time_block(block)
    print(total_time)
    
    # Write
    write_OPT(blocks, has)
    
main()