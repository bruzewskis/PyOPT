#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimEVLA.py: This module simulates the slewing motion of the EVLA for use in 
determining optimized schedules. Based largely on the equivalent code created
by NRAO SSS for use in the Observation Planning Tool. Small modifications have
been implemented to make it more scriptable.
"""

__author__ = "Seth Bruzewski"
__email__ = "bruzewskis@unm.edu"
__license__ = "GPL"

# Built in Modules
import logging

# Third Party Modules
import numpy as np
from astropy.coordinates import SkyCoord, Angle, EarthLocation
import astropy.units as u

class MotionSimulator(object):
    '''
    A pythonic class simulator of the EVLA slewing system, for implementation 
    in scheduling codes.
    '''
    
    # Generate a logger
    __logger = logging.getLogger(__name__)
    
    # Motion Ranges
    __MIN_AZ = Angle(-85.0, unit='deg')
    __MAX_AZ = Angle(445.0, unit='deg')
    __MIN_EL = Angle(8.0, unit='deg')
    __MAX_EL = Angle(125.0, unit='deg')
    
    # Velocity and acceleration rates
    __VEL_AZ = 40.0/60 * u.Unit('deg/s')
    __VEL_EL = 20.0/60 * u.Unit('deg/s')
    __ACC_AZ = 1000 * u.Unit('deg/s^2')
    __ACC_EL = 1000 * u.Unit('deg/s^2')
    
    # Setup and Settling Time
    __MIN_SETUP_TIME = 0 * u.s
    __SETTLING_TIME = 7.0 * u.s
    
    # Default currentAz/El to midpoint of their ranges.
    __currentAz = Angle(225.0, unit='deg')
    __currentEl = Angle(35.0, unit='deg')

    # These variables keep track of what the new Az/El will be set to during
    # the moveTime calculation
    __newAz = None
    __newEl = None
    
    # Keep track of error codes generated during moveTo
    __errors = []
    
    def __init__(self, startaz=None, startel=None, wrap=None):
        '''
        Creates a EvlaTelescopeMotionSimulator with an initial position of 
        `startAz` and `startEl` that is on wrap `wrap`.
        
        Args
        =======
        startaz (Angle) - Azimuth to initialize the telescope with as an
            Astropy Angle object. If not provided, telescope is initialized
            to default Azimuth, which is 225 degrees.
        startel (Angle) - Elevation to initialize the telescope with as an
            Astropy Angle object. If not provided, telescope is initialized
            to default Elevation, which is 35 degrees.
        wrap (string) - Preferred wrap of the telescope, may be CLOCKWISE,
            COUNTERCLOCKWISE, or NO_PREFERENCE
            
        Returns
        =======
        None
        
        Raises
        =======
        TypeError - Raised by class functions setCurrentAntennaAzimuth or 
            setCurrentAntennaElevation when inputs are not the correct type
        ValueError - Raised by class functions setCurrentAntennaAzimuth or 
            setCurrentAntennaElevation when inputs are outside of range
        '''
        
        self.setCurrentAntennaAzimuth(startaz, wrap)
        self.setCurrentAntennaElevation(startel)
        
    def moveTo(self, azEnd, elEnd, wrap=None, goOverTheTop=False):
        '''
        Returns the estimated move time from our current saved position to 
        `az`, `el`.  For the very first calculation, a starting position 
        due south at 65 degrees elevation is assumed.  The method takes into 
        account antenna wrap by calculating 4 different move times and 
        returning the smallest valid move time.  The 4 cases are as follows:

        1) The position we're moving to is on the COUNTERCLOCKWISE (left) 
        wrap. We simple rotate the antenna to that position and see how long 
        it takes. This is a valid option only if `wrap` is not CLOCKWISE 
        and `goOverTheTop` is false.

        2) The position is on the CLOCKWISE (right) wrap.  Again, we calculate 
        how long it will take to rotate the antenna to that position on that 
        wrap. This is a valid option only if {@code wrap} is not 
        COUNTERCLOCKWISE and `goOverTheTop` is false.

        3) We rotate the antenna 180 degrees and then go over the top to get 
        to our target position. (i.e. we're subtracting 180 degrees from the 
        AZ in case 1 above.)  This is only valid if `goOverTheTop' is 
        true and the elevation is greater than 55 degrees (180 - MAX_EL).

        4) The same as 3 above, but we add 180 to the az instead of 
        subtracting.

        This method also clears and refills the list of errors encountered 
        while attempting this move.
        
        Args
        =======
        azEnd (Angle) - Target azimuth to slew to as an Astropy Angle. Will
            be checked against telescope motion ranges.
        elEnd (Angle) - Target elevation to slew to as an Astropy Angle. Will
            be checked against telescope motion ranges.
        wrap (string) - Preferred wrap of the telescope, may be CLOCKWISE,
            COUNTERCLOCKWISE, or NO_PREFERENCE
        goOverTheTop (bool) - Whether or not to allow the telescope to 
            traverse in elevation beyond the zenith. For standard observing 
            this is not advised.
            
        Returns
        =======
        tmin (Quantity) - The total slew time expressed as an Astropy Quantity
            object with units of time. 
            
        Raises
        =======
        None
        '''
        
        # Check type of inputs
        if not isinstance(azEnd, Angle) or not isinstance(elEnd, Angle):
            self.__errors.append('IMPROPER_INPUT_TYPE')
            self.__logger.error('azEnd or elEnd are not Astropy Angles')
            return np.nan
        
        # Start up stuff
        azStart = self.__currentAz.copy()
        elStart = self.__currentEl.copy()
        
        # Keeping track of things
        self.__newAz = None
        self.__newEl = None
        self.__errors = []
        
        # Initialize min to something large
        tmin = 10 * u.day
        
        # Initialize wrap to a default if it was None
        w = 'NO_PREFERENCE' if wrap is None else wrap
        
        toAz = azStart
        toAzEnd = azEnd
        
        # Put the toAz on the left or COUNTERCLOCKWISE wrap.
        if toAz > self.__MIN_AZ + 360*u.deg: 
            toAz -= 360*u.deg
            
        # Make sure the end position is on the same wrap!  This is checked 
        # separately from the above because we aren't guaranteed that just 
        # because we need to subtract 360 from the start pos that we need to 
        # do so to the end as well.  This is mainly a concern for scans that 
        # cross the 0d/360d boundary.
        if toAzEnd > self.__MIN_AZ + 360*u.deg: 
            toAzEnd -= 360*u.deg
        
        toEl = elStart
        toElEnd = elEnd
        
        # Debug stuff
        coords = [toAz,toEl,toAzEnd,toElEnd]
        logstr = "Moving : ({0:.2f},{1:.2f}) --> ({2:.2f},{3:.2f})"
        self.__logger.debug(logstr.format(*[a.deg for a in coords]))
        
        # Whereas Longitude is always gaurunteed to be in bounds of our 
        # motion, Latitude (elevation) is not, so we have to check that here 
        # before going any further.  We only check the min, because the max a 
        # Latitude can hold is 90 and our MAX_EL is 125 or so.
        if toEl < self.__MIN_EL:
            self.__errors.append('ELEVATION_OUT_OF_RANGE')
            toEl = self.__MIN_EL.copy()
            
        # We need to do the same check for the elevation END position too!
        if toElEnd < self.__MIN_EL:
            self.__errors.append('SCAN_END_ELEVATION_OUT_OF_RANGE')
            toElEnd = self.__MIN_EL.copy()
            
        # This checks to make sure that the if the user selected a CLOCKWISE 
        # preference, but the source is in the region where the wraps do NOT 
        # overlap, we still get a valid calculation.  The reason this comes 
        # up is because for much of the logic below we're considering a wrap 
        # to be a full 360 degrees when it is only actually 265.  So, a 
        # source could be in the CLOCKWISE wrap (180 - 445 degrees) but be 
        # missed in the logic below if it's position is less than 275 degrees. 
        # (which is within the range (-85, -85 + 360))
        if w=='CLOCKWISE' and toAz > 85*u.deg:
            w = 'NO_PREFERENCE'
            
        # tmp variables for calculations
        tmpAz = toAz.copy()
        tmpAzEnd = toAzEnd.copy()
        tmpEl = toEl.copy()
        tmpElEnd = toElEnd.copy()
        
        # Counterclockwise or No Pref
        if w!='CLOCKWISE':
            # Case 3:
            if goOverTheTop:
                print('case3')
                tmpAz = toAz - 180*u.deg
                tmpAzEnd = toAzEnd - 180*u.deg
                tmpEl = 180*u.deg - toEl
                tmpElEnd = 180*u.deg - toElEnd
    
                if (tmpAz >= self.__MIN_AZ and tmpEl <= self.__MAX_EL):
                    tmin, updateBool = self.__updateMin(tmin,tmpAzEnd,tmpElEnd)
                    if updateBool:
                        self.__newAz = tmpAzEnd
                        self.__newEl = tmpElEnd
                    
                        time_str = tmin.to('s').value
                        case_str = "Case 3 min = {0:.2f} s"
                        self.__logger.debug(case_str.format(time_str))

            # Case 1:
            else:
                tmin, updateBool = self.__updateMin(tmin, tmpAzEnd, tmpElEnd)
                if updateBool:
                    self.__newAz = tmpAzEnd
                    self.__newEl = tmpElEnd
                    
                    time_str = tmin.to('s').value
                    case_str = "Case 1 min = {0:.2f} s"
                    self.__logger.debug(case_str.format(time_str))
        
        # Clockwise or No Pref
        if w!='COUNTERCLOCKWISE':
            # Case 4
            if goOverTheTop:
                print('case4')
                tmpAz = toAz + 180*u.deg
                tmpEl = 180*u.deg - toEl
                tmpAzEnd = toAzEnd + 180.0*u.deg
                tmpElEnd = 180*u.deg - toElEnd
                
                if tmpAz <= self.__MAX_AZ and tmpEl <= self.__MAX_EL:
                    tmin, updateBool = self.__updateMin(tmin,tmpAzEnd,tmpElEnd)
                    if updateBool:
                        self.__newAz = tmpAzEnd
                        self.__newEl = tmpElEnd
                    
                        time_str = tmin.to('s').value
                        case_str = "Case 4 min = {0:.2f} s"
                        self.__logger.debug(case_str.format(time_str))
            
            # Case 2
            else:
                tmpAz = toAz + 360*u.deg
                tmpAzEnd = toAzEnd + 360*u.deg
                tmpEl = toEl.copy()
                tmpElEnd = toElEnd.copy()
                if tmpAz < self.__MAX_AZ:
                    tmin, updateBool = self.__updateMin(tmin,tmpAzEnd,tmpElEnd)
                    if updateBool:
                        self.__newAz = tmpAzEnd
                        self.__newEl = tmpElEnd
                    
                        time_str = tmin.to('s').value
                        case_str = "Case 2 min = {0:.2f} s"
                        self.__logger.debug(case_str.format(time_str))
                        
        if self.__newAz is None or self.__newEl is None:
            self.__errors.append('MOVE_NOT_POSSIBLE')
        else:
            # Azimuth checks
            if self.__MIN_AZ > self.__newAz:
                self.__logger.error("newAz < min: " + self.__newAz.to_string())
                self.__errors.append('SCAN_END_AZIMUTH_OUT_OF_RANGE')
                self.__currentAz = self.__MIN_AZ.copy()
            elif self.__MAX_AZ < self.__newAz:
                self.__logger.error("newAz > max: " + self.__newAz.to_string())
                self.__errors.append('SCAN_END_AZIMUTH_OUT_OF_RANGE')
                self.__currentAz = self.__MAX_AZ.copy()
            else:
                self.__currentAz = self.__newAz
                
            # Elevation checks
            if self.__MIN_EL > self.__newEl:
                self.__logger.error("newEl < min: " + self.__newEl.to_string())
                self.__errors.append('SCAN_END_ELEVATION_OUT_OF_RANGE')
                self.__currentEl = self.__MIN_EL.copy()
            elif self.__MAX_EL < self.__newEl:
                self.__logger.error("newEl > max: " + self.__newEl.to_string())
                self.__errors.append('SCAN_END_ELEVATION_OUT_OF_RANGE')
                self.__currentEl = self.__MAX_EL.copy()
            else:
                self.__currentEl = self.__newEl
            
            # Add any settling time necessary.
            tmin += self.__SETTLING_TIME
            
            # If the slew took less time than the min. time it takes to 
            # prepare the antenna for observing, use that min. instead.
            if tmin < self.__MIN_SETUP_TIME:
                tmin = self.__MIN_SETUP_TIME.copy()
        
        # Decompose units just to be safe
        tmin = tmin.decompose()
            
        return tmin
        
    def getErrors(self):
        '''
        Returns a list of any errrors that may have occured during the lastest 
        call to the moveTo method. These errors are stored as simple strings
        for the user to parse or use as they see fit.
        
        Args
        =======
        None
        
        Returns
        =======
        self.__errors.copy() (list) - List of errors generated by moveTo
        
        Raises
        =======
        None
        '''
        
        return self.__errors.copy()
    
    def getCurrentAntennaWrap(self):
        '''
        Returns which AntennaWrap this simulator is currently on using the 
        following rules.  If the currentAz is greater than 275 degrees (MIN_AZ 
        + 360) then we're in the CLOCKWISE (right) wrap.  Otherwise we're in 
        the COUNTERCLOCKWISE (left) wrap.  Note: This could be changed to split 
        the range in half: greater than 180 degrees is CLOCKWISE, the rest is 
        COUNTERCLOCKWISE.  As it is now, you are only in the CLOCKWISE wrap if 
        you are in the overlapping region.
        
        Args
        =======
        None
        
        Returns
        =======
        wrap (string) - The current antenna wrap expressed as a string.
        
        Raises
        =======
        None
        '''
        
        if self.__currentAz > 180*u.deg:
            return 'CLOCKWISE' #AntennaWrap.CLOCKWISE
        else:
            return 'COUNTERCLOCKWISE' #AntennaWrap.COUNTERCLOCKWISE
        
    def getCurrentAntennaAzimuth(self):
        '''
        Returns a copy of the current antenna azimuth. This value should be
        in the range -85 to 445 degrees.
        
        Args
        =======
        None
        
        Returns
        =======
        self.__currentAz.copy() (Angle) - The current antenna azimuth
            expressed as an Astropy Angle object.
        
        Raises
        =======
        None
        '''
        
        return self.__currentAz.copy()
        
    def setCurrentAntennaAzimuth(self, a, w=None):
        '''
        Sets the current Antenna Az to a clone of Angle equivalent to 
        `a` at wrap `w` if `a` is within range.
        
        Args
        =======
        None
        
        Returns
        =======
        self.__currentAz.copy() (Angle) - The current antenna azimuth
            expressed as an Astropy Angle object.
        
        Raises
        =======
        TypeError - Raised when inputs are not the correct type
        ValueError - Raised when inputs are outside of range
        '''
        
        # Check input type
        if not a is None:
            if not isinstance(a, Angle):
                raise TypeError('Initializing Azimuth is not Astropy Angle')
        
        # Check that input is in range
        if w is None:
            if a is not None and self.__MIN_AZ < a < self.__MAX_AZ:
                self.__currentAz = a.copy()
            elif a is None:
                pass
            else:
                raise ValueError('Invalid Antenna Azimuth: '+str(a.deg))
        else:
            self.setCurrentAntennaAzimuth(self.toAntennaAzimuth(a, w))
    
    def getCurrentAntennaElevation(self):
        '''
        Returns a copy of the current antenna elevation. This value should be
        in the range 8 to 125 degrees.
        
        Args
        =======
        None
        
        Returns
        =======
        self.__currentEl.copy() (Angle) - The current antenna elevation
            expressed as an Astropy Angle object.
        
        Raises
        =======
        None
        '''
        
        return self.__currentEl.copy()
        
    def setCurrentAntennaElevation(self, a):
        '''
        Sets the current Antenna El to a clone of `a` if a is within 
        range.
        
        Args
        =======
        None
        
        Returns
        =======
        self.__currentEl.copy() (Angle) - The current antenna azimuth
            expressed as an Astropy Angle object.
        
        Raises
        =======
        TypeError - Raised when inputs are not the correct type
        ValueError - Raised when inputs are outside of range
        '''
        
        # Check input type
        if not a is None:
            if not isinstance(a, Angle):
                raise TypeError('Initializing Elevation is not Astropy Angle')
                
        # Check that input is in range
        if a is not None and self.__MIN_EL < a < self.__MAX_EL:
            self.__currentEl = a.copy()
        elif a is None:
            pass
        else:
            raise ValueError('Invalid Antenna Elevation: '+str(a.deg))
    
    def toAntennaAzimuth(self, az, w):
        '''
        Returns an Angle in degrees between -85 and 445 degrees that 
        represents `az` at AntennaWrap `w`.
        
        Args
        =======
        az (Angle) - Initial azimuth to be converted
        w (string) - Current wrap of the antenna
        
        Returns
        =======
        antennaAz (Angle) - Proper azimuth inside the antenna wrap
        
        Raises
        =======
        None
        '''
        
        antennaAz = None
        
        if az is not None:
            # az is 0 - 360 degrees
            if w=='CLOCKWISE':
                # 180 to 445 degrees
                if az < self.__MAX_AZ-360*u.deg:
                    az += 360*u.deg
            elif w=='COUNTERCLOCKWISE':
                # -85 to 180 degrees
                if az > self.__MIN_AZ+360*u.deg:
                    az -= 360*u.deg
                
            antennaAz = az
        
        return antennaAz
    
    def getAzimuthMinimum(self):
        '''
        Returns the minimum azimuth value for EVLA antenna pointings.
        
        Args
        =======
        None
        
        Returns
        =======
        self.__MIN_AZ.copy (Angle) - Telescope minimum azmiuth
        
        Raises
        =======
        None
        '''
        
        return self.__MIN_AZ.copy()
    
    def getAzimuthMaximum(self):
        '''
        Returns the maximum azimuth value for EVLA antenna pointings.
        
        Args
        =======
        None
        
        Returns
        =======
        self.__MAX_AZ.copy (Angle) - Telescope maximum azmiuth
        
        Raises
        =======
        None
        '''
        return self.__MAX_AZ.copy()
    
    def getAzimuthDefault(self):
        '''
        Returns the default azimuth value for EVLA antenna pointings.
        
        Args
        =======
        None
        
        Returns
        =======
        DEF_AZ (Angle) - Telescope default azimuth
        
        Raises
        =======
        None
        '''
        return Angle(225, unit='deg')
    
    def getElevationMinimum(self):
        '''
        Returns the minimum elevation value for EVLA antenna pointings.
        
        Args
        =======
        None
        
        Returns
        =======
        self.__MIN_EL.copy (Angle) - Telescope minimum elevation
        
        Raises
        =======
        None
        '''
        return self.__MIN_EL.copy()
    
    def getElevationMaximum(self):
        '''
        Returns the maximum elevation value for EVLA antenna pointings.
        
        Args
        =======
        None
        
        Returns
        =======
        self.__MAX_AZ.copy (Angle) - Telescope maximum elevation
        
        Raises
        =======
        None
        '''
        return self.__MIN_EL.copy()
    
    def getElevationDefault(self):
        '''
        Returns the default elevation value for EVLA antenna pointings.
        
        Args
        =======
        None
        
        Returns
        =======
        DEF_EL (Angle) - Telescope default elevation
        
        Raises
        =======
        None
        '''
        return Angle(35, unit='deg')
    
    def __updateMin(self, currentMin, az, el):
        '''
        This method updates the passed in `currentMin` variable with 
        the minimum value of 'tmin' in the moveTo function, the time it takes 
        to move from currentAz to az, and the time it takes to move from 
        currentEl to el. If the currentMin is changed, we return true.
        
        Args
        =======
        currentMin (Quantity) - The current slew time minimum expressed as an
            Astropy Quantity object with units of time.
        az (Angle) - Target azimuth to slew to.
        el (Angle) - Target elevation to slew to.
        
        Returns
        =======
        newCurrentMin (Quantity) - The new current slew time minimum expressed 
            as an Astropy Quantity object with units of time.
        update (bool) - Whether or not `tmin` will have been changed
        
        Raises
        =======
        None
        '''
        taz = self.__calcAzMoveTime(self.__currentAz, az)
        tel = self.__calcElMoveTime(self.__currentEl, el)
        
        # t is the larger of taz and tel
        t = max([taz,tel])
        
        if currentMin > t:
            return t, True
        else:
            return currentMin, False
    
    def __calcAzMoveTime(self, f, t):
        '''
        Calculates the move time from angle `f` to angle `t` in Azimuth.
        
        Args
        =======
        f (Angle) - Initial azimuth angle
        t (Angle) - Final azimuth angle
        
        Returns
        =======
        telSlewAz (Quantity) - Slew time, accounting for factors like the
            telescope slew acceleration, expressed as an Astropy Quantity
            with units of time.
        
        Raises
        =======
        None
        '''
        azd = abs(f-t)
        
        # Time it takes to reach full speed
        # Accounts for both acceleration & decceleration.
        timeAccAz = 2 * self.__VEL_AZ / self.__ACC_AZ
        
        # Distance it takes to reach full speed
        distAccAz = (self.__VEL_AZ * self.__VEL_AZ) / self.__ACC_AZ
        
        # If the antenna never reaches full speed, use this equation.
        if azd < distAccAz:
            telSlewAz = 2 * np.sqrt(azd / self.__ACC_AZ)
        # otherwise, use this equation.
        else:
            telSlewAz = timeAccAz + (azd - distAccAz) / self.__VEL_AZ

        return telSlewAz
    
    def __calcElMoveTime(self, f, t):
        '''
        Calculates the move time from angle `f` to angle `t` in Elevation.
        
        Args
        =======
        f (Angle) - Initial elevation angle
        t (Angle) - Final elevation angle
        
        Returns
        =======
        telSlewEl (Quantity) - Slew time, accounting for factors like the
            telescope slew acceleration, expressed as an Astropy Quantity
            with units of time.
        
        Raises
        =======
        None
        '''
        eld = abs(f-t)
        
        # Time it takes to reach full speed
        # Accounts for both acceleration & decceleration.
        timeAccEl = 2 * self.__VEL_EL / self.__ACC_EL

        # Distance it takes to reach full speed
        distAccEl = (self.__VEL_EL * self.__VEL_EL) / self.__ACC_EL

        
        if eld < distAccEl:
            # If the antenna never reaches full speed, use this equation.
            telSlewEl = 2 * np.sqrt(eld / self.__ACC_EL)
        else:
            # otherwise, use this equation.
            telSlewEl = timeAccEl + (eld - distAccEl) / self.__VEL_EL
            
        return telSlewEl
    
def EquitorialToHorizontal(ra, dec, lst):
    '''
    This function converts Equitorial coordinates (RA/DEC) to a horizontal
    coordinate system local to the VLA (AZ/EL). Ideally utilizes astropy 
    Angles, but will convert where possible. Also requires a specific time 
    given in LST. Note that Azimuth here is measured clockwise from the North.
    
    Args
    =======
    ra (various) - Either an Astropy Angle class, or a string/number in
        hourangle units. The latter will be converted to an Astropy Angle.
    dec (various) - Either an Astropy Angle class, or a string/number in
        degree units. The latter will be converted to an Astropy Angle.
    lst (string) - The local sidereal time at the VLA to be converted for,
        ideally in format HH:MM:SS.SS. This will be converted into hourangle.
    
    Returns
    =======
    Az (Angle) - The resulting Azimuth generated from the inputs. Measured
        clockwise from due North.
    Alt (Angle) - The resulting Altitude/Elevation generated from the inputs.
        Measured from the horizon.
        
    Raises
    =======
    UnitTypeError - Raised if the inputs cannot be coerced into Astropy Angles
    '''
    
    # Check type of inputs for safety
    if not isinstance(ra, Angle):
        ra = Angle(ra, unit=u.hourangle)
    if not isinstance(dec, Angle):
        dec = Angle(dec, unit=u.deg)
    if not isinstance(lst, Angle):
        dec = Angle(lst, unit=u.hourangle)
    
    # Let's just assume we're at the VLA
    vla = EarthLocation.of_site('VLA')
    ha = lst - ra
    lat = vla.lat
    
    # Do Trigonometry
    cosAlt_sinAz = -np.cos(dec)*np.sin(ha)
    cosAlt_cosAz = np.sin(dec)*np.cos(lat) - np.cos(dec)*np.cos(ha)*np.sin(lat)
    sinAlt = np.sin(dec)*np.sin(lat) + np.cos(dec)*np.cos(ha)*np.cos(lat)
    
    Az = Angle(np.arctan2(cosAlt_sinAz,cosAlt_cosAz)).wrap_at('360d')
    Alt = Angle(np.arcsin(sinAlt))
    
    return Az, Alt

def PointToPointTime(start_sc, end_sc, lst):
    '''
    Calculate the time to slew between two coordinates on the sky. Note that
    this function will not check if the slew is possible, instead allowing
    MotionSimulator to throw errors where it likes. This is largely useful
    for diagnostic checking.
    
    Args
    =======
    start_sc (SkyCoord) - The coordinates of the first point on the sky, where
        the telescope will begin its slew. Must be an Astropy SkyCoord object
    end_sc (SkyCoord) - The coordinates of the second point on the sky, where
        the telescope will end its slew. Must be an Astropy SkyCoord object
    lst (string) - The local sidereal time at the VLA to be converted for,
        ideally in format HH:MM:SS.SS. This will be converted into hourangle.
        
    Returns
    =======
    slewtime (Quantity) - The time it takes to slew between the input points.
        This is returned as an Astropy Quantity with units of time, that way it
        can be easily worked with.
        
    Raises
    =======
    TypeError - Raised when start_sc or end_sc are not SkyCoords
    '''
    # Check type of inputs
    if not isinstance(start_sc, SkyCoord):
        raise TypeError('param \'start_sc\' is not a Astropy SkyCoord')
    if not isinstance(end_sc, SkyCoord):
        raise TypeError('param \'end_sc\' is not a Astropy SkyCoord')
    
    # Basic logging
    mlogger = logging.getLogger(__name__)
    
    # Translate Start Point
    start_az, start_el = EquitorialToHorizontal(start_sc.ra,start_sc.dec,lst)
    
    # Translate End Point
    end_az, end_el = EquitorialToHorizontal(end_sc.ra,end_sc.dec,lst)
    
    # Nice Debug
    coo_fmt = '(ra:{0:.2f},dec:{1:.2f}) --> (az:{2:.2f},el:{3:.2f})'
    coo1 = [start_sc.ra.deg, start_sc.dec.deg, start_az.deg, start_el.deg]
    coo2 = [end_sc.ra.deg, end_sc.dec.deg, end_az.deg, end_el.deg]
    mlogger.debug('Converting : '+coo_fmt.format(*coo1))
    mlogger.debug('Converting : '+coo_fmt.format(*coo2))    
    
    # Generate Simulator
    sim = MotionSimulator(start_az, start_el)
    slewtime = sim.moveTo(end_az, end_el)
    mlogger.debug('Took {0:.2f} seconds to slew'.format(slewtime.value))
    
    # Errors
    err = sim.getErrors()
    mlogger.debug('Found {:d} errors'.format(len(err)))
    for e in err:
        mlogger.error(e)
        
    return slewtime
    
    
if __name__=='__main__':
    '''
    Psuedomain function for simple tests to give some idea how this works
    
    Args
    =======
    None
    
    Returns
    =======
    None
    
    Raises
    =======
    None
    '''
    
    # Test 1
    print('Test 1:')
    s1 = SkyCoord(21.915*u.hourangle, -30*u.deg)
    s2 = SkyCoord(21.915*u.hourangle, 30*u.deg)
    sc2str = lambda sc : sc.to_string('hmsdms')
    print('Slewing from {0:s} to {1:s}'.format(sc2str(s1), sc2str(s2)))
    
    lst0 = Angle('21:54:55', unit='hourangle')
    st = PointToPointTime(s1, s2, lst0)
    st_sec = st.to('s').value
    st_dst = s1.separation(s2).deg
    fmt_str = 'Took {0:.2f} seconds to slew {1:.2f} degrees'
    print(fmt_str.format(st_sec, st_dst))
    
    # Test 2
    print('\nTest 2:')
    sim = MotionSimulator()
    totaltime = 0*u.s
    for _ in range(10):
        new_az = Angle(np.random.uniform(0,360) * u.deg)
        new_el = Angle(np.random.uniform(10,90) * u.deg)
        slewt = sim.moveTo(new_az, new_el)
        totaltime += slewt
    
    t2_st = totaltime.to('s').value
    print('Slewed to 10 points over {0:.2f} seconds'.format(t2_st))