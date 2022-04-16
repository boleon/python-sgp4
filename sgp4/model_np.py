"""The Satellite class."""

from sgp4.earth_gravity import wgs72old, wgs72, wgs84, pz90
from sgp4.io import twoline_parse, day_year_2_jdsatepoch
from sgp4.propagation_np import sgp4deploy, sgp4_np
from sgp4.model import Satrec
import numpy as np

WGS72OLD = 0
WGS72 = 1
WGS84 = 2
PZ90  = 3
gravity_constants = wgs72old, wgs72, wgs84, pz90  # indexed using enum values above
minutes_per_day = 1440.

class Satrec_np(Satrec):
    """Slow Python-only version of the satellite object."""

    # Approximate the behavior of the C-accelerated class by locking
    # down attribute access, to avoid folks accidentally writing code
    # against this class and adding extra attributes, then moving to a
    # computer where the C-accelerated class is used and having their
    # code suddenly produce errors.
    
    def __init__(self) :
        self.satnum = []
        self.classification = []
        self.intldesg = []
        self.epochdays = []
        self.ndot = []
        self.nddot = []
        self.bstar = []
        self.ephtype = []
        self.elnum = []
        
        self.inclo = []
        self.nodeo = []
        self.ecco = []
        self.argpo = []
        self.mo = []
        self.no_kozai = []
        self.revnum = []
        
        self.epochyr = []
        self.jdsatepoch = []
        self.jdsatepochF = []
        self.epoch = []
        
        self.tumin = []
        self.mu = []
        self.radiusearthkm = []
        self.xke = []
        self.j2 = []
        self.j3 = []
        self.j4 = []
        self.j3oj2 = []
        
        self.operationmode = []
        
    def extend_sats(self, N ) :
        self.satnum *= N
        self.classification *= N
        self.intldesg *= N
        self.epochdays *= N
        self.ndot *= N
        self.nddot *= N
        self.bstar *= N
        self.ephtype *= N
        self.elnum *= N
        
        self.inclo *= N
        self.nodeo *= N
        self.ecco *= N
        self.argpo *= N
        self.mo *= N
        self.no_kozai *= N
        self.revnum *= N
        
        self.epochyr *= N
        self.jdsatepoch *= N
        self.jdsatepochF *= N
        self.epoch *= N
        
        self.tumin *= N
        self.mu *= N
        self.radiusearthkm *= N
        self.xke *= N
        self.j2 *= N
        self.j3 *= N
        self.j4 *= N
        self.j3oj2 *= N
        
        self.operationmode *= N
        
    def init_to_numpy(self) :
        self.satnum     = np.asarray(self.satnum)
        self.classification = np.asarray(self.classification)
        self.intldesg   = np.asarray(self.intldesg)
        self.epochdays  = np.asarray(self.epochdays)
        self.ndot       = np.asarray(self.ndot)
        self.nddot      = np.asarray(self.nddot)
        self.bstar      = np.asarray(self.bstar)
        self.ephtype    = np.asarray(self.ephtype)
        self.elnum      = np.asarray(self.elnum)
        
        self.inclo      = np.asarray(self.inclo)
        self.nodeo      = np.asarray(self.nodeo)
        self.ecco       = np.asarray(self.ecco)
        self.argpo      = np.asarray(self.argpo)
        self.mo         = np.asarray(self.mo)
        self.no_kozai   = np.asarray(self.no_kozai)
        self.revnum     = np.asarray(self.revnum)
        
        self.epochyr    = np.asarray(self.epochyr)
        self.jdsatepoch = np.asarray(self.jdsatepoch)
        self.jdsatepochF= np.asarray(self.jdsatepochF)
        self.epoch      = np.asarray(self.epoch)
        
        self.tumin  = np.asarray(self.tumin)
        self.mu     = np.asarray(self.mu)
        self.radiusearthkm = np.asarray(self.radiusearthkm)
        self.xke    = np.asarray(self.xke)
        self.j2     = np.asarray(self.j2)
        self.j3     = np.asarray(self.j3)
        self.j4     = np.asarray(self.j4)
        self.j3oj2  = np.asarray(self.j3oj2)
        
        self.operationmode = np.asarray(self.operationmode)
    
    def add_sat_lines( self, ll, N=None ) :
        ( satnum, classification, intldesg, epochdays, ndot, nddot, bstar,
        ephtype, elnum, inclo, nodeo, ecco, argpo, mo, no_kozai, revnum, 
        epochyr, jdsatepoch, jdsatepochF, epoch, whichconst, operationmode ) = ll
        
        self.satnum.append(satnum)
        self.classification.append(classification)
        self.intldesg.append(intldesg)
        self.epochdays.append(epochdays)
        self.ndot.append(ndot)
        self.nddot.append(nddot)
        self.bstar.append(bstar)
        self.ephtype.append(ephtype)
        self.elnum.append(elnum)
        
        self.inclo.append(inclo)
        self.nodeo.append(nodeo)
        self.ecco.append(ecco)
        self.argpo.append(argpo)
        self.mo.append(mo)
        self.no_kozai.append(no_kozai)
        self.revnum.append(revnum)
    
        self.epochyr.append(epochyr)
        self.jdsatepoch.append(jdsatepoch)
        self.jdsatepochF.append(jdsatepochF)
        self.epoch.append(epoch)
        self.tumin.append(whichconst.tumin)
        self.mu.append(whichconst.mu)
        self.radiusearthkm.append(whichconst.radiusearthkm)
        self.xke.append(whichconst.xke)
        self.j2.append(whichconst.j2)
        self.j3.append(whichconst.j3)
        self.j4.append(whichconst.j4)
        self.j3oj2.append(whichconst.j3oj2)
        self.operationmode.append(operationmode)
    
    @classmethod
    def sgp4init(cls, twolines, whichconst=WGS72) :
        whichconst = gravity_constants[whichconst]
        self = cls()
        for line1, line2 in twolines :
            self.add_sat_lines((*twoline_parse(line1, line2), whichconst, 'i'))
            
        self.init_to_numpy()
            
        sgp4deploy( self )

        # # Install a fancy split JD of the kind the C++ natively supports.
        # # We rebuild it from the TLE year and day to maintain precision.
        # year = self.epochyr
        # days, fraction = divmod(self.epochdays, 1.0)
        # self.jdsatepoch = year * 365 + (year - 1) // 4 + days + 1721044.5
        # self.jdsatepochF = round(fraction, 8)  # exact number of digits in TLE

        # # Remove the legacy datetime "epoch", which is not provided by
        # # the C++ version of the object.
        # del self.epoch

        # # Undo my non-standard 4-digit year
        # self.epochyr %= 100
        return self
    
    @classmethod
    def sgp4init_fromfile(cls, filename) :
        twolines = ( lines[1:] for lines in Satrec_np.tle_iter( filename ) )
        return Satrec_np.sgp4init( twolines )
    
    @classmethod
    def sgp4init_fromorbit( cls, satnum, intldesg, day_of_year, year, elnum, ecco, inc, Omega, omega, M, nrev, dn, ddn, bstar, whichconst=PZ90, N=None) :
        self = cls()
        
        
        ndot = dn * 2*np.pi*(60**2) # rev/s^2 -> rad/min^2
        nddot = ddn * 2*np.pi*(60**3) # rev/s^3 -> rad/min^3
        no_kozai = nrev * 2*np.pi*60 # rev/s -> rad/min
        
        revnum = day_of_year*86400*nrev
        
        
        epochdays = day_of_year
        epochyr = year
        
        jdsatepoch, jdsatepochF, epoch = day_year_2_jdsatepoch( epochdays, epochyr )
        
        self.add_sat_lines( (satnum, 'U', intldesg, epochdays, ndot, nddot, bstar,
        0, elnum, inc, Omega, ecco, omega, M, no_kozai, revnum, 
        epochyr, jdsatepoch, jdsatepochF, epoch, gravity_constants[whichconst], 'i') )
        
        if N :
            self.extend_sats( N )
        
        self.init_to_numpy()
            
        sgp4deploy( self )
        
        return self
    
    @staticmethod
    def tle_iter_file( file ) :
        name, line1, line2 = '','',''
        for line in file :
            
            if line[0] == '1' :
                line1 = line
            elif line[0] == '2' :
                line2 = line
            else :
                name = line
                
            if line1 and line2 :
                yield name, line1, line2
                name, line1, line2 = '','',''

    @staticmethod
    def tle_iter( filepath ) :
        with open( filepath ) as file :
            yield from Satrec_np.tle_iter_file( file )

    def sgp4(self, jd, fr):
        tsince = ((jd - self.jdsatepoch) * minutes_per_day +
                  (fr - self.jdsatepochF) * minutes_per_day)
        r, v = sgp4_np(self, tsince)
        return self.error, r, v

    def sgp4_tsince(self, tsince):
        r, v = sgp4_np(self, tsince)
        return self.error, r, v

    def sgp4_array(self, jd, fr):
        """Compute positions and velocities for the times in a NumPy array.

        Given NumPy arrays ``jd`` and ``fr`` of the same length that
        supply the whole part and the fractional part of one or more
        Julian dates, return a tuple ``(e, r, v)`` of three vectors:

        * ``e``: nonzero for any dates that produced errors, 0 otherwise.
        * ``r``: position vectors in kilometers.
        * ``v``: velocity vectors in kilometers per second.

        """
        # Import NumPy the first time sgp4_array() is called.
        array = self.array
        if array is None:
            from numpy import array
            Satrec.array = array

        results = []
        z = list(zip(jd, fr))
        for jd_i, fr_i in z:
            results.append(self.sgp4(jd_i, fr_i))
        elist, rlist, vlist = zip(*results)

        e = array(elist)
        r = array(rlist)
        v = array(vlist)

        r.shape = v.shape = len(jd), 3
        return e, r, v
