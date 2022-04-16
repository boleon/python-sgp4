"""The sgp4 procedures for analytical propagation of a satellite.

I have made the rather unorthodox decision to leave as much of this C++
code alone as possible: if a line of code would run without change in
Python, then I refused to re-indent it or remove its terminal semicolon,
so that in the future it will be easier to keep updating this file as
the original author's C++ continues to improve.  Thus, 5-space
indentation (!) prevails in this file.

I have even kept all of the C++ block comments (by turning them into
Python string constants) to make this easier to navigate and maintain,
as well as to make it more informative for people who encounter this
code for the first time here in its Python form.

| - Brandon Rhodes
|   Common Grounds Coffee House, Bluffton, Ohio
|   On a very hot August day in 2012

"""
from math import atan2, cos, fabs, pi, sin, sqrt
import numpy as np

deg2rad = pi / 180.0;
_nan = float('NaN')
false = (_nan, _nan, _nan)
true = True
twopi = 2.0 * pi

"""
/*     ----------------------------------------------------------------
*
*                               sgp4unit.cpp
*
*    this file contains the sgp4 procedures for analytical propagation
*    of a satellite. the code was originally released in the 1980 and 1986
*    spacetrack papers. a detailed discussion of the theory and history
*    may be found in the 2006 aiaa paper by vallado, crawford, hujsak,
*    and kelso.
*
*                            companion code for
*               fundamentals of astrodynamics and applications
*                                    2013
*                              by david vallado
*
*     (w) 719-573-2600, email dvallado@agi.com, davallado@gmail.com
*
*    current :
*               7 dec 15  david vallado
*                           fix jd, jdfrac
*    changes :
*               3 nov 14  david vallado
*                           update to msvs2013 c++
*              30 aug 10  david vallado
*                           delete unused variables in initl
*                           replace pow integer 2, 3 with multiplies for speed
*               3 nov 08  david vallado
*                           put returns in for error codes
*              29 sep 08  david vallado
*                           fix atime for faster operation in dspace
*                           add operationmode for afspc (a) or improved (i)
*                           performance mode
*              16 jun 08  david vallado
*                           update small eccentricity check
*              16 nov 07  david vallado
*                           misc fixes for better compliance
*              20 apr 07  david vallado
*                           misc fixes for constants
*              11 aug 06  david vallado
*                           chg lyddane choice back to strn3, constants, misc doc
*              15 dec 05  david vallado
*                           misc fixes
*              26 jul 05  david vallado
*                           fixes for paper
*                           note that each fix is preceded by a
*                           comment with "sgp4fix" and an explanation of
*                           what was changed
*              10 aug 04  david vallado
*                           2nd printing baseline working
*              14 may 01  david vallado
*                           2nd edition baseline
*                     80  norad
*                           original baseline
*       ----------------------------------------------------------------      */
"""

"""
/* -----------------------------------------------------------------------------
*
*                           procedure dpper
*
*  this procedure provides deep space long period periodic contributions
*    to the mean elements.  by design, these periodics are zero at epoch.
*    this used to be dscom which included initialization, but it's really a
*    recurring function.
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    e3          -
*    ee2         -
*    peo         -
*    pgho        -
*    pho         -
*    pinco       -
*    plo         -
*    se2 , se3 , sgh2, sgh3, sgh4, sh2, sh3, si2, si3, sl2, sl3, sl4 -
*    t           -
*    xh2, xh3, xi2, xi3, xl2, xl3, xl4 -
*    zmol        -
*    zmos        -
*    ep          - eccentricity                           0.0 - 1.0
*    inclo       - inclination - needed for lyddane modification
*    nodep       - right ascension of ascending node
*    argpp       - argument of perigee
*    mp          - mean anomaly
*
*  outputs       :
*    ep          - eccentricity                           0.0 - 1.0
*    inclp       - inclination
*    nodep        - right ascension of ascending node
*    argpp       - argument of perigee
*    mp          - mean anomaly
*
*  locals        :
*    alfdp       -
*    betdp       -
*    cosip  , sinip  , cosop  , sinop  ,
*    dalf        -
*    dbet        -
*    dls         -
*    f2, f3      -
*    pe          -
*    pgh         -
*    ph          -
*    pinc        -
*    pl          -
*    sel   , ses   , sghl  , sghs  , shl   , shs   , sil   , sinzf , sis   ,
*    sll   , sls
*    xls         -
*    xnoh        -
*    zf          -
*    zm          -
*
*  coupling      :
*    none.
*
*  references    :
*    hoots, roehrich, norad spacetrack report #3 1980
*    hoots, norad spacetrack report #6 1986
*    hoots, schumacher and glover 2004
*    vallado, crawford, hujsak, kelso  2006
  ----------------------------------------------------------------------------*/
"""

def _dpper_np(satrec, bl, inclo, init, ep, inclp, nodep, argpp, mp, opsmode) :

    # Copy satellite attributes into local variables for convenience
    # and symmetry in writing formulae.

    e3 = satrec.e3[bl]
    ee2 = satrec.ee2[bl]
    peo = satrec.peo[bl]
    pgho = satrec.pgho[bl]
    pho = satrec.pho[bl]
    pinco = satrec.pinco[bl]
    plo = satrec.plo[bl]
    se2 = satrec.se2[bl]
    se3 = satrec.se3[bl]
    sgh2 = satrec.sgh2[bl]
    sgh3 = satrec.sgh3[bl]
    sgh4 = satrec.sgh4[bl]
    sh2 = satrec.sh2[bl]
    sh3 = satrec.sh3[bl]
    si2 = satrec.si2[bl]
    si3 = satrec.si3[bl]
    sl2 = satrec.sl2[bl]
    sl3 = satrec.sl3[bl]
    sl4 = satrec.sl4[bl]
    t = satrec.t[bl]
    xgh2 = satrec.xgh2[bl]
    xgh3 = satrec.xgh3[bl]
    xgh4 = satrec.xgh4[bl]
    xh2 = satrec.xh2[bl]
    xh3 = satrec.xh3[bl]
    xi2 = satrec.xi2[bl]
    xi3 = satrec.xi3[bl]
    xl2 = satrec.xl2[bl]
    xl3 = satrec.xl3[bl]
    xl4 = satrec.xl4[bl]
    zmol = satrec.zmol[bl]
    zmos = satrec.zmos[bl]
    
    nsat = np.count_nonzero( bl )
    sinip = np.zeros(nsat)
    cosip = np.zeros(nsat)
    
    
     #  ---------------------- constants -----------------------------
    zns   = 1.19459e-5;
    zes   = 0.01675;
    znl   = 1.5835218e-4;
    zel   = 0.05490;

     #  --------------- calculate time varying periodics -----------
    zm = np.where(init == b'y', zmos, zmos + zns * t);
    zf    = zm + 2.0 * zes * np.sin(zm);
    sinzf = np.sin(zf);
    f2    =  0.5 * sinzf * sinzf - 0.25;
    f3    = -0.5 * sinzf * np.cos(zf);
    ses   = se2* f2 + se3 * f3;
    sis   = si2 * f2 + si3 * f3;
    sls   = sl2 * f2 + sl3 * f3 + sl4 * sinzf;
    sghs  = sgh2 * f2 + sgh3 * f3 + sgh4 * sinzf;
    shs   = sh2 * f2 + sh3 * f3;
    zm = np.where(init == b'y', zmol, zmol + znl * t);
    zf    = zm + 2.0 * zel * np.sin(zm);
    sinzf = np.sin(zf);
    f2    =  0.5 * sinzf * sinzf - 0.25;
    f3    = -0.5 * sinzf * np.cos(zf);
    sel   = ee2 * f2 + e3 * f3;
    sil   = xi2 * f2 + xi3 * f3;
    sll   = xl2 * f2 + xl3 * f3 + xl4 * sinzf;
    sghl  = xgh2 * f2 + xgh3 * f3 + xgh4 * sinzf;
    shll  = xh2 * f2 + xh3 * f3;
    pe    = ses + sel;
    pinc  = sis + sil;
    pl    = sls + sll;
    pgh   = sghs + sghl;
    ph    = shs + shll;
    
    bon = (init == b'n')
    if np.any( bon ) :
        pe[bon]    -= peo[bon];
        pinc[bon]  -= pinco[bon];
        pl[bon]    -= plo[bon];
        pgh[bon]   -= pgho[bon];
        ph[bon]    -= pho[bon];
        inclp[bon] += pinc[bon];
        ep[bon]    += pe[bon];
        sinip[bon]  = np.sin(inclp[bon]);
        cosip[bon]  = np.cos(inclp[bon]);
        
        binc = (inclp >= 0.2)
        
        boninc = bon & binc;
        
        if np.any( boninc ) :
            ph[boninc] /= sinip[boninc]
            pgh[boninc] -= cosip[boninc] * ph[boninc]
            argpp[boninc] += pgh[boninc]
            nodep[boninc] += ph[boninc]
            mp[boninc] += pl[boninc]
            
        boninc = bon & np.logical_not(binc);
        
        if np.any( boninc ) :
            sinop_  = np.sin(nodep[boninc]);
            cosop_  = np.cos(nodep[boninc]);
            alfdp_  = sinip[boninc] * sinop_;
            betdp_  = sinip[boninc] * cosop_;
            dalf_   =  ph[boninc] * cosop_ + pinc[boninc] * cosip[boninc] * sinop_;
            dbet_   = -ph[boninc] * sinop_ + pinc[boninc] * cosip[boninc] * cosop_;
            alfdp_  += dalf_;
            betdp_  += dbet_;
            nodep[boninc] = np.where(nodep[boninc] >= 0.0, nodep[boninc] % twopi, -(-nodep[boninc] % twopi) )
            #   sgp4fix for afspc written intrinsic functions
            #  nodep used without a trigonometric function ahead
                
            nodep[boninc] = np.where( (nodep[boninc] < 0.0) & (opsmode[boninc] == 'a'), nodep[boninc] + twopi, nodep[boninc] )
            
            xls_ = mp[boninc] + argpp[boninc] + pl[boninc] + pgh[boninc] \
            + (cosip[boninc] - pinc[boninc] * sinip[boninc]) * nodep[boninc]
            xnoh_   = nodep[boninc];
            nodep[boninc]  = np.arctan2(alfdp_, betdp_);
            #   sgp4fix for afspc written intrinsic functions
            #  nodep used without a trigonometric function ahead
            nodep[boninc] = np.where( (nodep[boninc] < 0.0) & (opsmode[boninc] == 'a'), nodep[boninc] + twopi, nodep[boninc] )
            
            nodep[boninc] += np.where( (np.fabs(xnoh_ - nodep[boninc]) > pi) & (nodep[boninc] < xnoh_), twopi, 0 )
            nodep[boninc] -= np.where( (np.fabs(xnoh_ - nodep[boninc]) > pi) & (nodep[boninc] >= xnoh_), twopi, 0 )
            
            mp[boninc] += pl[boninc]
            argpp[boninc] = xls_ - mp[boninc] - cosip[boninc] * nodep[boninc];
       
    return ep, inclp, nodep, argpp, mp

"""
/*-----------------------------------------------------------------------------
*
*                           procedure dscom
*
*  this procedure provides deep space common items used by both the secular
*    and periodics subroutines.  input is provided as shown. this routine
*    used to be called dpper, but the functions inside weren't well organized.
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    epoch       -
*    ep          - eccentricity
*    argpp       - argument of perigee
*    tc          -
*    inclp       - inclination
*    nodep       - right ascension of ascending node
*    np          - mean motion
*
*  outputs       :
*    sinim  , cosim  , sinomm , cosomm , snodm  , cnodm
*    day         -
*    e3          -
*    ee2         -
*    em          - eccentricity
*    emsq        - eccentricity squared
*    gam         -
*    peo         -
*    pgho        -
*    pho         -
*    pinco       -
*    plo         -
*    rtemsq      -
*    se2, se3         -
*    sgh2, sgh3, sgh4        -
*    sh2, sh3, si2, si3, sl2, sl3, sl4         -
*    s1, s2, s3, s4, s5, s6, s7          -
*    ss1, ss2, ss3, ss4, ss5, ss6, ss7, sz1, sz2, sz3         -
*    sz11, sz12, sz13, sz21, sz22, sz23, sz31, sz32, sz33        -
*    xgh2, xgh3, xgh4, xh2, xh3, xi2, xi3, xl2, xl3, xl4         -
*    nm          - mean motion
*    z1, z2, z3, z11, z12, z13, z21, z22, z23, z31, z32, z33         -
*    zmol        -
*    zmos        -
*
*  locals        :
*    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10         -
*    betasq      -
*    cc          -
*    ctem, stem        -
*    x1, x2, x3, x4, x5, x6, x7, x8          -
*    xnodce      -
*    xnoi        -
*    zcosg  , zsing  , zcosgl , zsingl , zcosh  , zsinh  , zcoshl , zsinhl ,
*    zcosi  , zsini  , zcosil , zsinil ,
*    zx          -
*    zy          -
*
*  coupling      :
*    none.
*
*  references    :
*    hoots, roehrich, norad spacetrack report #3 1980
*    hoots, norad spacetrack report #6 1986
*    hoots, schumacher and glover 2004
*    vallado, crawford, hujsak, kelso  2006
  ----------------------------------------------------------------------------*/
"""

def _dscom_np(
       epoch,  ep,     argpp,   tc,     inclp,
       nodep,  npp,
       e3,     ee2,
       peo,    pgho,  pho,
       pinco, plo, se2,   se3,
       sgh2,  sgh3,  sgh4,   sh2,   sh3,
       si2,   si3,   sl2,    sl3,   sl4,
       xgh2,  xgh3,   xgh4,  xh2,
       xh3,   xi2,   xi3,    xl2,   xl3,
       xl4,   zmol,  zmos,
     ):

     #  -------------------------- constants -------------------------
     zes     =  0.01675;
     zel     =  0.05490;
     c1ss    =  2.9864797e-6;
     c1l     =  4.7968065e-7;
     zsinis  =  0.39785416;
     zcosis  =  0.91744867;
     zcosgs  =  0.1945905;
     zsings  = -0.98088458;

     #  --------------------- local variables ------------------------
     nm     = npp;
     em     = ep;
     snodm  = np.sin(nodep);
     cnodm  = np.cos(nodep);
     sinomm = np.sin(argpp);
     cosomm = np.cos(argpp);
     sinim  = np.sin(inclp);
     cosim  = np.cos(inclp);
     emsq   = em * em;
     betasq = 1.0 - emsq;
     rtemsq = np.sqrt(betasq);

     #  ----------------- initialize lunar solar terms ---------------
     peo    = 0.0;
     pinco  = 0.0;
     plo    = 0.0;
     pgho   = 0.0;
     pho    = 0.0;
     day    = epoch + 18261.5 + tc / 1440.0;
     xnodce = (4.5236020 - 9.2422029e-4 * day) % twopi
     stem   = np.sin(xnodce);
     ctem   = np.cos(xnodce);
     zcosil = 0.91375164 - 0.03568096 * ctem;
     zsinil = np.sqrt(1.0 - zcosil * zcosil);
     zsinhl = 0.089683511 * stem / zsinil;
     zcoshl = np.sqrt(1.0 - zsinhl * zsinhl);
     gam    = 5.8351514 + 0.0019443680 * day;
     zx     = 0.39785416 * stem / zsinil;
     zy     = zcoshl * ctem + 0.91744867 * zsinhl * stem;
     zx     = np.arctan2(zx, zy);
     zx     = gam + zx - xnodce;
     zcosgl = np.cos(zx);
     zsingl = np.sin(zx);

     #  ------------------------- do solar terms ---------------------
     zcosg = zcosgs;
     zsing = zsings;
     zcosi = zcosis;
     zsini = zsinis;
     zcosh = cnodm;
     zsinh = snodm;
     cc    = c1ss;
     xnoi  = 1.0 / nm;

     for lsflg in 1, 2:

         a1  =   zcosg * zcosh + zsing * zcosi * zsinh;
         a3  =  -zsing * zcosh + zcosg * zcosi * zsinh;
         a7  =  -zcosg * zsinh + zsing * zcosi * zcosh;
         a8  =   zsing * zsini;
         a9  =   zsing * zsinh + zcosg * zcosi * zcosh;
         a10 =   zcosg * zsini;
         a2  =   cosim * a7 + sinim * a8;
         a4  =   cosim * a9 + sinim * a10;
         a5  =  -sinim * a7 + cosim * a8;
         a6  =  -sinim * a9 + cosim * a10;

         x1  =  a1 * cosomm + a2 * sinomm;
         x2  =  a3 * cosomm + a4 * sinomm;
         x3  = -a1 * sinomm + a2 * cosomm;
         x4  = -a3 * sinomm + a4 * cosomm;
         x5  =  a5 * sinomm;
         x6  =  a6 * sinomm;
         x7  =  a5 * cosomm;
         x8  =  a6 * cosomm;

         z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3;
         z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4;
         z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4;
         z1  =  3.0 *  (a1 * a1 + a2 * a2) + z31 * emsq;
         z2  =  6.0 *  (a1 * a3 + a2 * a4) + z32 * emsq;
         z3  =  3.0 *  (a3 * a3 + a4 * a4) + z33 * emsq;
         z11 = -6.0 * a1 * a5 + emsq *  (-24.0 * x1 * x7-6.0 * x3 * x5);
         z12 = -6.0 *  (a1 * a6 + a3 * a5) + emsq * \
                (-24.0 * (x2 * x7 + x1 * x8) - 6.0 * (x3 * x6 + x4 * x5));
         z13 = -6.0 * a3 * a6 + emsq * (-24.0 * x2 * x8 - 6.0 * x4 * x6);
         z21 =  6.0 * a2 * a5 + emsq * (24.0 * x1 * x5 - 6.0 * x3 * x7);
         z22 =  6.0 *  (a4 * a5 + a2 * a6) + emsq * \
                (24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8));
         z23 =  6.0 * a4 * a6 + emsq * (24.0 * x2 * x6 - 6.0 * x4 * x8);
         z1  = z1 + z1 + betasq * z31;
         z2  = z2 + z2 + betasq * z32;
         z3  = z3 + z3 + betasq * z33;
         s3  = cc * xnoi;
         s2  = -0.5 * s3 / rtemsq;
         s4  = s3 * rtemsq;
         s1  = -15.0 * em * s4;
         s5  = x1 * x3 + x2 * x4;
         s6  = x2 * x3 + x1 * x4;
         s7  = x2 * x4 - x1 * x3;

         #  ----------------------- do lunar terms -------------------
         if lsflg == 1:

             ss1   = s1;
             ss2   = s2;
             ss3   = s3;
             ss4   = s4;
             ss5   = s5;
             ss6   = s6;
             ss7   = s7;
             sz1   = z1;
             sz2   = z2;
             sz3   = z3;
             sz11  = z11;
             sz12  = z12;
             sz13  = z13;
             sz21  = z21;
             sz22  = z22;
             sz23  = z23;
             sz31  = z31;
             sz32  = z32;
             sz33  = z33;
             zcosg = zcosgl;
             zsing = zsingl;
             zcosi = zcosil;
             zsini = zsinil;
             zcosh = zcoshl * cnodm + zsinhl * snodm;
             zsinh = snodm * zcoshl - cnodm * zsinhl;
             cc    = c1l;

     zmol = (4.7199672 + 0.22997150  * day - gam) % twopi
     zmos = (6.2565837 + 0.017201977 * day) % twopi

     #  ------------------------ do solar terms ----------------------
     se2  =   2.0 * ss1 * ss6;
     se3  =   2.0 * ss1 * ss7;
     si2  =   2.0 * ss2 * sz12;
     si3  =   2.0 * ss2 * (sz13 - sz11);
     sl2  =  -2.0 * ss3 * sz2;
     sl3  =  -2.0 * ss3 * (sz3 - sz1);
     sl4  =  -2.0 * ss3 * (-21.0 - 9.0 * emsq) * zes;
     sgh2 =   2.0 * ss4 * sz32;
     sgh3 =   2.0 * ss4 * (sz33 - sz31);
     sgh4 = -18.0 * ss4 * zes;
     sh2  =  -2.0 * ss2 * sz22;
     sh3  =  -2.0 * ss2 * (sz23 - sz21);

     #  ------------------------ do lunar terms ----------------------
     ee2  =   2.0 * s1 * s6;
     e3   =   2.0 * s1 * s7;
     xi2  =   2.0 * s2 * z12;
     xi3  =   2.0 * s2 * (z13 - z11);
     xl2  =  -2.0 * s3 * z2;
     xl3  =  -2.0 * s3 * (z3 - z1);
     xl4  =  -2.0 * s3 * (-21.0 - 9.0 * emsq) * zel;
     xgh2 =   2.0 * s4 * z32;
     xgh3 =   2.0 * s4 * (z33 - z31);
     xgh4 = -18.0 * s4 * zel;
     xh2  =  -2.0 * s2 * z22;
     xh3  =  -2.0 * s2 * (z23 - z21);

     return (
       snodm, cnodm, sinim,  cosim, sinomm,
       cosomm,day,   e3,     ee2,   em,
       emsq,  gam,   peo,    pgho,  pho,
       pinco, plo,   rtemsq, se2,   se3,
       sgh2,  sgh3,  sgh4,   sh2,   sh3,
       si2,   si3,   sl2,    sl3,   sl4,
       s1,    s2,    s3,     s4,    s5,
       s6,    s7,    ss1,    ss2,   ss3,
       ss4,   ss5,   ss6,    ss7,   sz1,
       sz2,   sz3,   sz11,   sz12,  sz13,
       sz21,  sz22,  sz23,   sz31,  sz32,
       sz33,  xgh2,  xgh3,   xgh4,  xh2,
       xh3,   xi2,   xi3,    xl2,   xl3,
       xl4,   nm,    z1,     z2,    z3,
       z11,   z12,   z13,    z21,   z22,
       z23,   z31,   z32,    z33,   zmol,
       zmos
       )

"""
/*-----------------------------------------------------------------------------
*
*                           procedure dsinit
*
*  this procedure provides deep space contributions to mean motion dot due
*    to geopotential resonance with half day and one day orbits.
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    cosim, sinim-
*    emsq        - eccentricity squared
*    argpo       - argument of perigee
*    s1, s2, s3, s4, s5      -
*    ss1, ss2, ss3, ss4, ss5 -
*    sz1, sz3, sz11, sz13, sz21, sz23, sz31, sz33 -
*    t           - time
*    tc          -
*    gsto        - greenwich sidereal time                   rad
*    mo          - mean anomaly
*    mdot        - mean anomaly dot (rate)
*    no          - mean motion
*    nodeo       - right ascension of ascending node
*    nodedot     - right ascension of ascending node dot (rate)
*    xpidot      -
*    z1, z3, z11, z13, z21, z23, z31, z33 -
*    eccm        - eccentricity
*    argpm       - argument of perigee
*    inclm       - inclination
*    mm          - mean anomaly
*    xn          - mean motion
*    nodem       - right ascension of ascending node
*
*  outputs       :
*    em          - eccentricity
*    argpm       - argument of perigee
*    inclm       - inclination
*    mm          - mean anomaly
*    nm          - mean motion
*    nodem       - right ascension of ascending node
*    irez        - flag for resonance           0-none, 1-one day, 2-half day
*    atime       -
*    d2201, d2211, d3210, d3222, d4410, d4422, d5220, d5232, d5421, d5433    -
*    dedt        -
*    didt        -
*    dmdt        -
*    dndt        -
*    dnodt       -
*    domdt       -
*    del1, del2, del3        -
*    ses  , sghl , sghs , sgs  , shl  , shs  , sis  , sls
*    theta       -
*    xfact       -
*    xlamo       -
*    xli         -
*    xni
*
*  locals        :
*    ainv2       -
*    aonv        -
*    cosisq      -
*    eoc         -
*    f220, f221, f311, f321, f322, f330, f441, f442, f522, f523, f542, f543  -
*    g200, g201, g211, g300, g310, g322, g410, g422, g520, g521, g532, g533  -
*    sini2       -
*    temp        -
*    temp1       -
*    theta       -
*    xno2        -
*
*  coupling      :
*    getgravconst
*
*  references    :
*    hoots, roehrich, norad spacetrack report #3 1980
*    hoots, norad spacetrack report #6 1986
*    hoots, schumacher and glover 2004
*    vallado, crawford, hujsak, kelso  2006
  ----------------------------------------------------------------------------*/
"""

def _dsinit_np(
       # sgp4fix no longer needed pass in xke
       # whichconst,
       xke,
       cosim,  emsq,   argpo,   s1,     s2,
       s3,     s4,     s5,      sinim,  ss1,
       ss2,    ss3,    ss4,     ss5,    sz1,
       sz3,    sz11,   sz13,    sz21,   sz23,
       sz31,   sz33,   t,       tc,     gsto,
       mo,     mdot,   no,      nodeo,  nodedot,
       xpidot, z1,     z3,      z11,    z13,
       z21,    z23,    z31,     z33,    ecco,
       eccsq,  em,    argpm,  inclm, mm,
       nm,    nodem,
       irez,
       atime, d2201, d2211,  d3210, d3222,
       d4410, d4422, d5220,  d5232, d5421,
       d5433, dedt,  didt,   dmdt,
       dnodt, domdt, del1,   del2,  del3,
       xfact, xlamo, xli,    xni,
     ):

     q22    = 1.7891679e-6;
     q31    = 2.1460748e-6;
     q33    = 2.2123015e-7;
     root22 = 1.7891679e-6;
     root44 = 7.3636953e-9;
     root54 = 2.1765803e-9;
     rptim  = 4.37526908801129966e-3; # equates to 7.29211514668855e-5 rad/sec
     root32 = 3.7393792e-7;
     root52 = 1.1428639e-7;
     x2o3   = 2.0 / 3.0;
     znl    = 1.5835218e-4;
     zns    = 1.19459e-5;

	 # sgp4fix identify constants and allow alternate values
	 # just xke is used here so pass it in rather than have multiple calls
     # xke = whichconst.xke

     #  -------------------- deep space initialization ------------
     irez = 0;
     irez = np.where( (0.0034906585 < nm) & (nm < 0.0052359877), 1, 0 )
     irez = np.where( (8.26e-3 <= nm) & ( nm <= 9.24e-3) & (em >= 0.5), 2, irez )

     #  ------------------------ do solar terms -------------------
     ses  =  ss1 * zns * ss5;
     sis  =  ss2 * zns * (sz11 + sz13);
     sls  = -zns * ss3 * (sz1 + sz3 - 14.0 - 6.0 * emsq);
     sghs =  ss4 * zns * (sz31 + sz33 - 6.0);
     shs  = -zns * ss2 * (sz21 + sz23);
     #  sgp4fix for 180 deg incl
     shs = np.where( (inclm < 5.2359877e-2) | (inclm > pi - 5.2359877e-2), 0.0, shs )
     bt = (sinim != 0.0)
     if np.any(bt) :
       shs[bt] /= sinim[bt];
     sgs  = sghs - cosim * shs;

     #  ------------------------- do lunar terms ------------------
     dedt = ses + s1 * znl * s5;
     didt = sis + s2 * znl * (z11 + z13);
     dmdt = sls - znl * s3 * (z1 + z3 - 14.0 - 6.0 * emsq);
     sghl = s4 * znl * (z31 + z33 - 6.0);
     shll = -znl * s2 * (z21 + z23);
     #  sgp4fix for 180 deg incl
     shll = np.where( (inclm < 5.2359877e-2) | (inclm > pi - 5.2359877e-2), 0.0, shll )
     domdt = sgs + sghl;
     dnodt = shs;
     bt = (sinim != 0.0)
     if np.any(bt) :
         domdt[bt] -= cosim[bt] / sinim[bt] * shll[bt];
         dnodt[bt] += shll[bt] / sinim[bt];


     #  ----------- calculate deep space resonance effects --------
     dndt   = 0.0;
     theta  = (gsto + tc * rptim) % twopi
     em     = em + dedt * t;
     inclm  = inclm + didt * t;
     argpm  = argpm + domdt * t;
     nodem  = nodem + dnodt * t;
     mm     = mm + dmdt * t;
     """
     //   sgp4fix for negative inclinations
     //   the following if statement should be commented out
     //if (inclm < 0.0)
     //  {
     //    inclm  = -inclm;
     //    argpm  = argpm - pi;
     //    nodem = nodem + pi;
     //  }
     """
     
     

     #  -------------- initialize the resonance terms -------------
     bt = (irez == 2)
     if np.any(bt) :
         aonv_ = pow(nm[bt] / xke[bt], x2o3);
         
         cosisq_ = cosim[bt] * cosim[bt];
         emo_    = em[bt];
         em_     = ecco[bt];
         emsqo_  = emsq[bt];
         emsq_   = eccsq[bt];
         eoc_    = em_ * emsq_;
         g201_   = -0.306 - (em_ - 0.64) * 0.440;
         
         g211_ = np.where( em_ <= 0.65, 
                         3.616 - 13.2470*em_ + 16.2900*emsq_, 
                         -72.099 + 331.819*em_ - 508.738*emsq_ + 266.724*eoc_)
         g310_ = np.where( em_ <= 0.65, 
                         -19.302 + 117.3900*em_ - 228.4190*emsq_ + 156.5910*eoc_,
                         -346.844 +  1582.851*em_ - 2415.925*emsq_ + 1246.113*eoc_)
         g322_ = np.where( em_ <= 0.65, 
                         -18.9068 + 109.7927*em_ - 214.6334*emsq_ + 146.5816*eoc_,
                         -342.585 + 1554.908*em_ - 2366.899*emsq_ + 1215.972*eoc_)
         g410_ = np.where( em_ <= 0.65, 
                         -41.122 + 242.6940*em_ - 471.0940*emsq_ + 313.9530*eoc_,
                         -1052.797 + 4758.686*em_ - 7193.992*emsq_ + 3651.957*eoc_)
         g422_ = np.where( em_ <= 0.65, 
                         -146.407 + 841.8800*em_ - 1629.014*emsq_ + 1083.4350*eoc_, 
                         -3581.690 + 16178.110*em_ - 24462.770*emsq_ + 12422.520*eoc_)
         g520_ = np.where( em_ <= 0.65,
                         -532.114 + 3017.977*em_ - 5740.032*emsq_ + 3708.2760*eoc_,
                         1464.74 - 4664.75*em_ + 3763.64*emsq_)
         g520_ = np.where( em_ > 0.715,
                         -5149.66 + 29936.92*em_ - 54087.36*emsq_ + 31324.56*eoc_,
                         g520_)
         
         g533_ = np.where( em_ < 0.7,
                         -919.22770 + 4988.6100*em_ - 9064.7700*emsq_ + 5542.21*eoc_,
                         -37995.780 + 161616.52*em_ - 229838.20*emsq_ + 109377.94*eoc_)
         g521_ = np.where( em_ < 0.7,
                         -822.71072 + 4568.6173*em_ - 8491.4146*emsq_ + 5337.524*eoc_,
                         -51752.104 + 218913.95*em_ - 309468.16*emsq_ + 146349.42*eoc_)
         g532_ = np.where( em_ < 0.7,
                         -853.66600 + 4690.2500*em_ - 8624.7700*emsq_ + 5341.4*eoc_,
                         -40023.880 + 170470.89*em_ - 242699.48*emsq_ + 115605.82*eoc_)
         
         sini2_=  sinim[bt] * sinim[bt];
         f220_ =  0.75 * (1.0 + 2.0 * cosim[bt]+cosisq_);
         f221_ =  1.5 * sini2_;
         f321_ =  1.875 * sinim[bt]  *  (1.0 - 2.0 * cosim[bt] - 3.0 * cosisq_);
         f322_ = -1.875 * sinim[bt]  *  (1.0 + 2.0 * cosim[bt] - 3.0 * cosisq_);
         f441_ = 35.0 * sini2_ * f220_;
         f442_ = 39.3750 * sini2_ * sini2_;
         f522_ =  9.84375 * sinim[bt] * (sini2_ * (1.0 - 2.0 * cosim[bt]- 5.0 * cosisq_) +
                                    0.33333333 * (-2.0 + 4.0 * cosim[bt] + 6.0 * cosisq_) );
         f523_ = sinim[bt] * (4.92187512 * sini2_ * (-2.0 - 4.0 * cosim[bt] +
               10.0 * cosisq_) + 6.56250012 * (1.0+2.0 * cosim[bt] - 3.0 * cosisq_));
         f542_ = 29.53125 * sinim[bt] * (2.0 - 8.0 * cosim[bt]+cosisq_ *
               (-12.0 + 8.0 * cosim[bt] + 10.0 * cosisq_));
         f543_ = 29.53125 * sinim[bt] * (-2.0 - 8.0 * cosim[bt]+cosisq_ *
               (12.0 + 8.0 * cosim[bt] - 10.0 * cosisq_));
         xno2_  =  nm[bt] * nm[bt];
         ainv2_ =  aonv_ * aonv_;
         temp1_ =  3.0 * xno2_ * ainv2_;
         temp_  =  temp1_ * root22;
         d2201[bt] =  temp_ * f220_ * g201_;
         d2211[bt] =  temp_ * f221_ * g211_;
         temp1_ =  temp1_ * aonv_;
         temp_  =  temp1_ * root32;
         d3210[bt] =  temp_ * f321_ * g310_;
         d3222[bt] =  temp_ * f322_ * g322_;
         temp1_ =  temp1_ * aonv_;
         temp_  =  2.0 * temp1_ * root44;
         d4410[bt] =  temp_ * f441_ * g410_;
         d4422[bt] =  temp_ * f442_ * g422_;
         temp1_ =  temp1_ * aonv_;
         temp_  =  temp1_ * root52;
         d5220[bt] =  temp_ * f522_ * g520_;
         d5232[bt] =  temp_ * f523_ * g532_;
         temp_  =  2.0 * temp1_ * root54;
         d5421[bt] =  temp_ * f542_ * g521_;
         d5433[bt] =  temp_ * f543_ * g533_;
         xlamo[bt] =  (mo[bt] + nodeo[bt] + nodeo[bt]-theta[bt] - theta[bt]) % twopi
         xfact[bt] =  mdot[bt] + dmdt[bt] + 2.0 * (nodedot[bt] + dnodt[bt] - rptim) - no[bt];
         em[bt]    = emo_;
         emsq[bt]  = emsqo_;
         
     bt = (irez == 1)
     if np.any(bt) :
         aonv_ = pow(nm[bt] / xke[bt], x2o3);
         g200_  = 1.0 + emsq[bt] * (-2.5 + 0.8125 * emsq[bt]);
         g310_  = 1.0 + 2.0 * emsq[bt];
         g300_  = 1.0 + emsq[bt] * (-6.0 + 6.60937 * emsq[bt]);
         f220_  = 0.75 * (1.0 + cosim[bt]) * (1.0 + cosim[bt]);
         f311_  = 0.9375 * sinim[bt] * sinim[bt] * (1.0 + 3.0 * cosim[bt]) - 0.75 * (1.0 + cosim[bt]);
         f330_  = 1.0 + cosim[bt];
         f330_  = 1.875 * f330_ * f330_ * f330_;
         del1[bt]  = 3.0 * nm[bt] * nm[bt] * aonv_ * aonv_;
         del2[bt]  = 2.0 * del1[bt] * f220_ * g200_ * q22;
         del3[bt]  = 3.0 * del1[bt] * f330_ * g300_ * q33 * aonv_;
         del1[bt]  = del1[bt] * f311_ * g310_ * q31 * aonv_;
         xlamo[bt] = (mo[bt] + nodeo[bt] + argpo[bt] - theta[bt]) % twopi
         xfact[bt] = mdot[bt] + xpidot[bt] - rptim + dmdt[bt] + domdt[bt] + dnodt[bt] - no[bt];
         
     xli   = xlamo;
     xni   = no;
     atime = 0.0;
     nm    = no + dndt;
         
     return (
       em,    argpm,  inclm, mm,
       nm,    nodem,
       irez, atime,
       d2201, d2211,  d3210, d3222,
       d4410, d4422, d5220,  d5232,
       d5421, d5433, dedt,  didt,
       dmdt,  dndt, dnodt, domdt,
       del1,   del2,  del3, xfact,
       xlamo, xli,    xni,
       )


"""
/*-----------------------------------------------------------------------------
*
*                           procedure dspace
*
*  this procedure provides deep space contributions to mean elements for
*    perturbing third body.  these effects have been averaged over one
*    revolution of the sun and moon.  for earth resonance effects, the
*    effects have been averaged over no revolutions of the satellite.
*    (mean motion)
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    d2201, d2211, d3210, d3222, d4410, d4422, d5220, d5232, d5421, d5433 -
*    dedt        -
*    del1, del2, del3  -
*    didt        -
*    dmdt        -
*    dnodt       -
*    domdt       -
*    irez        - flag for resonance           0-none, 1-one day, 2-half day
*    argpo       - argument of perigee
*    argpdot     - argument of perigee dot (rate)
*    t           - time
*    tc          -
*    gsto        - gst
*    xfact       -
*    xlamo       -
*    no          - mean motion
*    atime       -
*    em          - eccentricity
*    ft          -
*    argpm       - argument of perigee
*    inclm       - inclination
*    xli         -
*    mm          - mean anomaly
*    xni         - mean motion
*    nodem       - right ascension of ascending node
*
*  outputs       :
*    atime       -
*    em          - eccentricity
*    argpm       - argument of perigee
*    inclm       - inclination
*    xli         -
*    mm          - mean anomaly
*    xni         -
*    nodem       - right ascension of ascending node
*    dndt        -
*    nm          - mean motion
*
*  locals        :
*    delt        -
*    ft          -
*    theta       -
*    x2li        -
*    x2omi       -
*    xl          -
*    xldot       -
*    xnddt       -
*    xndt        -
*    xomi        -
*
*  coupling      :
*    none        -
*
*  references    :
*    hoots, roehrich, norad spacetrack report #3 1980
*    hoots, norad spacetrack report #6 1986
*    hoots, schumacher and glover 2004
*    vallado, crawford, hujsak, kelso  2006
  ----------------------------------------------------------------------------*/
"""

def _dspace_np(
       irez,
       d2201,  d2211,  d3210,   d3222,  d4410,
       d4422,  d5220,  d5232,   d5421,  d5433,
       dedt,   del1,   del2,    del3,   didt,
       dmdt,   dnodt,  domdt,   argpo,  argpdot,
       t,      tc,     gsto,    xfact,  xlamo,
       no,
       atime, em,    argpm,  inclm, xli,
       mm,    xni,   nodem,  nm,
       ):

    fasx2 = 0.13130908;
    fasx4 = 2.8843198;
    fasx6 = 0.37448087;
    g22   = 5.7686396;
    g32   = 0.95240898;
    g44   = 1.8014998;
    g52   = 1.0508330;
    g54   = 4.4108898;
    rptim = 4.37526908801129966e-3; # equates to 7.29211514668855e-5 rad/sec
    stepp =    720.0;
    stepn =   -720.0;
    step2 = 259200.0;

    #  ----------- calculate deep space resonance effects -----------
    
    nsat   = len(irez)
    dndt   = np.zeros(nsat);
    theta  = (gsto + tc * rptim) % twopi
    em     = em + dedt * t;

    inclm  = inclm + didt * t;
    argpm  = argpm + domdt * t;
    nodem  = nodem + dnodt * t;
    mm     = mm + dmdt * t;

    """
     //   sgp4fix for negative inclinations
     //   the following if statement should be commented out
     //  if (inclm < 0.0)
     // {
     //    inclm = -inclm;
     //    argpm = argpm - pi;
     //    nodem = nodem + pi;
     //  }

     /* - update resonances : numerical (euler-maclaurin) integration - */
     /* ------------------------- epoch restart ----------------------  */
     //   sgp4fix for propagator problems
     //   the following integration works for negative time steps and periods
     //   the specific changes are unknown because the original code was so convoluted

     // sgp4fix take out atime = 0.0 and fix for faster operation
     """
    ft    = np.zeros(nsat);
    borez = (irez != 0)
    delt = np.zeros(nsat)
    xndt = np.zeros(nsat)
    xldot = np.zeros(nsat)
    xnddt = np.zeros(nsat)
    xl = np.zeros(nsat)
    if np.any(borez) :

        #  sgp4fix streamline check
        bt = borez & ((atime == 0.0) | (t * atime <= 0.0) | (np.fabs(t) < np.fabs(atime)))
        if np.any( bt ):

            atime[bt]  = 0.0;
            xni[bt]    = no[bt];
            xli[bt]    = xlamo[bt];

         # sgp4fix move check outside loop
        delt[borez] = stepn
        delt[borez & (t > 0.0)] = stepp

        iretn = np.ones(nsat,dtype=int)*381; # added for do loop
         # iret  =   0; # added for loop
        boret = (iretn == 381) & borez
        while np.any(boret) :

             #  ------------------- dot terms calculated -------------
             #  ----------- near - synchronous resonance terms -------
            borez2 = (irez != 2) & boret
            if np.any(borez2) :

                xndt[borez2]  = del1[borez2] * np.sin(xli[borez2] - fasx2) + del2[borez2] * np.sin(2.0 * (xli[borez2] - fasx4)) + \
                         del3[borez2] * np.sin(3.0 * (xli[borez2] - fasx6));
                xldot[borez2] = xni[borez2] + xfact[borez2];
                xnddt[borez2] = del1[borez2] * np.cos(xli[borez2] - fasx2) + \
                         2.0 * del2[borez2] * np.cos(2.0 * (xli[borez2] - fasx4)) + \
                         3.0 * del3[borez2] * np.cos(3.0 * (xli[borez2] - fasx6));
                xnddt[borez2] *= xldot[borez2];

            borez2 = (irez == 2) & boret
            if np.any(borez2) :
                 # --------- near - half-day resonance terms --------
                xomi_  = argpo[borez2] + argpdot[borez2] * atime[borez2];
                x2omi_ = xomi_ + xomi_;
                x2li_  = xli[borez2] + xli[borez2];
                xndt[borez2]  = (d2201[borez2] * np.sin(x2omi_ + xli[borez2] - g22) + d2211[borez2] * np.sin(xli[borez2] - g22) +
                       d3210[borez2] * np.sin(xomi_ + xli[borez2] - g32)  + d3222[borez2] * np.sin(-xomi_ + xli[borez2] - g32)+
                       d4410[borez2] * np.sin(x2omi_ + x2li_ - g44)+ d4422[borez2] * np.sin(x2li_ - g44) +
                       d5220[borez2] * np.sin(xomi_ + xli[borez2] - g52)  + d5232[borez2] * np.sin(-xomi_ + xli[borez2] - g52)+
                       d5421[borez2] * np.sin(xomi_ + x2li_ - g54) + d5433[borez2] * np.sin(-xomi_ + x2li_ - g54));
                xldot[borez2] = xni[borez2] + xfact[borez2];
                xnddt[borez2] = (d2201[borez2] * np.cos(x2omi_ + xli[borez2] - g22) + d2211[borez2] * np.cos(xli[borez2] - g22) +
                       d3210[borez2] * np.cos(xomi_ + xli[borez2] - g32) + d3222[borez2] * np.cos(-xomi_ + xli[borez2] - g32) +
                       d5220[borez2] * np.cos(xomi_ + xli[borez2] - g52) + d5232[borez2] * np.cos(-xomi_ + xli[borez2] - g52) +
                       2.0 * (d4410[borez2] * np.cos(x2omi_ + x2li_ - g44) +
                       d4422[borez2] * np.cos(x2li_ - g44) + d5421[borez2] * np.cos(xomi_ + x2li_ - g54) +
                       d5433[borez2] * np.cos(-xomi_ + x2li_ - g54)));
                xnddt[borez2] *= xldot[borez2];

             #  ----------------------- integrator -------------------
             #  sgp4fix move end checks to end of routine
            bocont = (np.fabs(t - atime) >= stepp)
            bt = boret & bocont
            iretn[ bt] = 381
            bt = boret & np.logical_not(bocont)
            iretn[bt] = 0
            ft[bt]    = t[bt] - atime[bt]

            boret = (iretn == 381) & borez
            if np.any(boret) :
                xli[boret]   = xli[boret] + xldot[boret] * delt[boret] + xndt[boret] * step2;
                xni[boret]   = xni[boret] + xndt[boret] * delt[boret] + xnddt[boret] * step2;
                atime[boret] = atime[boret] + delt[boret];

        nm[borez] = xni[borez] + xndt[borez] * ft[borez] + xnddt[borez] * ft[borez] * ft[borez] * 0.5;
        xl[borez] = xli[borez] + xldot[borez] * ft[borez] + xndt[borez] * ft[borez] * ft[borez] * 0.5;
        
        borez1 = (irez != 1)
        bt = borez & borez1
        if np.any(bt) :
            mm[bt]   = xl[bt] - 2.0 * nodem[bt] + 2.0 * theta[bt];
            dndt[bt] = nm[bt] - no[bt];

        bt = borez & np.logical_not(borez1)
        if np.any(bt) :
             mm[bt]   = xl[bt] - nodem[bt] - argpm[bt] + theta[bt];
             dndt[bt] = nm[bt] - no[bt];

        nm[borez] = no[borez] + dndt[borez];

    return (
       atime, em,    argpm,  inclm, xli,
       mm,    xni,   nodem,  dndt,  nm,
       )

"""
/*-----------------------------------------------------------------------------
*
*                           procedure initl
*
*  this procedure initializes the spg4 propagator. all the initialization is
*    consolidated here instead of having multiple loops inside other routines.
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    satn        - satellite number - not needed, placed in satrec
*    xke         - reciprocal of tumin
*    j2          - j2 zonal harmonic
*    ecco        - eccentricity                           0.0 - 1.0
*    epoch       - epoch time in days from jan 0, 1950. 0 hr
*    inclo       - inclination of satellite
*    no          - mean motion of satellite
*
*  outputs       :
*    ainv        - 1.0 / a
*    ao          - semi major axis
*    con41       -
*    con42       - 1.0 - 5.0 cos(i)
*    cosio       - cosine of inclination
*    cosio2      - cosio squared
*    eccsq       - eccentricity squared
*    method      - flag for deep space                    'd', 'n'
*    omeosq      - 1.0 - ecco * ecco
*    posq        - semi-parameter squared
*    rp          - radius of perigee
*    rteosq      - square root of (1.0 - ecco*ecco)
*    sinio       - sine of inclination
*    gsto        - gst at time of observation               rad
*    no          - mean motion of satellite
*
*  locals        :
*    ak          -
*    d1          -
*    del         -
*    adel        -
*    po          -
*
*  coupling      :
*    getgravconst- no longer used
*    gstime      - find greenwich sidereal time from the julian date
*
*  references    :
*    hoots, roehrich, norad spacetrack report #3 1980
*    hoots, norad spacetrack report #6 1986
*    hoots, schumacher and glover 2004
*    vallado, crawford, hujsak, kelso  2006
  ----------------------------------------------------------------------------*/
"""

def gstime_np(jdut1):

    tut1 = (jdut1 - 2451545.0) / 36525.0;
    temp = -6.2e-6* tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + \
             (876600.0*3600 + 8640184.812866) * tut1 + 67310.54841;  #  sec
    temp = (temp * deg2rad / 240.0) % twopi # 360/86400 = 1/240, to deg, to rad

     #  ------------------------ check quadrants ---------------------
    temp[temp < 0.0] += twopi
    
    return temp;

def gstime70_np(ts70):

    ds70 = (ts70 + 1.0e-8) // 1.0;
    tfrac = ts70 - ds70;
    #  find greenwich location at epoch
    c1    = 1.72027916940703639e-2;
    thgr70= 1.7321343856509374;
    fk5r  = 5.07551419432269442e-15;
    c1p2p = c1 + twopi;
    temp  = (thgr70 + c1*ds70 + c1p2p*tfrac + ts70*ts70*fk5r) % twopi

     #  ------------------------ check quadrants ---------------------
    temp[temp < 0.0] += twopi
    
    return temp;    

def _initl_np(
       # not needeed. included in satrec if needed later
       # satn,
       # sgp4fix assin xke and j2
       # whichconst,
       xke, j2,
       ecco,   epoch,  inclo,   no,
       opsmode,
       ):

     # sgp4fix use old way of finding gst

     #  ----------------------- earth constants ----------------------
     #  sgp4fix identify constants and allow alternate values
	 #  only xke and j2 are used here so pass them in directly
     #  tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 = whichconst
     x2o3   = 2.0 / 3.0;

     #  ------------- calculate auxillary epoch quantities ----------
     eccsq  = ecco * ecco;
     omeosq = 1.0 - eccsq;
     rteosq = np.sqrt(omeosq);
     cosio  = np.cos(inclo);
     cosio2 = cosio * cosio;

     #  ------------------ un-kozai the mean motion -----------------
     ak    = np.power(xke / no, x2o3);
     d1    = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq);
     del_  = d1 / (ak * ak);
     adel  = ak * (1.0 - del_ * del_ - del_ *
             (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0));
     del_  = d1/(adel * adel);
     no    = no / (1.0 + del_);

     ao    = np.power(xke / no, x2o3);
     sinio = np.sin(inclo);
     po    = ao * omeosq;
     con42 = 1.0 - 5.0 * cosio2;
     con41 = -con42-cosio2-cosio2;
     ainv  = 1.0 / ao;
     posq  = po * po;
     rp    = ao * (1.0 - ecco);

     gsto = np.where( opsmode == 'a', gstime70_np(epoch - 7305.0), gstime_np(epoch + 2433281.5) ) 

     return (
       no,
       ainv,  ao,    con41,  con42, cosio,
       cosio2,eccsq, omeosq, posq,
       rp,    rteosq,sinio , gsto,
       )

"""
/*-----------------------------------------------------------------------------
*
*                             procedure sgp4init
*
*  this procedure initializes variables for sgp4.
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    opsmode     - mode of operation afspc or improved 'a', 'i'
*    whichconst  - which set of constants to use  72, 84
*    satn        - satellite number
*    bstar       - sgp4 type drag coefficient              kg/m2er
*    ecco        - eccentricity
*    epoch       - epoch time in days from jan 0, 1950. 0 hr
*    argpo       - argument of perigee (output if ds)
*    inclo       - inclination
*    mo          - mean anomaly (output if ds)
*    no          - mean motion
*    nodeo       - right ascension of ascending node
*
*  outputs       :
*    satrec      - common values for subsequent calls
*    return code - non-zero on error.
*                   1 - mean elements, ecc >= 1.0 or ecc < -0.001 or a < 0.95 er
*                   2 - mean motion less than 0.0
*                   3 - pert elements, ecc < 0.0  or  ecc > 1.0
*                   4 - semi-latus rectum < 0.0
*                   5 - epoch elements are sub-orbital
*                   6 - satellite has decayed
*
*  locals        :
*    cnodm  , snodm  , cosim  , sinim  , cosomm , sinomm
*    cc1sq  , cc2    , cc3
*    coef   , coef1
*    cosio4      -
*    day         -
*    dndt        -
*    em          - eccentricity
*    emsq        - eccentricity squared
*    eeta        -
*    etasq       -
*    gam         -
*    argpm       - argument of perigee
*    nodem       -
*    inclm       - inclination
*    mm          - mean anomaly
*    nm          - mean motion
*    perige      - perigee
*    pinvsq      -
*    psisq       -
*    qzms24      -
*    rtemsq      -
*    s1, s2, s3, s4, s5, s6, s7          -
*    sfour       -
*    ss1, ss2, ss3, ss4, ss5, ss6, ss7         -
*    sz1, sz2, sz3
*    sz11, sz12, sz13, sz21, sz22, sz23, sz31, sz32, sz33        -
*    tc          -
*    temp        -
*    temp1, temp2, temp3       -
*    tsi         -
*    xpidot      -
*    xhdot1      -
*    z1, z2, z3          -
*    z11, z12, z13, z21, z22, z23, z31, z32, z33         -
*
*  coupling      :
*    getgravconst-
*    initl       -
*    dscom       -
*    dpper       -
*    dsinit      -
*    sgp4        -
*
*  references    :
*    hoots, roehrich, norad spacetrack report #3 1980
*    hoots, norad spacetrack report #6 1986
*    hoots, schumacher and glover 2004
*    vallado, crawford, hujsak, kelso  2006
  ----------------------------------------------------------------------------*/
"""
def sgp4deploy( satrec ) :
   
   
    """
    ( satnum, classification, intldesg, epochdays, ndot, nddot, bstar,
       ephtype, elnum, inclo, nodeo, ecco, argpo, mo, no_kozai, revnum, 
       epochyr, jdsatepoch, epoch, whichconst, operationmode )
    /* ------------------------ initialization --------------------- */
    // sgp4fix divisor for divide by zero check on inclination
    // the old check used 1.0 + cos(pi-1.0e-9), but then compared it to
    // 1.5 e-12, so the threshold was changed to 1.5e-12 for consistency
    """
    temp4    =   1.5e-12;
    nsat = len(satrec.satnum)
    satrec.error = np.zeros(nsat, dtype=int);
   
    """
    // sgp4fix - note the following variables are also passed directly via satrec.
    // it is possible to streamline the sgp4init call by deleting the "x"
    // variables, but the user would need to set the satrec.* values first. we
    // include the additional assignments in case twoline2rv is not used.
    """
   
   # single averaged mean elements
    satrec.am = np.zeros(nsat)
    satrec.em = np.zeros(nsat)
    satrec.im = np.zeros(nsat)
    satrec.Om = np.zeros(nsat)
    satrec.mm = np.zeros(nsat)
    satrec.nm = np.zeros(nsat)
   
   # ------------------------ earth constants ----------------------- */
   	# sgp4fix identify constants and allow alternate values no longer needed
   	# getgravconst( whichconst, tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 );
    ss     = 78.0 / satrec.radiusearthkm + 1.0;
    #  sgp4fix use multiply for speed instead of pow
    qzms2ttemp = (120.0 - 78.0) / satrec.radiusearthkm;
    qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp;
    x2o3   =  2.0 / 3.0;
   
    satrec.init = np.chararray(nsat)
    satrec.init[:] = b'y';
    satrec.method = np.chararray(nsat)
    satrec.method[:] = b'n';
    satrec.t	 = np.zeros(nsat);
   
   # sgp4fix remove satn as it is not needed in initl
    (
      satrec.no_unkozai,
      ainv,  ao,    satrec.con41,  con42, cosio,
      cosio2,eccsq, omeosq, posq,
      rp,    rteosq,sinio , satrec.gsto,
      ) = _initl_np(
          satrec.xke, satrec.j2, satrec.ecco, satrec.jdsatepoch-2433281.5+satrec.jdsatepochF, satrec.inclo, satrec.no_kozai,
          satrec.operationmode
        );
    satrec.a    = np.power( satrec.no_unkozai*satrec.tumin , (-2.0/3.0) );
    satrec.alta = satrec.a*(1.0 + satrec.ecco) - 1.0;
    satrec.altp = satrec.a*(1.0 - satrec.ecco) - 1.0;
   
    """
    // sgp4fix remove this check as it is unnecessary
    // the mrt check in sgp4 handles decaying satellite cases even if the starting
    // condition is below the surface of te earth
    //     if (rp < 1.0)
    //       {
    //         printf("# *** satn%d epoch elts sub-orbital ***\n", satn);
    //         satrec.error = 5;
    //       }
    """
    
    
    satrec.isimp   = np.zeros(nsat,dtype=int)
    satrec.eta     = np.zeros(nsat)
    satrec.cc1     = np.zeros(nsat)
    satrec.x1mth2  = np.zeros(nsat)
    satrec.cc4     = np.zeros(nsat)
    satrec.cc5     = np.zeros(nsat)
    satrec.mdot    = np.zeros(nsat)
    satrec.argpdot = np.zeros(nsat)
    satrec.nodedot = np.zeros(nsat)
    satrec.omgcof  = np.zeros(nsat)
    satrec.xmcof   = np.zeros(nsat)
    satrec.nodecf  = np.zeros(nsat)
    satrec.t2cof   = np.zeros(nsat)
    satrec.xlcof   = np.zeros(nsat)
    satrec.aycof   = np.zeros(nsat)
    satrec.delmo   = np.zeros(nsat)
    satrec.sinmao  = np.zeros(nsat)
    satrec.x7thm1  = np.zeros(nsat)
    satrec.pinco   = np.zeros(nsat)
    satrec.e3      = np.zeros(nsat)
    satrec.ee2     = np.zeros(nsat) 
    satrec.peo     = np.zeros(nsat)
    satrec.pgho    = np.zeros(nsat)
    satrec.pho     = np.zeros(nsat)
    satrec.plo     = np.zeros(nsat)
    satrec.se2     = np.zeros(nsat)
    satrec.se3     = np.zeros(nsat)
    satrec.sgh2    = np.zeros(nsat)
    satrec.sgh3    = np.zeros(nsat)
    satrec.sgh4    = np.zeros(nsat)
    satrec.sh2     = np.zeros(nsat)
    satrec.sh3     = np.zeros(nsat)
    satrec.si2     = np.zeros(nsat)
    satrec.si3     = np.zeros(nsat)
    satrec.sl2     = np.zeros(nsat)
    satrec.sl3     = np.zeros(nsat)
    satrec.sl4     = np.zeros(nsat)
    satrec.xgh2    = np.zeros(nsat)
    satrec.xgh3    = np.zeros(nsat)
    satrec.xgh4    = np.zeros(nsat)
    satrec.xh2     = np.zeros(nsat)
    satrec.xh3     = np.zeros(nsat)
    satrec.xi2     = np.zeros(nsat)
    satrec.xi3     = np.zeros(nsat)
    satrec.xl2     = np.zeros(nsat)
    satrec.xl3     = np.zeros(nsat)
    satrec.xl4     = np.zeros(nsat)
    satrec.zmol    = np.zeros(nsat)
    satrec.zmos    = np.zeros(nsat)
    satrec.irez    = np.zeros(nsat,dtype=int)
    satrec.atime   = np.zeros(nsat)
    satrec.d2201   = np.zeros(nsat)
    satrec.d2211   = np.zeros(nsat)
    satrec.d3210   = np.zeros(nsat)
    satrec.d3222   = np.zeros(nsat)
    satrec.d4410   = np.zeros(nsat)
    satrec.d4422   = np.zeros(nsat)
    satrec.d5220   = np.zeros(nsat)
    satrec.d5232   = np.zeros(nsat)
    satrec.d5421   = np.zeros(nsat)
    satrec.d5433   = np.zeros(nsat)
    satrec.dedt    = np.zeros(nsat)
    satrec.didt    = np.zeros(nsat)
    satrec.dmdt    = np.zeros(nsat)
    satrec.dnodt   = np.zeros(nsat)
    satrec.domdt   = np.zeros(nsat)
    satrec.del1    = np.zeros(nsat)
    satrec.del2    = np.zeros(nsat)
    satrec.del3    = np.zeros(nsat)
    satrec.xfact   = np.zeros(nsat)
    satrec.xlamo   = np.zeros(nsat)
    satrec.xli     = np.zeros(nsat)
    satrec.xni     = np.zeros(nsat)
    satrec.d2      = np.zeros(nsat) 
    satrec.d3      = np.zeros(nsat)
    satrec.d4      = np.zeros(nsat)
    satrec.t3cof   = np.zeros(nsat)
    satrec.t4cof   = np.zeros(nsat) 
    satrec.t5cof   = np.zeros(nsat)
    xpidot         = np.zeros(nsat)
    
    bomnoun = (omeosq >= 0.0) | (satrec.no_unkozai >= 0.0)
    if np.any( bomnoun ) :
        #len_bomnoun = np.count_nonzero( bomnoun )
        satrec.isimp[bomnoun & (rp < 220.0 / satrec.radiusearthkm + 1.0)] = 1
        
        sfour_  = ss[bomnoun];
        qzms24_ = qzms2t[bomnoun];
        perige_ = (rp[bomnoun] - 1.0) * satrec.radiusearthkm[bomnoun];
        sfour_[perige_ < 156.0] = perige_[perige_ < 156.0] - 78.0;
        sfour_[perige_ < 98.0] = 20
        
        qzms24_[perige_ < 156.0] = np.power((120.0 - sfour_[perige_ < 156.0]) / satrec.radiusearthkm[bomnoun][perige_ < 156.0],4)
        sfour_[perige_ < 156.0] = sfour_[perige_ < 156.0] / satrec.radiusearthkm[bomnoun][perige_ < 156.0] + 1.0;
        
        pinvsq_ = 1.0 / posq[bomnoun];
        
        tsi_  = 1.0 / (ao[bomnoun] - sfour_);
        
        satrec.eta[bomnoun]  = ao[bomnoun] * satrec.ecco[bomnoun] * tsi_;
        
        etasq_ = satrec.eta[bomnoun] * satrec.eta[bomnoun];
        eeta_  = satrec.ecco[bomnoun] * satrec.eta[bomnoun];
        
        psisq_ = np.fabs(1.0 - etasq_);
        coef_  = qzms24_ * np.power(tsi_, 4.0);
        coef1_ = coef_ / np.power(psisq_, 3.5);
        
        cc2_   = coef1_ * satrec.no_unkozai[bomnoun] * (ao[bomnoun] * (1.0 + 1.5 * etasq_ + eeta_ *
                           (4.0 + etasq_)) + 0.375 * satrec.j2[bomnoun] * tsi_ / psisq_ * satrec.con41[bomnoun] *
                           (8.0 + 3.0 * etasq_ * (8.0 + etasq_)));
        
        satrec.cc1[bomnoun] = satrec.bstar[bomnoun] * cc2_;
        
        cc3_ = np.where( satrec.ecco[bomnoun] > 1.0e-4, 
                        -2.0 * coef_ * tsi_ * satrec.j3oj2[bomnoun] * satrec.no_unkozai[bomnoun] * sinio[bomnoun] / satrec.ecco[bomnoun], 
                        0 );
        
        satrec.x1mth2[bomnoun] = 1.0 - cosio2[bomnoun];
        satrec.cc4[bomnoun]    = 2.0* satrec.no_unkozai[bomnoun] * coef1_ * ao[bomnoun] * omeosq[bomnoun] * \
                          (satrec.eta[bomnoun] * (2.0 + 0.5 * etasq_) + satrec.ecco[bomnoun] *
                          (0.5 + 2.0 * etasq_) - satrec.j2[bomnoun] * tsi_ / (ao[bomnoun] * psisq_) *
                          (-3.0 * satrec.con41[bomnoun] * (1.0 - 2.0 * eeta_ + etasq_ *
                          (1.5 - 0.5 * eeta_)) + 0.75 * satrec.x1mth2[bomnoun] *
                          (2.0 * etasq_ - eeta_ * (1.0 + etasq_)) * np.cos(2.0 * satrec.argpo[bomnoun])));
                          
        satrec.cc5[bomnoun] = 2.0 * coef1_ * ao[bomnoun] * omeosq[bomnoun] * (1.0 + 2.75 *
                       (etasq_ + eeta_) + eeta_ * etasq_);
        
        cosio4_ = cosio2[bomnoun] * cosio2[bomnoun];
        temp1_  = 1.5 * satrec.j2[bomnoun] * pinvsq_ * satrec.no_unkozai[bomnoun];
        temp2_  = 0.5 * temp1_ * satrec.j2[bomnoun] * pinvsq_;
        temp3_  = -0.46875 * satrec.j4[bomnoun] * pinvsq_ * pinvsq_ * satrec.no_unkozai[bomnoun];
        satrec.mdot[bomnoun]     = satrec.no_unkozai[bomnoun] + 0.5 * temp1_ * rteosq[bomnoun] * satrec.con41[bomnoun] + 0.0625 * \
                           temp2_ * rteosq[bomnoun] * (13.0 - 78.0 * cosio2[bomnoun] + 137.0 * cosio4_);
        satrec.argpdot[bomnoun]  = (-0.5 * temp1_ * con42[bomnoun] + 0.0625 * temp2_ *
                            (7.0 - 114.0 * cosio2[bomnoun] + 395.0 * cosio4_) +
                            temp3_ * (3.0 - 36.0 * cosio2[bomnoun] + 49.0 * cosio4_));
        xhdot1_            = -temp1_ * cosio[bomnoun];
        satrec.nodedot[bomnoun] = xhdot1_ + (0.5 * temp2_ * (4.0 - 19.0 * cosio2[bomnoun]) +
                             2.0 * temp3_ * (3.0 - 7.0 * cosio2[bomnoun])) * cosio[bomnoun];
        
        xpidot[bomnoun]         =  satrec.argpdot[bomnoun]+ satrec.nodedot[bomnoun];
        satrec.omgcof[bomnoun]   = satrec.bstar[bomnoun] * cc3_ * np.cos(satrec.argpo[bomnoun]);
        
        satrec.xmcof[bomnoun] = np.where( satrec.ecco[bomnoun] > 1.0e-4, 
                                         -x2o3 * coef_ * satrec.bstar[bomnoun] / eeta_,
                                         0.0 );
        
        satrec.nodecf[bomnoun] = 3.5 * omeosq[bomnoun] * xhdot1_ * satrec.cc1[bomnoun];
        satrec.t2cof[bomnoun]   = 1.5 * satrec.cc1[bomnoun];
        
        temp5_ = np.where( np.fabs(cosio[bomnoun]+1.0) > 1.5e-12, (1.0 + cosio[bomnoun]), temp4 )
        satrec.xlcof[bomnoun] = -0.25 * satrec.j3oj2[bomnoun] * sinio[bomnoun] * (3.0 + 5.0 * cosio[bomnoun]) / temp5_ 
       
        satrec.aycof[bomnoun]   = -0.5 * satrec.j3oj2[bomnoun] * sinio[bomnoun];
        satrec.delmo[bomnoun] = np.power(1.0 + satrec.eta[bomnoun] * np.cos(satrec.mo[bomnoun]), 3);
        satrec.sinmao[bomnoun]  = np.sin(satrec.mo[bomnoun]);
        satrec.x7thm1[bomnoun]  = 7.0 * cosio2[bomnoun] - 1.0;
        
        bomnound = bomnoun & (2*pi / satrec.no_unkozai >= 225.0)
        if np.any( bomnound ) :
            len_bomnound = np.count_nonzero( bomnound )
            satrec.method[bomnound] = b'd';
            satrec.isimp[bomnound]  = 1;
            tc__    =  np.zeros(len_bomnound);
            inclm__ = satrec.inclo[bomnound];
            
            (
                snodm, cnodm, sinim,  cosim, sinomm,
                cosomm,day,   satrec.e3[bomnound],     satrec.ee2[bomnound],   em,
                emsq,  gam,   satrec.peo[bomnound],    satrec.pgho[bomnound],  satrec.pho[bomnound],
                satrec.pinco[bomnound], satrec.plo[bomnound],   rtemsq, 
                satrec.se2[bomnound],   satrec.se3[bomnound],
                satrec.sgh2[bomnound], satrec.sgh3[bomnound], satrec.sgh4[bomnound], 
                satrec.sh2[bomnound], satrec.sh3[bomnound],
                satrec.si2[bomnound],  satrec.si3[bomnound],  satrec.sl2[bomnound], 
                satrec.sl3[bomnound], satrec.sl4[bomnound],
                s1,    s2,    s3,     s4,    s5,
                s6,    s7,    ss1,    ss2,   ss3,
                ss4,   ss5,   ss6,    ss7,   sz1,
                sz2,   sz3,   sz11,   sz12,  sz13,
                sz21,  sz22,  sz23,   sz31,  sz32,
                sz33,  satrec.xgh2[bomnound], satrec.xgh3[bomnound], 
                satrec.xgh4[bomnound], satrec.xh2[bomnound],
                satrec.xh3[bomnound], satrec.xi2[bomnound], satrec.xi3[bomnound],
                satrec.xl2[bomnound], satrec.xl3[bomnound],
                satrec.xl4[bomnound],   nm,    z1,     z2,    z3,
                z11,   z12,   z13,    z21,   z22,
                z23,   z31,   z32,    z33,   satrec.zmol[bomnound],
                satrec.zmos[bomnound]
            ) = _dscom_np(
                  satrec.jdsatepoch[bomnound]-2433281.5+satrec.jdsatepochF[bomnound], satrec.ecco[bomnound], satrec.argpo[bomnound], tc__,
                  satrec.inclo[bomnound], satrec.nodeo[bomnound],
                  satrec.no_unkozai[bomnound],
                  satrec.e3[bomnound], satrec.ee2[bomnound],
                  satrec.peo[bomnound],  satrec.pgho[bomnound],   
                  satrec.pho[bomnound], satrec.pinco[bomnound],
                  satrec.plo[bomnound], satrec.se2[bomnound], satrec.se3[bomnound],
                  satrec.sgh2[bomnound], satrec.sgh3[bomnound],   satrec.sgh4[bomnound],
                  satrec.sh2[bomnound],  satrec.sh3[bomnound],    
                  satrec.si2[bomnound], satrec.si3[bomnound],
                  satrec.sl2[bomnound],  satrec.sl3[bomnound],    satrec.sl4[bomnound],
                  satrec.xgh2[bomnound], satrec.xgh3[bomnound],   
                  satrec.xgh4[bomnound], satrec.xh2[bomnound],
                  satrec.xh3[bomnound],  satrec.xi2[bomnound],    
                  satrec.xi3[bomnound],  satrec.xl2[bomnound],
                  satrec.xl3[bomnound],  satrec.xl4[bomnound],
                  satrec.zmol[bomnound], satrec.zmos[bomnound]
                );
            (satrec.ecco[bomnound], satrec.inclo[bomnound], satrec.nodeo[bomnound], 
             satrec.argpo[bomnound], satrec.mo[bomnound]
             ) = _dpper_np(
                  satrec, bomnound, inclm__, satrec.init[bomnound],
                  satrec.ecco[bomnound], satrec.inclo[bomnound], satrec.nodeo[bomnound], 
                  satrec.argpo[bomnound], satrec.mo[bomnound], satrec.operationmode
                );
            argpm  = np.zeros(len_bomnound);
            nodem  = np.zeros(len_bomnound);
            mm     = np.zeros(len_bomnound);
            
            (
                 em,    argpm,  inclm, mm,
                 nm,    nodem,
                 satrec.irez[bomnound],  satrec.atime[bomnound],
                 satrec.d2201[bomnound], satrec.d2211[bomnound], 
                 satrec.d3210[bomnound], satrec.d3222[bomnound],
                 satrec.d4410[bomnound], satrec.d4422[bomnound], 
                 satrec.d5220[bomnound], satrec.d5232[bomnound],
                 satrec.d5421[bomnound], satrec.d5433[bomnound], 
                 satrec.dedt[bomnound],  satrec.didt[bomnound],
                 satrec.dmdt[bomnound], dndt,  satrec.dnodt[bomnound], 
                 satrec.domdt[bomnound],
                 satrec.del1[bomnound],  satrec.del2[bomnound],  
                 satrec.del3[bomnound],  satrec.xfact[bomnound],
                 satrec.xlamo[bomnound], satrec.xli[bomnound],   satrec.xni[bomnound]
             ) = _dsinit_np(
                   satrec.xke[bomnound],
                   cosim, emsq, satrec.argpo[bomnound], s1, s2, s3, s4, s5,
                   sinim, ss1, ss2, ss3, ss4,
                   ss5, sz1, sz3, sz11, sz13, sz21, sz23, sz31, sz33, satrec.t[bomnound], tc__,
                   satrec.gsto[bomnound], satrec.mo[bomnound], satrec.mdot[bomnound], 
                   satrec.no_unkozai[bomnound], satrec.nodeo[bomnound],
                   satrec.nodedot[bomnound], xpidot[bomnound], z1, z3, z11, z13, z21, z23, z31, z33,
                   satrec.ecco[bomnound], eccsq[bomnound], em, argpm, inclm__, mm, nm, nodem,
                   satrec.irez[bomnound],  satrec.atime[bomnound],
                   satrec.d2201[bomnound], satrec.d2211[bomnound], 
                   satrec.d3210[bomnound], satrec.d3222[bomnound] ,
                   satrec.d4410[bomnound], satrec.d4422[bomnound], 
                   satrec.d5220[bomnound], satrec.d5232[bomnound],
                   satrec.d5421[bomnound], satrec.d5433[bomnound],
                   satrec.dedt[bomnound],  satrec.didt[bomnound],
                   satrec.dmdt[bomnound],  satrec.dnodt[bomnound], satrec.domdt[bomnound],
                   satrec.del1[bomnound],  satrec.del2[bomnound],
                   satrec.del3[bomnound],  satrec.xfact[bomnound],
                   satrec.xlamo[bomnound], satrec.xli[bomnound],   satrec.xni[bomnound]
                 );
        bomnound = bomnoun & (satrec.isimp != 1)
        bt = (satrec.isimp != 1)[bomnoun]
        if np.any( bomnound ) :
            cc1sq_          = satrec.cc1[bomnound] * satrec.cc1[bomnound];
            satrec.d2[bomnound]    = 4.0 * ao[bomnound] * tsi_[bt] * cc1sq_;
            temp_           = satrec.d2[bomnound] * tsi_[bt] * satrec.cc1[bomnound] / 3.0;
            satrec.d3[bomnound]    = (17.0 * ao[bomnound] + sfour_[bt]) * temp_;
            satrec.d4[bomnound]    = 0.5 * temp_ * ao[bomnound] * tsi_[bt] * (221.0 * ao[bomnound] + 31.0 * sfour_[bt]) * satrec.cc1[bomnound];
            satrec.t3cof[bomnound] = satrec.d2[bomnound] + 2.0 * cc1sq_;
            satrec.t4cof[bomnound] = 0.25 * (3.0 * satrec.d3[bomnound] + satrec.cc1[bomnound] *
                            (12.0 * satrec.d2[bomnound] + 10.0 * cc1sq_));
            satrec.t5cof[bomnound] = 0.2 * (3.0 * satrec.d4[bomnound] +
                            12.0 * satrec.cc1[bomnound] * satrec.d3[bomnound] +
                            6.0 * satrec.d2[bomnound] * satrec.d2[bomnound] +
                            15.0 * cc1sq_ * (2.0 * satrec.d2[bomnound] + cc1sq_))
    
    sgp4_np(satrec, 0.0);
    
    satrec.init[:] = b'n';

     # sgp4fix return boolean. satrec.error contains any error codes
    return true;
         
"""
/*-----------------------------------------------------------------------------
*
*                             procedure sgp4
*
*  this procedure is the sgp4 prediction model from space command. this is an
*    updated and combined version of sgp4 and sdp4, which were originally
*    published separately in spacetrack report #3. this version follows the
*    methodology from the aiaa paper (2006) describing the history and
*    development of the code.
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    satrec	 - initialised structure from sgp4init() call.
*    tsince	 - time eince epoch (minutes)
*
*  outputs       :
*    r           - position vector                     km
*    v           - velocity                            km/sec
*  return code - non-zero on error.
*                   1 - mean elements, ecc >= 1.0 or ecc < -0.001 or a < 0.95 er
*                   2 - mean motion less than 0.0
*                   3 - pert elements, ecc < 0.0  or  ecc > 1.0
*                   4 - semi-latus rectum < 0.0
*                   5 - epoch elements are sub-orbital
*                   6 - satellite has decayed
*
*  locals        :
*    am          -
*    axnl, aynl        -
*    betal       -
*    cosim   , sinim   , cosomm  , sinomm  , cnod    , snod    , cos2u   ,
*    sin2u   , coseo1  , sineo1  , cosi    , sini    , cosip   , sinip   ,
*    cosisq  , cossu   , sinsu   , cosu    , sinu
*    delm        -
*    delomg      -
*    dndt        -
*    eccm        -
*    emsq        -
*    ecose       -
*    el2         -
*    eo1         -
*    eccp        -
*    esine       -
*    argpm       -
*    argpp       -
*    omgadf      -
*    pl          -
*    r           -
*    rtemsq      -
*    rdotl       -
*    rl          -
*    rvdot       -
*    rvdotl      -
*    su          -
*    t2  , t3   , t4    , tc
*    tem5, temp , temp1 , temp2  , tempa  , tempe  , templ
*    u   , ux   , uy    , uz     , vx     , vy     , vz
*    inclm       - inclination
*    mm          - mean anomaly
*    nm          - mean motion
*    nodem       - right asc of ascending node
*    xinc        -
*    xincp       -
*    xl          -
*    xlm         -
*    mp          -
*    xmdf        -
*    xmx         -
*    xmy         -
*    nodedf      -
*    xnode       -
*    nodep       -
*    np          -
*
*  coupling      :
*    getgravconst-
*    dpper
*    dpspace
*
*  references    :
*    hoots, roehrich, norad spacetrack report #3 1980
*    hoots, norad spacetrack report #6 1986
*    hoots, schumacher and glover 2004
*    vallado, crawford, hujsak, kelso  2006
  ----------------------------------------------------------------------------*/
"""

def sgp4_np(satrec, tsince):

    nsat = len( satrec.satnum ) 
    mrt = np.zeros(nsat)
    ux = np.zeros(nsat)
    uy = np.zeros(nsat)
    uz = np.zeros(nsat)
    vx = np.zeros(nsat)
    vy = np.zeros(nsat)
    vz = np.zeros(nsat)
    mvt= np.zeros(nsat)
    rvdot = np.zeros(nsat)
    
    satrec.om = np.zeros(nsat)

    """
    /* ------------------ set mathematical constants --------------- */
    // sgp4fix divisor for divide by zero check on inclination
    // the old check used 1.0 + cos(pi-1.0e-9), but then compared it to
    // 1.5 e-12, so the threshold was changed to 1.5e-12 for consistency
    """
    temp4 =   1.5e-12;
    twopi = 2.0 * pi;
    x2o3  = 2.0 / 3.0;
    #  sgp4fix identify constants and allow alternate values
    # tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 = whichconst
    vkmpersec     = satrec.radiusearthkm * satrec.xke/60.0;

    #  --------------------- clear sgp4 error flag -----------------
    satrec.t[:]     = tsince;
    satrec.error.fill(0);
    satrec.error_message = ''

    #  ------- update for secular gravity and atmospheric drag -----
    xmdf    = satrec.mo + satrec.mdot * satrec.t;
    argpdf  = satrec.argpo + satrec.argpdot * satrec.t;
    nodedf  = satrec.nodeo + satrec.nodedot * satrec.t;
    argpm   = argpdf.copy();
    mm      = xmdf.copy();
    t2      = satrec.t * satrec.t;
    nodem   = nodedf + satrec.nodecf * t2;
    tempa   = 1.0 - satrec.cc1 * satrec.t;
    tempe   = satrec.bstar * satrec.cc4 * satrec.t;
    templ   = satrec.t2cof * t2;
    
    bosimp = (satrec.isimp != 1)
    if np.any(bosimp) :

        delomg_ = satrec.omgcof[bosimp] * satrec.t[bosimp];
        #  sgp4fix use mutliply for speed instead of pow
        delmtemp_ =  1.0 + satrec.eta[bosimp] * np.cos(xmdf[bosimp]);
        delm_   = satrec.xmcof[bosimp] * \
                  (delmtemp_ * delmtemp_ * delmtemp_ -
                  satrec.delmo[bosimp]);
        temp_   = delomg_ + delm_;
        mm[bosimp]     = xmdf[bosimp] + temp_;
        argpm[bosimp]  = argpdf[bosimp] - temp_;
        t3_     = t2[bosimp] * satrec.t[bosimp];
        t4_     = t3_ * satrec.t[bosimp];
        tempa[bosimp]  += - satrec.d2[bosimp] * t2[bosimp] - satrec.d3[bosimp] * t3_ - \
                          satrec.d4[bosimp] * t4_;
        tempe[bosimp]  += satrec.bstar[bosimp] * satrec.cc5[bosimp] * (np.sin(mm[bosimp]) -
                          satrec.sinmao[bosimp]);
        templ[bosimp]  += satrec.t3cof[bosimp] * t3_ + t4_ * (satrec.t4cof[bosimp] +
                          satrec.t[bosimp] * satrec.t5cof[bosimp]);

    nm    = satrec.no_unkozai.copy();
    em    = satrec.ecco.copy();
    inclm = satrec.inclo.copy();
    bomd = (satrec.method == b'd')
    if np.any(bomd) :

         tc = satrec.t[bomd];
         (
             atime, em[bomd],    argpm[bomd],  inclm[bomd], xli,
             mm[bomd],    xni,   nodem[bomd],  dndt,  nm[bomd],
         ) = _dspace_np(
               satrec.irez[bomd],
               satrec.d2201[bomd], satrec.d2211[bomd], satrec.d3210[bomd],
               satrec.d3222[bomd], satrec.d4410[bomd], satrec.d4422[bomd],
               satrec.d5220[bomd], satrec.d5232[bomd], satrec.d5421[bomd],
               satrec.d5433[bomd], satrec.dedt[bomd],  satrec.del1[bomd],
               satrec.del2[bomd],  satrec.del3[bomd],  satrec.didt[bomd],
               satrec.dmdt[bomd],  satrec.dnodt[bomd], satrec.domdt[bomd],
               satrec.argpo[bomd], satrec.argpdot[bomd], satrec.t[bomd], tc,
               satrec.gsto[bomd], satrec.xfact[bomd], satrec.xlamo[bomd],
               satrec.no_unkozai[bomd], satrec.atime[bomd],
               em[bomd], argpm[bomd], inclm[bomd], satrec.xli[bomd], mm[bomd], satrec.xni[bomd],
               nodem[bomd], nm[bomd]
             );

    boer = (nm <= 0.0)
    #satrec.error_message[boer ] = 'mean motion {0:f} is less than zero'.format(nm)
    
    satrec.error[boer] = 2
    bonoer = (satrec.error == 0)
    
    am = np.zeros(nsat)
    am[bonoer] = np.power((satrec.xke[bonoer] / nm[bonoer]),x2o3) * tempa[bonoer] * tempa[bonoer];
    nm[bonoer] = satrec.xke[bonoer] / np.power(am[bonoer], 1.5);
    em[bonoer] = em[bonoer] - tempe[bonoer];

    #  fix tolerance for error recognition
    #  sgp4fix am is fixed from the previous nm check
    boer = ((em >= 1.0) | (em < -0.001)) & bonoer
    #satrec.error_message[boer] = 'mean eccentricity {0:f} not within range 0.0 <= e < 1.0'.format(em)
    
    satrec.error[boer] = 1;
    
    bonoer = (satrec.error == 0)

    #  sgp4fix fix tolerance to avoid a divide by zero
    xlm = np.zeros(nsat)
    emsq = np.zeros(nsat)
    temp = np.zeros(nsat)
    
    em[(em < 1.0e-6) & bonoer] = 1.0e-6
    mm[bonoer]     = mm[bonoer] + satrec.no_unkozai[bonoer] * templ[bonoer];
    xlm[bonoer]    = mm[bonoer] + argpm[bonoer] + nodem[bonoer];
    emsq[bonoer]   = em[bonoer] * em[bonoer];
    temp[bonoer]   = 1.0 - emsq[bonoer];

    nodem[bonoer] = np.where( nodem[bonoer] >= 0.0 , nodem[bonoer] % twopi, -(-nodem[bonoer] % twopi) )    
    argpm  = argpm % twopi
    xlm    = xlm % twopi
    mm[bonoer]     = (xlm[bonoer] - argpm[bonoer] - nodem[bonoer]) % twopi

     # sgp4fix recover singly averaged mean elements
    satrec.am[bonoer] = am[bonoer];
    satrec.em[bonoer] = em[bonoer];
    satrec.im[bonoer] = inclm[bonoer];
    satrec.Om[bonoer] = nodem[bonoer];
    satrec.om[bonoer] = argpm[bonoer];
    satrec.mm[bonoer] = mm[bonoer];
    satrec.nm[bonoer] = nm[bonoer];

     #  ----------------- compute extra mean quantities -------------
    sinim = np.zeros(nsat)
    cosim = np.zeros(nsat)
    sinim[bonoer] = np.sin(inclm[bonoer]);
    cosim[bonoer] = np.cos(inclm[bonoer]);

    #  -------------------- add lunar-solar periodics --------------
    ep = np.zeros(nsat)
    xincp = np.zeros(nsat)
    argpp = np.zeros(nsat)
    nodep = np.zeros(nsat)
    mp    = np.zeros(nsat)
    sinip = np.zeros(nsat)
    cosip = np.zeros(nsat)
    
    ep[bonoer]     = em[bonoer];
    xincp[bonoer]  = inclm[bonoer];
    argpp[bonoer]  = argpm[bonoer];
    nodep[bonoer]  = nodem[bonoer];
    mp[bonoer]     = mm[bonoer];
    sinip[bonoer]  = sinim[bonoer];
    cosip[bonoer]  = cosim[bonoer];
    bomd = bomd & bonoer;
    if np.any(bomd) :

        ep[bomd], xincp[bomd], nodep[bomd], argpp[bomd], mp[bomd] = _dpper_np(
               satrec, bomd, satrec.inclo[bomd],
               b'n', ep[bomd], xincp[bomd], nodep[bomd], argpp[bomd], mp[bomd], satrec.operationmode[bomd] );
        boncp = bomd & (xincp < 0.0)
        if np.any( boncp ) :
            xincp[boncp] = -xincp[boncp]
            nodep[boncp] += np.pi
            argpp[boncp] -= np.pi
            
        boer = bomd & ( (ep < 0.0) | (ep > 1.0) )
        #satrec.error_message[boer] = ('perturbed eccentricity {0:f} not within'
        #                             ' range 0.0 <= e <= 1.0'.format(ep))
        satrec.error[boer] = 3;
        bonoer = (satrec.error == 0)
        bomd = bomd & bonoer;

     #  -------------------- long period periodics ------------------

        sinip[bomd] =  np.sin(xincp[bomd]);
        cosip[bomd] =  np.cos(xincp[bomd]);
        satrec.aycof[bomd] = -0.5*satrec.j3oj2[bomd]*sinip[bomd];
         #  sgp4fix for divide by zero for xincp = 180 deg
        bosip = np.fabs(cosip+1.0) > temp4
        bosipmd = bomd & bosip
        if np.any( bosipmd ) :
            satrec.xlcof[bosipmd] = -0.25 * satrec.j3oj2[bosipmd] * sinip[bosipmd] * (3.0 + 5.0 * cosip[bosipmd]) / (1.0 + cosip[bosipmd]);
            
        bosipmd = bomd & np.logical_not(bosip)
        if np.any( bosipmd ) :
            satrec.xlcof[bosipmd] = -0.25 * satrec.j3oj2[bosipmd] * sinip[bosipmd] * (3.0 + 5.0 * cosip[bosipmd]) / temp4;
    
    axnl = np.zeros(nsat)
    aynl = np.zeros(nsat)
    xl   = np.zeros(nsat)

    axnl[bonoer] = ep[bonoer] * np.cos(argpp[bonoer]);
    temp[bonoer] = 1.0 / (am[bonoer] * (1.0 - ep[bonoer] * ep[bonoer]));
    aynl[bonoer] = ep[bonoer]* np.sin(argpp[bonoer]) + temp[bonoer] * satrec.aycof[bonoer];
    xl[bonoer]   = mp[bonoer] + argpp[bonoer] + nodep[bonoer] + temp[bonoer] * satrec.xlcof[bonoer] * axnl[bonoer];
     #  --------------------- solve kepler's equation ---------------
    u = np.zeros(nsat)
    tem5 = np.zeros(nsat)
    u[bonoer]    = (xl[bonoer] - nodep[bonoer]) % twopi
    tem5[bonoer] = 9999.9;
    coseo1 = np.zeros(nsat)
    sineo1 = np.zeros(nsat)
    eo1 = np.zeros(nsat)
    eo1[bonoer]  = u[bonoer]
    ktr = 1;
     #    sgp4fix for kepler iteration
     #    the following iteration needs better limits on corrections
    boit = (np.fabs(tem5) >= 1.0e-12) & bonoer
    while np.any(boit) and ktr <= 10:
        
        sineo1[boit] = np.sin(eo1[boit]);
        coseo1[boit] = np.cos(eo1[boit]);
        tem5[boit]  = (1.0 - coseo1[boit] * axnl[boit] - sineo1[boit] * aynl[boit])
        tem5[boit]   = (u[boit] - aynl[boit] * coseo1[boit] + axnl[boit] * sineo1[boit] - eo1[boit]) / tem5[boit];
        tem5[tem5 >= 0.95] = 0.95
        tem5[tem5 <= -0.95] = -0.95
        eo1[boit]    += tem5[boit];
        ktr = ktr + 1;
        boit = (np.fabs(tem5) >= 1.0e-12) & bonoer

    #  ------------- short period preliminary quantities -----------
    ecose = np.zeros(nsat)
    esine = np.zeros(nsat)
    el2   = np.zeros(nsat)
    pl    = np.zeros(nsat)
    
    ecose[bonoer] = axnl[bonoer]*coseo1[bonoer] + aynl[bonoer]*sineo1[bonoer];
    esine[bonoer] = axnl[bonoer]*sineo1[bonoer] - aynl[bonoer]*coseo1[bonoer];
    el2[bonoer]   = axnl[bonoer]*axnl[bonoer] + aynl[bonoer]*aynl[bonoer];
    pl[bonoer]    = am[bonoer]*(1.0-el2[bonoer]);
    boer = (pl < 0.0) & bonoer
    #satrec.error_message[boer] = 'semilatus rectum {0:f} is less than zero'.format(pl)
    satrec.error[boer] = 4
    bonoer = (satrec.error == 0)
    boer = np.logical_not(boer) & bonoer

    if np.any(boer) :

        rl_     = am[boer] * (1.0 - ecose[boer]);
        rdotl_  = np.sqrt(am[boer]) * esine[boer]/rl_;
        rvdotl_ = np.sqrt(pl[boer]) / rl_;
        betal_  = np.sqrt(1.0 - el2[boer]);
        temp[boer]   = esine[boer] / (1.0 + betal_);
        sinu_   = am[boer] / rl_ * (sineo1[boer] - aynl[boer] - axnl[boer] * temp[boer]);
        cosu_   = am[boer] / rl_ * (coseo1[boer] - axnl[boer] + aynl[boer] * temp[boer]);
        su_     = np.arctan2(sinu_, cosu_);
        sin2u_  = (cosu_ + cosu_) * sinu_;
        cos2u_  = 1.0 - 2.0 * sinu_ * sinu_;
        temp[boer]   = 1.0 / pl[boer];
        temp1_  = 0.5 * satrec.j2[boer] * temp[boer];
        temp2_  = temp1_ * temp[boer];

         #  -------------- update for short period periodics ------------
        bt = (bomd & boer)
        if np.any(bt):
            cosisq_                 = cosip[bt] * cosip[bt];
            satrec.con41[bt]  = 3.0*cosisq_ - 1.0;
            satrec.x1mth2[bt] = 1.0 - cosisq_;
            satrec.x7thm1[bt] = 7.0*cosisq_ - 1.0;

        mrt[boer]   = rl_ * (1.0 - 1.5 * temp2_ * betal_ * satrec.con41[boer]) + \
                 0.5 * temp1_ * satrec.x1mth2[boer] * cos2u_;
        su_   -= 0.25 * temp2_ * satrec.x7thm1[boer] * sin2u_;
        xnode_ = nodep[boer] + 1.5 * temp2_ * cosip[boer] * sin2u_;
        xinc_  = xincp[boer] + 1.5 * temp2_ * cosip[boer] * sinip[boer] * cos2u_;
        mvt[boer]   = rdotl_ - nm[boer] * temp1_ * satrec.x1mth2[boer] * sin2u_ / satrec.xke[boer];
        rvdot[boer] = rvdotl_ + nm[boer] * temp1_ * (satrec.x1mth2[boer] * cos2u_ +
                 1.5 * satrec.con41[boer]) / satrec.xke[boer];

        #  --------------------- orientation vectors -------------------
        sinsu_ =  np.sin(su_);
        cossu_ =  np.cos(su_);
        snod_  =  np.sin(xnode_);
        cnod_  =  np.cos(xnode_);
        sini_  =  np.sin(xinc_);
        cosi_  =  np.cos(xinc_);
        xmx_   = -snod_ * cosi_;
        xmy_   =  cnod_ * cosi_;
        ux[boer]    =  xmx_ * sinsu_ + cnod_ * cossu_;
        uy[boer]    =  xmy_ * sinsu_ + snod_ * cossu_;
        uz[boer]    =  sini_ * sinsu_;
        vx[boer]    =  xmx_ * cossu_ - cnod_ * sinsu_;
        vy[boer]    =  xmy_ * cossu_ - snod_ * sinsu_;
        vz[boer]    =  sini_ * cossu_;
        
         #  --------- position and velocity (in km and km/sec) ----------
    _mr = np.zeros(nsat)
    _mr[bonoer] = mrt[bonoer] * satrec.radiusearthkm[bonoer]
    r = (_mr * ux, _mr * uy, _mr * uz)
    v = ((mvt * ux + rvdot * vx) * vkmpersec,
              (mvt * uy + rvdot * vy) * vkmpersec,
              (mvt * uz + rvdot * vz) * vkmpersec)

     #  sgp4fix for decaying satellites
    
    boer = (mrt < 1.0) & bonoer
    #satrec.error_message[boer] = ('mrt {0:f} is less than 1.0 indicating'
    #                             ' the satellite has decayed'.format(mrt))
    satrec.error[boer] = 6;

    return r, v;

"""
/* -----------------------------------------------------------------------------
*
*                           function gstime
*
*  this function finds the greenwich sidereal time.
*
*  author        : david vallado                  719-573-2600    1 mar 2001
*
*  inputs          description                    range / units
*    jdut1       - julian date in ut1             days from 4713 bc
*
*  outputs       :
*    gstime      - greenwich sidereal time        0 to 2pi rad
*
*  locals        :
*    temp        - temporary variable for doubles   rad
*    tut1        - julian centuries from the
*                  jan 1, 2000 12 h epoch (ut1)
*
*  coupling      :
*    none
*
*  references    :
*    vallado       2004, 191, eq 3-45
* --------------------------------------------------------------------------- */
"""

def gstime(jdut1):

     tut1 = (jdut1 - 2451545.0) / 36525.0;
     temp = -6.2e-6* tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + \
             (876600.0*3600 + 8640184.812866) * tut1 + 67310.54841;  #  sec
     temp = (temp * deg2rad / 240.0) % twopi # 360/86400 = 1/240, to deg, to rad

     #  ------------------------ check quadrants ---------------------
     if temp < 0.0:
         temp += twopi;

     return temp;
 
# The routine was originally marked private, so make it available under
# the old name for compatibility:
_gstime = gstime

"""
/* -----------------------------------------------------------------------------
*
*                           function getgravconst
*
*  this function gets constants for the propagator. note that mu is identified to
*    facilitiate comparisons with newer models. the common useage is wgs72.
*
*  author        : david vallado                  719-573-2600   21 jul 2006
*
*  inputs        :
*    whichconst  - which set of constants to use  wgs72old, wgs72, wgs84
*
*  outputs       :
*    tumin       - minutes in one time unit
*    mu          - earth gravitational parameter
*    radiusearthkm - radius of the earth in km
*    xke         - reciprocal of tumin
*    j2, j3, j4  - un-normalized zonal harmonic values
*    j3oj2       - j3 divided by j2
*
*  locals        :
*
*  coupling      :
*    none
*
*  references    :
*    norad spacetrack report #3
*    vallado, crawford, hujsak, kelso  2006
  --------------------------------------------------------------------------- */
"""

def getgravconst(whichconst):

       if whichconst == 'wgs72old':
           mu     = 398600.79964;        #  in km3 / s2
           radiusearthkm = 6378.135;     #  km
           xke    = 0.0743669161;
           tumin  = 1.0 / xke;
           j2     =   0.001082616;
           j3     =  -0.00000253881;
           j4     =  -0.00000165597;
           j3oj2  =  j3 / j2;

           #  ------------ wgs-72 constants ------------
       elif whichconst == 'wgs72':
           mu     = 398600.8;            #  in km3 / s2
           radiusearthkm = 6378.135;     #  km
           xke    = 60.0 / sqrt(radiusearthkm*radiusearthkm*radiusearthkm/mu);
           tumin  = 1.0 / xke;
           j2     =   0.001082616;
           j3     =  -0.00000253881;
           j4     =  -0.00000165597;
           j3oj2  =  j3 / j2;

       elif whichconst == 'wgs84':
           #  ------------ wgs-84 constants ------------
           mu     = 398600.5;            #  in km3 / s2
           radiusearthkm = 6378.137;     #  km
           xke    = 60.0 / sqrt(radiusearthkm*radiusearthkm*radiusearthkm/mu);
           tumin  = 1.0 / xke;
           j2     =   0.00108262998905;
           j3     =  -0.00000253215306;
           j4     =  -0.00000161098761;
           j3oj2  =  j3 / j2;

       return tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2
