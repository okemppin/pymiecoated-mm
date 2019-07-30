"""
Copyright (C) 2012-2013 Jussi Leinonen

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import array, arange, dot, zeros, vstack, sqrt, sin, cos
import numba
import numpy as np


def mie_props(coeffs,y):
    """The scattering properties.
    """
    anp = coeffs.an.real
    anpp = coeffs.an.imag
    bnp = coeffs.bn.real
    bnpp = coeffs.bn.imag
    nmax = coeffs.nmax

    n1 = nmax-1
    n = arange(1,nmax+1,dtype=float)
    cn = 2*n+1
    c1n = n*(n+2)/(n+1)
    c2n = cn/(n*(n+1))
    y2 = y**2

    dn = cn*(anp+bnp)
    q = dn.sum()
    qext = 2*q/y2

    en = cn*(anp**2+anpp**2+bnp**2+bnpp**2)
    q = en.sum()
    qsca = 2*q/y2
    qabs = qext-qsca

    fn = (coeffs.an-coeffs.bn)*cn
    gn=(-1)**n # this line takes third of the whole function exec time
    q = (fn*gn).sum()
    qb = dot(q,q.conj()).real/y2

    g1 = zeros((4,nmax),dtype=float)
    g1[:,:n1] = vstack((anp[1:nmax], anpp[1:nmax], bnp[1:nmax], bnpp[1:nmax]))

    asy1 = c1n*(anp*g1[0,:]+anpp*g1[1,:]+bnp*g1[2,:]+bnpp*g1[3,:])
    asy2 = c2n*(anp*bnp+anpp*bnpp)

    asy = 4/y2*(asy1+asy2).sum()/qsca
    qratio = qb/qsca

    return {"qext":qext, "qsca":qsca, "qabs":qabs, "qb":qb, "asy":asy, 
        "qratio":qratio}

#@numba.jit("complex64(complex128[:], float64[:])", nopython=True)

# issues with signature, so we just use automatic signature
@numba.jit(nopython=True)
def numba_dot(a, b):
  ret = 0 + 0j
  for i in range(len(a)):
    ret += a[i] * b[i]
  return ret

def mie_S12(coeffs,u):
  #return mie_S12old(coeffs,u) # use this to fall back to non-numba version
  return mie_S12_backend(coeffs.nmax, coeffs.an, coeffs.bn, u)

  # use these for self-tests
  #s1, s2 = mie_S12_backend(coeffs.nmax, coeffs.an, coeffs.bn, u)
  #return (np.complex128(s1), np.complex128(s2)) #required to comply with the self-test 

def mie_S12_pt(coeffs,pin, tin):
  return mie_S12_backend_pt(coeffs.nmax, coeffs.an, coeffs.bn, pin, tin)

@numba.jit(nopython=True)
def mie_S12_backend(nmax,an,bn,u):
    """
    Do note that pin and tin do not depend on the refractive index, and thus
    can be shared between runs of different mr, mi
    """
    pin = mie_p(u, nmax)
    tin = mie_t(u, nmax, pin)
    return mie_S12_backend_pt(nmax,an,bn,pin,tin)

@numba.jit(nopython=True)
def mie_S12_backend_pt(nmax,an,bn,pin, tin):
    """The amplitude scattering matrix.
    """


    s1 = numba_dot(an,pin)+numba_dot(bn,tin)
    s2 = numba_dot(an,tin)+numba_dot(bn,pin)
    return (s1, s2)

def mie_S12old(coeffs,u):
    """The amplitude scattering matrix.
    """
    (pin,tin) = mie_ptold(u,coeffs.nmax)
    n = arange(1, coeffs.nmax+1, dtype=float)
    n2 = (2*n+1)/(n*(n+1))
    pin *= n2
    tin *= n2

    s1 = dot(coeffs.an,pin)+dot(coeffs.bn,tin)
    s2 = dot(coeffs.an,tin)+dot(coeffs.bn,pin)
    return (s1, s2)

#@numba.jit("float64[:](float64, int64)", nopython=True)
@numba.jit(nopython=True)
def mie_p(u, nmax):
    #p = zeros(nmax, dtype=float)
    p = [0. for i in range(nmax)]
    p[0] = 1
    p[1] = 3*u
    #nn = arange(2,nmax,dtype=float)
    nn = [float(i) for i in range(2,nmax)]

    for n in nn:
        n_i = int(n)
        p[n_i] = (2*n+1)/n*p[n_i-1]*u - (n+1)/n*p[n_i-2]

    return p


"""
Numba version is faster than numpy version
"""
@numba.jit(nopython=True,fastmath=False)
def mie_t(u, nmax, p):
    #nn = arange(2,nmax,dtype=float)
    nn = [float(i) for i in range(2,nmax)]
    t = [0. for i in range(nmax)]
    #t = zeros(nmax, dtype=float)
    t[0] = u
    t[1] = 6*u**2 - 3
    for n in nn:
        n_i = int(n)
        t[n_i] = (n+1) * u * p[n_i] - (n+2) * p[n_i-1]
    #t[2:] = (nn+1)*u*p[2:] - (nn+2)*p[1:-1]
    return t


def mie_pt(u, nmax):
  return mie_ptnumba(u, nmax)

def mie_ptold(u,nmax):
    u = float(u)
    p = zeros(nmax, dtype=float)
    p[0] = 1
    p[1] = 3*u
    t = zeros(nmax, dtype=float)
    t[0] = u
    t[1] = 6*u**2 - 3

    nn = arange(2,nmax,dtype=float)

    for n in nn:
        n_i = int(n)
        p[n_i] = (2*n+1)/n*p[n_i-1]*u - (n+1)/n*p[n_i-2]
    
    t[2:] = (nn+1)*u*p[2:] - (nn+2)*p[1:-1]

    return (p,t)

@numba.jit(nopython=True)
def mie_ptnumba(u,nmax):
    u = float(u)

    p = mie_p(u, nmax)
    t = mie_t(u, nmax, p)
    n = [float(i) for i in range(1, nmax+1)]
    n2 = [(2 * ni + 1) / (ni * (ni + 1)) for ni in n]
    # faster to just store these n2-multiplied terms, since they only depend on nmax?
    # it is massively faster for size 0.01 ... 1000 range (this function is about 10x faster, total about 33%)
    for i in range(nmax):
      p[i] = p[i] * n2[i]
      t[i] = t[i] * n2[i]

    return (array(p),array(t))
