import numpy as np

class Polynomial():
  
  def __init__(self, c):
    """
    
    Parameters:
    -----------
    c : 1D array-like
      Coefficients of the polynomial.
    """
    assert np.all(c != 0), "Can't all coefficients be 0!"
    
    self.c = np.trim_zeros(np.array(c), trim='f')
    self.cprime = self.c[:-1] * (np.arange(self.c.size,1,-1)-1)

  def coeff_(self):
    return self.c
    
  def pcoeff_(self):
    return selc.cprime
    
  def calc_poly(self, x, c):
    return np.polyval(c, x)
    
  def f(self, x):
    return self.calc_poly(x, self.c)

  def fprime(self, x):
    return self.calc_poly(x, self.cprime)
  
  def _get_str(self, c):
    s = ''
    fmt = ' {0} {1}x^{{{2}}}'
    e = np.arange(c.size,0,-1)-1

    for c_, e_ in zip(c, e):
      if c_ > 0: s += fmt.format('+', np.abs(c_), e_)
      if c_ < 0: s += fmt.format('-', np.abs(c_), e_)
    
    s = s.replace('x^{0}', '')  # Delete `x^{0}` from the strings
    s = s.replace('x^{1}', 'x') # Replace `x^{1}` with just `x`
    
    return s[3:] if c[0] > 0 else s[1:]
  
  def f_str(self):
    return self._get_str(self.c)
    
  def fprime_str(self):
    return self._get_str(self.cprime)
  
  def roots(self):
    return np.roots(self.c)