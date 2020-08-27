from colorama import Fore, Back, Style 

#To do - Expand the class to include other vector operations like dot and cross products, returning vector values at given time. 
class Vector():
    """Class to create a three dimensional vector object whose components are function of times"""	

    def __init__(self,t,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
    
        self._radial_dist = 0.
        self._theta = 0.
        self._phi = 0.
        self._mag = 0.
        
        
    @property
    def magnitude(self):
 	"""Returns the magnitude of the vector"""
        self._mag = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        return self._mag
    
    @property
    def radial_comp(self):
	"""Returns the radial coordinate"""
        self._radial_comp = self.mag
        return self._radial_comp
    
    @property
    def phi(self):
	"""Returns the phi coordinate"""
        self._phi = np.unwrap(np.arctan(self.y/self.x))
        return self._phi
            
    @property
    def theta(self):
	"""Returns the theta coordinate"""
        self._theta =  np.arccos(self.z/self.magnitude)
        return self._theta
    

    def __add__(self, vec2):
        xx = self.x + vec2.x
        yy = self.y + vec2.y
        zz = self.z + vec2.z
        return vector(self.t, xx, yy, zz)
        
    def __sub__(self, vec2):
        xx = self.x - vec2.x
        yy = self.y - vec2.y
        zz = self.z - vec2.z
        return vector(self.t, xx, yy, zz)
   
    def __mul__(self, alpha):
        xx = self.x * alpha
        yy = self.y * alpha
        zz = self.z * alpha
        return vector(self.t, xx, yy, zz)
   
    def __truediv__(self, alpha):
        assert(alpha!=0.),"Division by zero!"
        xx = self.x / alpha
        yy = self.y / alpha
        zz = self.z / alpha
        return vector(self.t, xx, yy, zz)
   
    def __pow__(self, alpha):
        xx = self.x ** alpha
        yy = self.y ** alpha
        zz = self.z ** alpha
        return vector(self.t, xx, yy, zz)


