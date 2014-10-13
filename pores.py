import numpy as np
from numpy import array
from scipy.spatial import Delaunay
import itertools

def unique_rows(a):
    "Returns the indices of the unique rows of an array"
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return idx

def triangle_edges(r1, r2, r3):
    """
    Returns the length of the edges of a triangle with vertices at
    r1, r2, r3, with a box of size 1.
    
    a is across from r1 (between r2 and r3), b is across from r2, etc.
    """
    xt, yt, zt = array((r1, r2, r3, r1)).T
    dx, dy, dz = np.diff(xt,1), np.diff(yt,1), np.diff(zt,1)
    c, a, b = np.sqrt(dx**2 + dy**2 + dz**2).T
    return a, b, c

def midcirc(a,b,c,r1,r2,r3):
    """
    For three circles of radius r1, r2, r3, with distances between them of a,b,c
    (where a is across from r1, etc.), returns the radius of the circle that would fit in the middle
    """
    rad = ((-(r1*pow(a,4)) - r2*pow(b,4) + 
         pow(c,2)*((r1 - r3)*(-r2 + r3)*(r1 + r2 + 2*r3) - 
            r3*pow(c,2)) + pow(a,2)*
          (-((r1 - r2)*(r1 - r3)*(2*r1 + r2 + r3)) + 
            (r1 + r2)*pow(b,2) + (r1 + r3)*pow(c,2)) + 
         pow(b,2)*((r1 - r2)*(r2 - r3)*(r1 + 2*r2 + r3) + 
            (r2 + r3)*pow(c,2)) - 
         np.sqrt(-((a - b - c)*(a + b - c)*(a - b + c)*(a + b + c)*
             (c + r1 - r2)*(c - r1 + r2)*(b + r1 - r3)*(a + r2 - r3)*
             (b - r1 + r3)*(a - r2 + r3))))/
        (pow(a,4) + pow(b,4) - 2*pow(b,2)*
          (2*(r1 - r2)*(r2 - r3) + pow(c,2)) + 
         pow(c,2)*(4*(r1 - r3)*(r2 - r3) + pow(c,2)) - 
         2*pow(a,2)*(-2*(r1 - r2)*(r1 - r3) + pow(b,2) + pow(c,2))))
     
    # I think when we have an obtuse triangle, sometimes we get a negative result
    aa, bb, cc = a**2, b**2, c**2
    if not ((aa < bb + cc) & (bb < aa + cc) & (cc < aa + bb)):
        rad = abs(rad)
    return rad

class Pores:
    def __init__(self, x,y,z,sig):
        """Requires periodic BC, box length 1"""
        self.points = np.remainder(array((x,y,z), dtype=float), 1)
        self.sigmas = array(sig, dtype=float)
        self.N = N = len(sig)
        pts = array(list(self.points) + [sig])
        pts1 = np.concatenate([pts.T +(n,m,p,0) for n in [0,-1] for m in [0,-1] for p in [0,-1]], axis=0)
        self.allpoints = pts2 = np.remainder(pts1[:,:3] + 0.5,2)-1
        self.allsigmas = s2 = pts1[:,3]
        d = self.delaunay = Delaunay(pts2)
        
        d.simplices
        triangs = [(d.simplices[:,0], d.simplices[:,1], d.simplices[:,2]),
                   (d.simplices[:,0], d.simplices[:,1], d.simplices[:,3]),
                   (d.simplices[:,0], d.simplices[:,2], d.simplices[:,3]),
                   (d.simplices[:,1], d.simplices[:,2], d.simplices[:,3])]

        triangs = np.concatenate(triangs,axis=1).T
        #print(shape(array(triangs)))

        triangs.sort(1)
        triangs2 = triangs[triangs[:,0] < self.N]
        #print(shape(array(triangs2)))

        trirem = np.remainder(triangs2,N)
        #trirem.sort(1)
        self.triangles = triangs2[unique_rows(trirem)]
        #self.allpoints = pts2
    
    def shapes(self, cut_acute=True):
        """
        Returns sidelengths a,b,c and point radii r1,r2,r3 for each tube.
        
        Note that r1 is the radius across from side a.
        
        if "cutacute" is true, it only returns shapes for acute triangles.
        """
        x,y,z = array(self.allpoints).T
        ix0 = array(self.triangles)
        ix1,ix2,ix3 = ix0.T
        ix = array((ix1,ix2,ix3,ix1)).T
        xt, yt, zt = x[ix],y[ix], z[ix]
        dx, dy, dz = np.diff(xt,1), np.diff(yt,1), np.diff(zt,1)
        c, a, b = np.sqrt(dx**2 + dy**2 + dz**2).T
        r1, r2, r3 = self.allsigmas[ix0].T / 2
        allshapes = array((a,b,c,r1,r2,r3))
        aa, bb, cc = a**2, b**2, c**2
        if not cut_acute: return allshapes
        acutetris = (aa < bb + cc) & (bb < aa + cc) & (cc < aa + bb)
        return allshapes[:, acutetris]
    
    def is_acute(self):
        a,b,c,_,_,_ = self.shapes()
        aa, bb, cc = a**2, b**2, c**2
        acutetris = (aa < bb + cc) & (bb < aa + cc) & (cc < aa + bb)
        return acutetris
    
    def radii(self, cut_acute=True):
        return midcirc(*self.shapes(cut_acute=cut_acute))
    
    def tube_graph(self, min_radius=None, use_outside=True):
        import networkx as nx
        if min_radius is None:
            min_radius = min(self.sigmas)/2.0
        
        G = nx.Graph()
        
        simplices = self.delaunay.simplices
        # select tetrahedrons for which the "minimum" corner is inside the box
        idx1 = np.amin(simplices, axis=1) == np.amin(np.remainder(simplices, self.N), axis=1)
        simplices_in_box = simplices[idx1]
        
        if use_outside:
            # select tetrahedrons for which any corner is inside the box
            idx2 = np.any(simplices == np.remainder(simplices, self.N), axis=1)
            simplices_near_box = simplices[idx2]
        else:
            simplices_near_box = simplices_in_box
        
        pairlen = len(simplices_near_box) * (len(simplices_in_box))
        
        print("Checking", end=' ')
        lastk = -1
        printn = 100
        pausek = round(pairlen / printn)
        for k, (s1, s2) in enumerate(itertools.product(simplices_in_box, simplices_near_box)):
            if k // pausek > lastk:
                print(int(printn - k // pausek), end= ", ")
                lastk = k // pausek
            
            # only check pairs for i < j
            #if min(s2) <= min(s1): continue
            if np.all(s2 == s1): continue
            
            m1, m2 = ((np.remainder(s1, self.N), np.remainder(s2, self.N))
                        if not use_outside
                        else (s1, s2))
            
            
            # m1 and m2 are the "boxed" corners
            # now we see if the two tetrahedrons share a triangle
            intersec = set(m1).intersection(m2)
            if len(intersec) < 3:
                # tetrahedrons are not neighbors
                continue
            if len(intersec) > 3:
                #if use_outside: continue
                print(s1)
                print(m1)
                print(s2)
                print(m2)
                print(intersec)
                raise ValueError("Duplicate tetrahedrons!")
            
            # we've got a triangle, let's get its location
            idx1 = np.array([i in intersec for i in m1])
            ptsidx = s1[idx1]
            x1, x2, x3 = self.allpoints[ptsidx, :]
            
            # now we find the size of the tube through
            a,b,c = triangle_edges(x1, x2, x3)
            r1, r2, r3 = self.allsigmas[ptsidx] / 2
            rad = midcirc(a,b,c, r1, r2, r3)
            
            aa, bb, cc = a**2, b**2, c**2
            acute = ((aa < bb + cc) & (bb < aa + cc) & (cc < aa + bb))
            
            if rad < min_radius: continue # particle won't fit
            G.add_edge(tuple(s1), tuple(s2), weight=(rad), acute=acute)# - maxr))
            
        return G
