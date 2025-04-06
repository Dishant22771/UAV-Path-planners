
import numpy as np

def R_matrix(z_angle,y_angle,x_angle):
    return np.array([[np.cos(z_angle), -np.sin(z_angle), 0.0], [np.sin(z_angle), np.cos(z_angle), 0.0], [0.0, 0.0, 1.0]])@ \
           np.array([[np.cos(y_angle), 0.0, np.sin(y_angle)], [0.0, 1.0, 0.0], [-np.sin(y_angle), 0.0, np.cos(y_angle)]])@ \
           np.array([[1.0, 0.0, 0.0], [0.0, np.cos(x_angle), -np.sin(x_angle)], [0.0, np.sin(x_angle), np.cos(x_angle)]])

def getblocks():
    block = [
        [4.00, 12.00, 0.00, 4.50, 18.00, 1.50],  
        [3.50, 12.00, 1.50, 5.00, 18.00, 3.00], 
        [10.00, 12.00, 0.00, 10.50, 13.00, 1.50],  
        [9.50, 12.00, 1.50, 11.00, 13.00, 3.00]]  
    
    Obstacles = []
    for i in block:
        i = np.array(i)
        Obstacles.append([j for j in i])
    return np.array(Obstacles)

def getballs():
    spheres = [[2.0,6.0,2.5,1.0],[14.0,14.0,2.5,2]]
    Obstacles = []
    for i in spheres:
        Obstacles.append([j for j in i])
    return np.array(Obstacles)





def getballs():
    # Spheres representing animals with varying sizes and positions
    spheres = [
        [2.0, 6.0, 2.5, 1.0],  
        [10.0, 10.0, 1.0, 0.5], 
        [18.0, 3.0, 1.0, 1.5],  
        [1.0, 18.0, 0.5, 0.3],  
        [14.0, 14.0, 2.5, 2],
        [5.0, 15.0, 2.0, 2.5],
        [12.0, 5.0, 1.0, 1.2],
        [15.0, 18.0, 3.0, 2.8],
        [3.0, 10.0, 1.0, 1.1],
        [19.0, 16.0, 2.0, 1.9]
        
    ]
    Obstacles = []
    for i in spheres:
        Obstacles.append([j for j in i])
    return np.array(Obstacles)

def getAABB(blocks):
    AABB = []
    for i in blocks:
        AABB.append(aabb(i))  
    return AABB

def getAABB2(blocks):
    AABB = []
    for i in blocks:
        AABB.append(aabb(i))
    return AABB

def add_block(block = [1.51e+01, 0.00e+00, 2.10e+00, 1.59e+01, 5.00e+00, 6.00e+00]):
    return block

class aabb(object):
    # P: center point
    # E: extents
    # O: Rotation matrix in SO(3), in {w}
    def __init__(self,AABB):
        self.P = [(AABB[3] + AABB[0])/2, (AABB[4] + AABB[1])/2, (AABB[5] + AABB[2])/2]
        self.E = [(AABB[3] - AABB[0])/2, (AABB[4] - AABB[1])/2, (AABB[5] - AABB[2])/2]
        self.O = [[1,0,0],[0,1,0],[0,0,1]]

class obb(object):
    def __init__(self, P, E, O):
        self.P = P
        self.E = E
        self.O = O
        self.T = np.vstack([np.column_stack([self.O.T,-self.O.T@self.P]),[0,0,0,1]])

class env():
    def __init__(self, xmin=0, ymin=0, zmin=0, xmax=20, ymax=20, zmax=5, resolution=1):
        self.resolution = resolution
        self.boundary = np.array([xmin, ymin, zmin, xmax, ymax, zmax]) 
        self.blocks = getblocks()
        self.AABB = getAABB2(self.blocks)
        self.AABB_pyrr = getAABB(self.blocks)
        self.balls = getballs()
        self.OBB = np.array([obb([5.0,7.0,2.5],[0.5,2.0,2.5],R_matrix(135,0,0)),
                             obb([12.0,4.0,2.5],[0.5,2.0,2.5],R_matrix(45,0,0))])
        self.start = np.array([2.0, 2.0, 2.0])
        self.goal = np.array([6.0, 16.0, 4.0])
        self.t = 0 

    def get_distance_to_obstacle(self, node, obstacle):
        if len(obstacle) == 4:
            center = np.array(obstacle[:3])
        else:
            center = np.array([(obstacle[0] + obstacle[3]) / 2, (obstacle[1] + obstacle[4]) / 2, (obstacle[2] + obstacle[5]) / 2])
        return np.linalg.norm(np.array(node) - center)


    def New_block(self):
        newblock = add_block()
        self.blocks = np.vstack([self.blocks,newblock])
        self.AABB = getAABB2(self.blocks)
        self.AABB_pyrr = getAABB(self.blocks)

    def move_start(self, x):
        self.start = x

    def move_block(self, a = [0,0,0], s = 0, v = [0.1,0,0], block_to_move = 0, mode = 'translation'):
        # t is time , v is velocity in R3, a is acceleration in R3, s is increment ini time, 
        # R is an orthorgonal transform in R3*3, is the rotation matrix
        # (s',t') = (s + tv, t) is uniform transformation
        # (s',t') = (s + a, t + s) is a translation
        if mode == 'translation':
            ori = np.array(self.blocks[block_to_move])
            self.blocks[block_to_move] = \
                np.array([ori[0] + a[0],\
                    ori[1] + a[1],\
                    ori[2] + a[2],\
                    ori[3] + a[0],\
                    ori[4] + a[1],\
                    ori[5] + a[2]])

            self.AABB[block_to_move].P = \
            [self.AABB[block_to_move].P[0] + a[0], \
            self.AABB[block_to_move].P[1] + a[1], \
            self.AABB[block_to_move].P[2] + a[2]]
            self.t += s
            a = self.blocks[block_to_move]
            return np.array([a[0] - self.resolution, a[1] - self.resolution, a[2] - self.resolution, \
                            a[3] + self.resolution, a[4] + self.resolution, a[5] + self.resolution]), \
                    np.array([ori[0] - self.resolution, ori[1] - self.resolution, ori[2] - self.resolution, \
                            ori[3] + self.resolution, ori[4] + self.resolution, ori[5] + self.resolution])
    def move_OBB(self, obb_to_move = 0, theta=[0,0,0], translation=[0,0,0]):
        ori = [self.OBB[obb_to_move]]
        self.OBB[obb_to_move].P = \
            [self.OBB[obb_to_move].P[0] + translation[0], 
            self.OBB[obb_to_move].P[1] + translation[1], 
            self.OBB[obb_to_move].P[2] + translation[2]]
        self.OBB[obb_to_move].O = R_matrix(z_angle=theta[0],y_angle=theta[1],x_angle=theta[2])
        self.OBB[obb_to_move].T = np.vstack([np.column_stack([self.OBB[obb_to_move].O.T,\
            -self.OBB[obb_to_move].O.T@self.OBB[obb_to_move].P]),[translation[0],translation[1],translation[2],1]])
        return self.OBB[obb_to_move], ori[0]
          
if __name__ == '__main__':
    newenv = env()
    node = [3, 3, 3]
    distance_to_sphere = environment.get_distance_to_obstacle(node, environment.balls[0])
    distance_to_block = environment.get_distance_to_obstacle(node, environment.blocks[0])
    print("Distance to sphere:", distance_to_sphere)
    print("Distance to block:", distance_to_block)
