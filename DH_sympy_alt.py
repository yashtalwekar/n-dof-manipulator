
# coding: utf-8

# In[5]:


import sympy as sp
# sp.init_printing()
import numpy as np
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt

from numpy.linalg import inv as npinv
import matplotlib.animation as animation


# In[6]:


def sp_mat_dh(alpha, a, d, theta):
        tmp = sp.eye(4)
        ctheta = sp.cos(theta)
        calpha = sp.cos(alpha)
        stheta = sp.sin(theta)
        salpha = sp.sin(alpha)
        tmp[0, :] = sp.Matrix([[ctheta, -stheta*calpha, stheta*salpha, a*ctheta]])
        tmp[1, :] = sp.Matrix([[stheta, ctheta*calpha, -ctheta*salpha, a*stheta]])
        tmp[2, :] = sp.Matrix([[0, salpha, calpha, d]])
        return tmp

def mix_mat_dh(dh_pars, subscript=-1):
    if dh_pars['joint_var'] is 'theta':
        if subscript is -1:
            theta = sp.symbols('theta')
        else:
            theta = sp.symbols('theta' + str(subscript))
        d = dh_pars['d']
        ctheta = sp.cos(theta)
        stheta = sp.sin(theta)
    elif dh_pars['joint_var'] is 'd':
        if subscript is -1:
            d = sp.symbols('d')
        else:
            d = sp.symbols('d' + str(subscript))
        theta = dh_pars['theta']
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
    
    alpha = dh_pars['alpha']
    a = dh_pars['a']
    calpha = np.cos(alpha)
    salpha = np.sin(alpha)
    tmp = sp.eye(4)
    tmp[0, :] = sp.Matrix([[ctheta, -stheta*calpha, stheta*salpha, a*ctheta]])
    tmp[1, :] = sp.Matrix([[stheta, ctheta*calpha, -ctheta*salpha, a*stheta]])
    tmp[2, :] = sp.Matrix([[0, salpha, calpha, d]])
    return tmp
    
def np_mat_dh(dh_pars):
    # return mix_mat_dh(dh_pars)
    return lambdify((sp.symbols(dh_pars['joint_var'])), mix_mat_dh(dh_pars), 'numpy')


# In[7]:


class Manipulator():
    def __init__(self, dh_pars_init):
        # Takes and maintains a list of dictionaries for each joint
        # dh_pars = [{'joint_var':'theta', 'alpha':0, 'a': 1, 'd': 0, 'theta': np.pi/6}, ...]
        self.dh_pars = dh_pars_init
        self.num_joints = len(dh_pars)
        
        # List of joint variable symbols (q_i) and values
        self.dh_vars_sym = [sp.symbols(dh_pars[i]['joint_var'] + str(i+1)) for i in range(self.num_joints)]
        self.dh_vars = self.get_dh_vars()

        # List of num_joints number of joint matrices: joint1->joint2, joint2->joint3, ..., last_joint -> end_effector         
        # Symbolic matrices i-1_A_i, in terms of q_i with i = idx + 1 as physically joints are numbered from 1 and not 0
        self.mat_dh_sym = [mix_mat_dh(elem, idx+1) for idx, elem in enumerate(self.dh_pars)]
        # Numeric functions of above, input q, output numpy matrix
        self.mat_dh_num = [np_mat_dh(elem) for elem in self.dh_pars]
        
        # Numeric functions of matrices 0_A_j, j from 1 to num_joints (transformation from base to end_eff)
        self.dh_funcs_num = [self.fk_transformation_matrix(0, j) for j in range(1, self.num_joints + 1)]
        
        self.point_coords = np.zeros((4, self.num_joints + 1)) # 4-vector representation for positions
        self.point_coords[-1, :] = 1
        # self.calc_point_coords()
        
        # [theta_x, theta_y, theta_z].T
        self.joint_orientations = np.zeros((3, self.num_joints + 1))
        # Generalise this
        self.joint_orientations[2, 1:] = self.dh_vars
        
        self.calc_all_joint_pos_ang()
        
        self.jacobian_sym = self.mat_jacobian()
        self.jacobian_num = lambdify((self.dh_vars_sym), self.jacobian_sym, 'numpy')

    def get_dh_vars(self):
        return [elem[elem['joint_var']] for elem in self.dh_pars]

    def set_dh_vars(self, joint_ind_val):
        # Assigns values from given joint_ind_val to dh_vars
        # joint_ind_val = [[index, value], [index, value], [index, value],...]
        
        for elem in joint_ind_val:
            self.dh_pars[elem[0]][dh_pars[elem[0]]['joint_var']] = elem[1]
            self.dh_vars = self.get_dh_vars()

    def calc_point_coords(self, i):
        # Calculates and sets the current positions for the ith joint
        
        # for i in range(1, self.num_joints + 1):        
        self.point_coords[:, i] = self.dh_funcs_num[i-1](*self.dh_vars[0:i])[:, 3]
    
    def calc_joint_orientations(self, i):
        # Calculates and sets the current orientation for the given joint
        
        # for i in range(1, self.num_joints+1):
        tmp = self.dh_funcs_num[i-1](*self.dh_vars[0:i])
        self.joint_orientations[0, i-1] = np.arctan2(tmp[2, 1], tmp[2, 2])
        self.joint_orientations[1, i-1] = np.arctan2(-tmp[2, 0], np.sqrt(tmp[2, 1]**2 + tmp[2, 2]**2))
        self.joint_orientations[2, i-1] = np.arctan2(tmp[1, 0], tmp[0, 0])
            
    def calc_all_joint_pos_ang(self):
        # Calculates and sets the all joint positions and orientations for the current dh parameter values
        
        for i in range(1, self.num_joints + 1):
            tmp = self.dh_funcs_num[i-1](*self.dh_vars[0:i])
            # positions
            self.point_coords[:, i] = tmp[:, 3]
            
            # orientations
            self.joint_orientations[0, i] = np.arctan2(tmp[2, 1], tmp[2, 2])
            self.joint_orientations[1, i] = np.arctan2(-tmp[2, 0], np.sqrt(tmp[2, 1]**2 + tmp[2, 2]**2))
            self.joint_orientations[2, i] = np.arctan2(tmp[1, 0], tmp[0, 0])
    
    def fk_transformation_matrix(self, i, j):
        # Returns callable function for forward transformation matrix from ith joint to the jth
        
        # i and j range from 0 to num_joints
        # Exception if i > j
        if i is 0:
            tmp = sp.eye(4)
        else:
            tmp = self.mat_dh_sym[i-1]
        t = i
        while t < j:
            t += 1
            tmp *= self.mat_dh_sym[t-1]
            
        return lambdify((self.dh_vars_sym[i:j]), tmp, 'numpy')
    
    def mat_jacobian(self):
        # Returns a symbolic Jacobian matrix
        
        # Will be called only once
        tmp = sp.zeros(6, self.num_joints)
        dh_matrices_0_i = [sp.eye(4)] # Here i = j-1 where j is the joint for which we are computing this
        for i in range(1, self.num_joints + 1): # +1 for including base -> end_effector
            dh_matrices_0_i.append(sp.simplify(dh_matrices_0_i[i-1]*self.mat_dh_sym[i-1]))
        # Can't club the two loops as the above will have to be completed for end effector
        
        coords_end_eff = dh_matrices_0_i[-1][0:3, 3]
        for i in range(0, self.num_joints):
            z_i = sp.Matrix(dh_matrices_0_i[i][0:3, 2])
            if self.dh_pars[i]['joint_var'] is 'theta':
                tmp[:, i] = sp.simplify(z_i.cross(sp.Matrix(coords_end_eff) - sp.Matrix(dh_matrices_0_i[i][0:3, 3])).col_join(z_i))
            elif self.dh_pars[i]['joint_var'] is 'd':
                tmp[0:3, i] = z_i
        return tmp
        # return lambdify((self.dh_vars_sym), tmp, 'numpy')
    
    def plot_current_config(self):
        # Plots the current configuration
        
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-4, 4), ylim=(-4, 4))
        ax.grid()
        ax.plot(manipulator.point_coords[0, :], manipulator.point_coords[1, :], '-o')
        plt.show()
        
    def mat_jacob_pseudo_inv(self, pars_list, pos_only=False):
        # Calculate transpose based pseudo inverse of jacobian
        
        j = self.jacobian_num(*pars_list)
        if pos_only is True:
            j = j[0:3, :]
        j_t = j.transpose()
        j_pseudo = np.matmul(npinv(np.matmul(j_t, j)), j_t)
        return j_pseudo
    
    # def ik_transformation_matrix(self, end_eff_coords, end_eff_orientation):
    def dh_from_req_config(self, end_eff_pos, end_eff_ang, q=None):
        # Obtain dh parameters for a given end effector state
        # end_eff_pos: Required end effector position
        # end_eff_ang: Required end effector orientation
        # q: Initial guess for Newton's method; if omitted, will default as below
        
        # Preserve existing orientation as it will be modified for the following computatiopns
        q_current = self.dh_vars
        
        # Newton's Method
        if q is None:
            q = 0.1*np.ones(self.num_joints)
        end_eff_err = 1
        max_iter = 1e4
        iter_count = 0
        while np.linalg.norm(end_eff_err) > 1e-10 and iter_count < max_iter:
            self.set_dh_vars([[i, q[i]] for i in range(0, manipulator.num_joints)])
            self.calc_all_joint_pos_ang()
            q_pos = np.asarray([self.point_coords[0, -1], self.point_coords[1, -1], self.point_coords[2, -1]])
            q_ang = np.asarray([self.joint_orientations[0, -1], self.joint_orientations[1, -1], self.joint_orientations[2, -1]])
            end_eff_err = [*(end_eff_pos - q_pos), *(end_eff_ang - q_ang)]
            q = q + np.matmul(self.mat_jacob_pseudo_inv(q), end_eff_err)
            q = q%(2*np.pi)
            iter_count += 1
            print("\rIteration {}".format(iter_count), end="")
        
        # Target dh parameters have been acquired, reset the config to before
        self.set_dh_vars([[i, q_current[i]] for i in range(0, self.num_joints)])
        self.calc_all_joint_pos_ang()
        return q
    
    def state_A_to_B(self, q_req, t_req):
        # Generates trajectory as q(t) for the manipulator at state A to attain state B
        # Trajectory obtained by interpolating between corresponding parameter values of A and B
        # q_req: dh parameter values at B
        # t_req: time in which the transition is to be accomplished
        
        def q_of_t(t):
            # t ranges from 0 to t_req
            q_curr = np.asarray(self.dh_vars)
            if t > t_req:
                t = t_req
            return q_curr + t*(q_req - q_curr)/t_req
        return q_of_t
    
    # def follow_trajectory(self, func_lin_vel, func_ang_vel):
        
        


# In[14]:


dh_pars = [{'joint_var':'theta', 'alpha':0, 'a': 0.5, 'd': 0, 'theta': np.pi},            {'joint_var':'theta', 'alpha':0, 'a': 0.5, 'd': 0, 'theta': -np.pi/2}, #            {'joint_var':'theta', 'alpha':0, 'a': 0.5, 'd': 0, 'theta': np.pi/7}, \
           {'joint_var':'theta', 'alpha':0, 'a': 0.5, 'd': 0, 'theta': -np.pi/2}]


# In[15]:


manipulator = Manipulator(dh_pars)

# Inverse Kinematics - Path following  
# {

# In[16]:


# Required end effector position and orientation as functions of time
sym_t = sp.symbols('t')
tmp_start_pos = manipulator.point_coords[0:2, -1]
trajectory_path = [0.25*(1 - sp.cos(np.pi*sym_t)), 0.25*(2 + sp.sin(np.pi*sym_t)), sp.Float(0.0)]
trajectory_orientation = [sp.Float(0.0), sp.Float(0.0), sp.sin(sym_t*np.pi/24)]

# trajectory_path = [0.3*sp.sin(np.pi*sym_t), 0.5*sp.cos(np.pi*sym_t), sp.Float(0.0)]
# trajectory_orientation = [sp.Float(0.0), sp.Float(0.0), sp.Float(0.0)]


# In[17]:


trajectory_lin_pos = lambda t: [lambdify((sym_t), elem, 'numpy')(t) for elem in trajectory_path]
trajectory_ang_pos = lambda t: [lambdify((sym_t), elem, 'numpy')(t) for elem in trajectory_orientation]

trajectory_lin_vel = lambda t: [lambdify((sym_t), elem.diff(), 'numpy')(t) for elem in trajectory_path]
trajectory_ang_vel = lambda t: [lambdify((sym_t), elem.diff(), 'numpy')(t) for elem in trajectory_orientation]

# trajectory_vel = [*trajectory_lin_vel, *trajectory_ang_vel]


# In[18]:

tmp_coords = np.zeros((10, 6, manipulator.num_joints + 1))
tmp_coords[0, 0:3, :] = [manipulator.point_coords[0, :], manipulator.point_coords[1, :], manipulator.point_coords[2, :]]
tmp_coords[0, 3:6, :] = [manipulator.joint_orientations[0, :], manipulator.joint_orientations[1, :], manipulator.joint_orientations[2, :]]
# In[19]:


q = manipulator.dh_vars
dt = 0.001
t_range = np.arange(0, 2, dt)
# time instant, 3 coordinate axis then 3 angles along those axes, joint index
tmp_coords = np.zeros((len(t_range) + 1, 6, manipulator.num_joints + 1))
tmp_coords[0, 0:3, :] = [manipulator.point_coords[0, :], manipulator.point_coords[1, :], manipulator.point_coords[2, :]]
tmp_coords[0, 3:6, :] = [manipulator.joint_orientations[0, :], manipulator.joint_orientations[1, :], manipulator.joint_orientations[2, :]]

for idx, tt in enumerate(t_range):
    print("\rt={}".format(tt), end="")
    q_dot = np.matmul(manipulator.mat_jacob_pseudo_inv(q), [*trajectory_lin_vel(tt), *trajectory_ang_vel(tt)])
    q += q_dot*dt
    manipulator.set_dh_vars([[i, q[i]] for i in range(0, manipulator.num_joints)])
    manipulator.calc_all_joint_pos_ang()
    # tmp_coords[idx + 1, :, :] = [manipulator.point_coords[0, :], manipulator.point_coords[1, :]]
    tmp_coords[idx + 1, 0:3, :] = [manipulator.point_coords[0, :], manipulator.point_coords[1, :], manipulator.point_coords[2, :]]
    tmp_coords[idx + 1, 3:6, :] = [manipulator.joint_orientations[0, :], manipulator.joint_orientations[1, :], manipulator.joint_orientations[2, :]]

# Probably works

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure()
# ax.plot(manipulator.point_coords[0, :], manipulator.point_coords[1, :], '-o')
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-4, 4), ylim=(-4, 4))
ax.grid()
ax.plot(tmp_coords[:, 0, -1], tmp_coords[:, 1, -1])

line, = ax.plot([], [], 'o-')

def init():
    line.set_data([], [])
    return line,

def animate(i):
    
    line.set_data(tmp_coords[i, 0, :], tmp_coords[i, 1, :])

#     line.set_data(item, item + 1)
    return line,
    
ani = animation.FuncAnimation(fig, animate, range(0, len(tmp_coords)),
                              interval=5, blit=True, init_func=init)
# ani.save('im.mp4', writer=writer)

plt.show()


# In[21]:
