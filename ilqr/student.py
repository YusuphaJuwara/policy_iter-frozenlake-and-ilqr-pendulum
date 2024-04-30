from autograd import grad, jacobian
import autograd.numpy as np
from numpy import linalg


class ILqr:
    
    def __init__(self, dynamics, cost, horizon=50):
        
        self.f = dynamics
        self.horizon = horizon
        
        self.getA = jacobian(self.f,0)
        self.getB = jacobian(self.f,1)

        self.cost = cost
        self.getq = grad(self.cost,0)
        self.getr = grad(self.cost,1)
        
        self.getQ = jacobian(self.getq,0)
        self.getR = jacobian(self.getr,1)
        
    def backward(self, x_seq, u_seq):
        
        pt1 = self.getq(x_seq[-1],u_seq[-1]) # shape: (2,) val: [a,b]
        Pt1 = self.getQ(x_seq[-1],u_seq[-1]) # shape: (2, 2) val: [[a,b],[c,d]]
        
        k_seq = []
        K_seq = []
        
        for t in range(self.horizon-1,-1,-1):

            xt = x_seq[t] # shape: (2,)
            ut = u_seq[t] # shape: (1,)
            
            At = self.getA(xt,ut) # shape: (2, 2)
            Bt = self.getB(xt,ut) # shape: (2, 1)
            
            qt = self.getq(xt,ut) # shape: (2,)
            rt = self.getr(xt,ut) # shape: (1,)
            
            Qt = self.getQ(xt,ut) # shape: (2, 2)
            Rt = self.getR(xt,ut) # shape: (1, 1)

            # TODO
            kt = -linalg.inv(Rt+Bt.T@Pt1@Bt)@(rt+Bt.T@pt1)  # shape: (1,)
            Kt = -linalg.inv(Rt+Bt.T@Pt1@Bt)@Bt.T@Pt1@At    # shape: (1, 2)
            # if t == 0:
            #     print(f"kt: {kt}\tKt shape: {Kt.shape}\tKt: {Kt}\n")
            # TODO
            pt = qt + Kt.T@(Rt@kt+rt)+(At+Bt@Kt).T@pt1+(At+Bt@Kt).T@Pt1@Bt@kt
            Pt = Qt + Kt.T@Rt@Kt + (At+Bt@Kt).T@Pt1@(At+Bt@Kt)

            pt1 = pt
            Pt1 = Pt

            k_seq.append(kt)
            K_seq.append(Kt)
        
        k_seq.reverse()
        K_seq.reverse()
        
        return k_seq,K_seq
    
    def forward(self, x_seq, u_seq, k_seq, K_seq):
        
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        
        for t in range(len(u_seq)):
            # TODO
            # Compute control using the feedback control law
            control = k_seq[t] + K_seq[t]@(x_seq_hat[t] - x_seq[t])
            
            # clip controls to the actual range from gymnasium
            u_seq_hat[t] = np.clip(u_seq[t] + control,-2,2)

            # Propagate the system dynamics forward to the next time step
            x_seq_hat[t+1] = self.f(x_seq_hat[t], u_seq_hat[t])
            
        return x_seq_hat, u_seq_hat


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def cost(x,u):
    costs = angle_normalize(x[0])**2 + .1*x[1]**2 + .001*(u**2)
    return costs

def pendulum_dyn(x,u):    
    th = x[0]
    thdot = x[1]

    g = 10.
    m = 1.
    l = 1.
    dt = 0.05

    u = np.clip(u, -2, 2)[0]

    # TODO
    num_1 = np.sin(th)*3*g
    den_1 = 2*l
    term_1 = num_1 / den_1
    num_2 = 3.0*u
    den_2 = m*(l**2)
    term_2 = num_2 / den_2

    newthdot = thdot + (term_1 + term_2) * dt
    newth = th + newthdot*dt
    
    newthdot = np.clip(newthdot, -8, 8)

    x = np.array([newth, newthdot])
    return x
