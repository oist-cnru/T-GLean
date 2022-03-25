import os, sys
import numpy as np
import math, random as rnd
from contextlib import contextmanager

@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout


class AgentController:
    def __init__(self, home_pos, home_hdg, Kp=0.2, Ki=4.0, Kd=0.0, errsum_decay=0.99, windup=10.0, step_time=0.1, vel=0.025, max_vel_mul=2.0, min_vel_mul=0.0, accel=0.01, max_delta_angle=math.radians(50)):
        # PID control
        self.active = True
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.errsum_decay = errsum_decay
        self.windup = windup
        self.step_time = step_time # assuming a constant sample rate in AgentController
        self.errsum = 0.0
        self.last_err = 0.0

        # Agent limits
        self.target_vel = vel # cruising speed
        self.max_vel_mul = max_vel_mul
        self.min_vel_mul = min_vel_mul
        self.accel = accel # assuming constant acceleration up to cruising speed
        self.max_delta_angle = max_delta_angle

        # Agent movement
        self.home_pos = np.asarray(home_pos) # XY location
        self.home_hdg = home_hdg # heading in radians
        self.pos = self.home_pos.copy() # real position
        self.rpos = np.zeros_like(self.pos) # relative position to start
        self.hdg = self.home_hdg
        self.vel = 0.0     

    def reset(self, reset_pos=True, reset_vel=True, active=True):
        self.active = active
        self.errsum = 0.0
        self.last_err = 0.0
        if reset_pos:
            self.pos = self.home_pos.copy()
            self.rpos = np.zeros_like(self.pos)
            self.hdg = self.home_hdg
        if reset_vel:
            self.vel = 0.0

    def move(self, target_hdg, override_vel=None):
        if self.active: # PID control
            err = target_hdg - self.hdg
            # Fix rel heading
            if (err < math.radians(-180)): err += math.radians(360)
            elif (err > math.radians(180)): err -= math.radians(360)
            self.errsum = np.clip((self.errsum_decay*self.errsum) + (err*self.step_time), a_min=-self.windup, a_max=self.windup)
            derr = (err-self.last_err) / self.step_time
            self.last_err = err
            agent_hdg = (self.Kp*err) + (self.Ki*self.errsum) + (self.Kd*derr) # P + I + D
            # print("in = " + str(self.hdg) + " sp = " + str(target_hdg) + " pv = " + str(agent_hdg))
            # print("step = " + str(self.step_time) + " err = " + str(err) + " errsum = " + str(self.errsum) + " derr = " + str(derr))

            # Limit turning (slow down if limit is reached)
            vel_mul = 1.0
            if abs(agent_hdg-self.hdg) > self.max_delta_angle:
                vel_mul = self.max_delta_angle / (abs(agent_hdg-self.hdg)*5)
                # print("Agent rotation limited! Was turning", self.hdg, "to", agent_hdg)
                if agent_hdg > self.hdg:
                    agent_hdg = self.hdg + self.max_delta_angle
                else:
                    agent_hdg = self.hdg - self.max_delta_angle
                # print("Now", self.hdg, "to", agent_hdg, "at w/ speed multiplier", vel_mul)

            # Set target velocity
            vel_mul = np.clip(vel_mul, a_min=self.min_vel_mul, a_max=self.max_vel_mul)
            if override_vel is None:
                agent_vel = vel_mul*self.target_vel
            else:
                agent_vel = vel_mul*override_vel
            # Limit acceleration
            if (abs(self.vel-agent_vel) > self.accel):
                # print("Agent acceleration limited!")
                if self.vel > agent_vel:
                    agent_vel = self.vel-self.accel
                else:
                    agent_vel = self.vel+self.accel
            if agent_vel > 0.0:
                self.hdg = agent_hdg
            self.vel = agent_vel
        else:
            self.hdg = target_hdg
            self.vel = self.target_vel

        # New position
        self.pos += np.array([math.sin(self.hdg)*self.vel, math.cos(self.hdg)*self.vel])
        self.rpos += np.array([math.sin(self.hdg)*self.vel, math.cos(self.hdg)*self.vel])

    def move_pos(self, target_pos):
        target_hdg = math.atan2(target_pos[0]-self.pos[0], target_pos[1]-self.pos[1])
        target_vel = np.linalg.norm(self.pos-target_pos)
        self.move(target_hdg, target_vel)

    def move_rpos(self, target_rpos):
        target_hdg = math.atan2(target_rpos[0]-self.rpos[0], target_rpos[1]-self.rpos[1])
        target_vel = np.linalg.norm(self.rpos-target_rpos)
        self.move(target_hdg, target_vel)
    
    def teleport(self, target_pos):
        self.pos = target_pos
        self.rpos = target_pos - self.home_pos
        # print("Teleported to", self.pos, self.rpos)

class AgentExp:
    # This class stores agent movement during experiments
    def __init__(self, agent, store_axy, ref_pos, max_traj_len, step_hdg_var):
        self.agent = agent # agent controller
        self.traj_goal = []
        self.traj_pos = []
        self.traj_hdg = []
        self.traj_vel = []
        self.agent_brng = []
        self.agent_dist = []
        self.barrier_box = None
        self.store_axy = store_axy
        self.ref_pos = ref_pos
        self.max_traj_len = max_traj_len
        self.step_hdg_var = step_hdg_var
        self.step = 0
        self.last_agent_pos = None
        self.status = "off" # off, ready, go_to_ipos, collision, timeout

    def is_close(self, pos, target, tolerance, last_pos=None):
        # print(np.linalg.norm(pos-target), tolerance)
        if abs(np.linalg.norm(pos-target)) < tolerance:
            return True # end point near target
        elif last_pos is not None:
            # check if we've gone through the target
            if abs((np.linalg.norm(last_pos-target) + np.linalg.norm(pos-target)) - np.linalg.norm(pos-target)) < tolerance:
                return True

    def check_collision(self, agent, last_agent_pos, barrier_box, subdiv=10):
        if agent.pos[0] > barrier_box[0] and agent.pos[1] > barrier_box[1] and agent.pos[0] < barrier_box[0]+barrier_box[2] and agent.pos[1] < barrier_box[1]+barrier_box[3]:
            print("Agent collision!")
            return True
        else:
            # Check for agent passing through barrier
            check_x = np.linspace(last_agent_pos[0], agent.pos[0], subdiv)
            check_y = np.linspace(last_agent_pos[1], agent.pos[1], subdiv)
            for i in range(len(check_x)):
                if check_x[i] > barrier_box[0] and check_y[i] > barrier_box[1] and check_x[i] < barrier_box[0]+barrier_box[2] and check_y[i] < barrier_box[1]+barrier_box[3]:
                    return True
        return False
    
    def reset(self):
        # Move agent back to start
        self.traj_goal = []
        self.traj_pos = []
        self.traj_hdg = []
        self.traj_vel = []
        self.agent_brng = []
        self.agent_dist = []
        self.agent.reset()
        self.traj_pos.append(self.agent.rpos.copy() if self.store_axy else self.agent.pos.copy())
        self.traj_hdg.append(self.agent.hdg)
        self.traj_vel.append(self.agent.vel)
        self.agent_brng.append(math.atan2(self.agent.pos[1]-self.ref_pos[1], self.agent.pos[0]-self.ref_pos[0]))
        self.agent_dist.append(np.linalg.norm(self.agent.pos-self.ref_pos))
        self.last_agent_pos = self.agent.pos.copy()
        self.status = "ready"
        self.step = 1

    def go_to_pos(self, target_pos, goal_signal=None, slow_to_stop=False):
        # Moves the agent towards a goal position
        if self.status != "ready":
            print("go_to_pos: Agent is not ready! status:", self.status)
            return False
        print("go_to_pos: Agent heading to", target_pos)
        while not self.is_close(self.agent.pos, target_pos, self.agent.vel*self.agent.min_vel_mul+0.01, self.last_agent_pos) and self.step < self.max_traj_len:
            target_hdg = math.atan2(target_pos[0]-self.agent.pos[0], target_pos[1]-self.agent.pos[1]) # based on current agent pos, determine target heading
            if self.step_hdg_var > 0:
                target_hdg = rnd.normal(target_hdg, self.step_hdg_var)
            self.status = "go_to_pos"
            self.last_agent_pos = self.agent.pos.copy()

            # slow down as the goal approaches
            if slow_to_stop and self.is_close(self.agent.pos, target_pos, self.agent.vel*2.0):
                agent_step /= 1.4
            else:
                agent_step = self.agent.target_vel # default speed

            self.agent.move(target_hdg, override_vel=agent_step) # move with controller

            # Test obstruction
            if self.barrier_box is not None and self.check_collision(self.agent, self.last_agent_pos, self.barrier_box):
                self.status = "collision"
                return False

            # Record
            self.traj_pos.append(self.agent.rpos.copy() if self.store_axy else self.agent.pos.copy())
            self.traj_hdg.append(self.agent.hdg)
            self.traj_vel.append(self.agent.vel)
            self.traj_goal.append(goal_signal.copy() if goal_signal is not None else target_pos.copy())
            self.agent_brng.append(math.atan2(self.agent.pos[1]-self.ref_pos[1], self.agent.pos[0]-self.ref_pos[0]))
            self.agent_dist.append(np.linalg.norm(self.agent.pos-self.ref_pos))
            # print("Reconstructed pos:", np.array([math.cos(self.agent_brng[-1])*self.agent_dist[-1], 0.5+math.sin(self.agent_brng[-1])*self.agent_dist[-1]]))
            # print("Agent rpos:", math.atan2(self.agent_pos[1]-self.ref_pos[1], self.agent_pos[0]-self.ref_pos[0]), np.linalg.norm(self.agent_pos-self.ref_pos))
            self.step += 1
            # print("Agent move:", agent_hdg, agent_step)

        if self.is_close(self.agent.pos, target_pos, self.agent.vel*self.agent.min_vel_mul+0.01, self.last_agent_pos):
            # print("Intermediate", str(target_pos[0]) + ", " + str(target_pos[1]), "reached in", step)
            self.status = "ready"
            return True
        else:
            self.status = "timeout"
            return False


def sm_dec(input, min=0.0, max=1.0):
    # Decodes a softmax vector into an analog scalar value
    ref = np.linspace(min, max, input.size)
    return np.matmul(input, ref.T)

def sm_enc(input, min=-1.0, max=1.0, sm_temp=0.02, sm_bins=10):
    # Encodes an analog scalar value into a softmax vector
    ref = np.linspace(min, max, sm_bins)
    enc = np.exp(-np.square(ref-input)/sm_temp)
    return enc/np.sum(enc)

def padded_moving_avg(arr, n=3, skip=0):
    ret = np.cumsum(arr[skip:], axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[n-1:] /= n
    ret[:n] = arr[skip:skip+n]
    if skip > 0:
        ret = np.concatenate((arr[:skip], ret), axis=0)
    return ret

def project_pos(ref_pos, brng, dist):
    return [ref_pos[0]+np.cos(brng)*dist, ref_pos[1]+np.sin(brng)*dist]

def project_pos_plan(ref_pos, plan):
    return [ref_pos[0]+np.cos(plan[:, 0])*plan[:, 1], ref_pos[1]+np.sin(plan[:, 0])*plan[:, 1]]

def project_to(ref_pos, pos):
    return [math.atan2(pos[1]-ref_pos[1], pos[0]-ref_pos[0]), np.linalg.norm(np.asarray(pos)-np.asarray(ref_pos))] # bearing, distance
