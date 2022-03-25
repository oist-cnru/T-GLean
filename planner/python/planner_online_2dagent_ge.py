'''
Report goal error for generated plans using a simulated 2D agent
Please see readme.md for how to use this script

Authored by Takazumi Matsumoto <takazumi.matsumoto@oist.jp>
Copyright (C) 2022 Okinawa Institute of Science and Technology Graduate University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
from gpvrnn import GPvrnn as model
from planner_utils import AgentController, sm_dec, stdout_redirected
import numpy as np
# np.set_printoptions(suppress=True)
from scipy.special import softmax
import os
import sys
import math
from multiprocessing import Pool
import random

## Setup RNN
exp_suffix = "_1a_lr"
ds_suffix = "_a"
exp_runmod = ""
config_path =  "configs/glean/2dagent" + exp_suffix + ".cfg"
output_subdir = "results/2dagent" + exp_suffix + exp_runmod +"/planning_output"
data_path = "datasets/2dagent/2dagent" + ds_suffix + ".npy"
basedir = os.path.relpath(os.path.dirname(os.path.abspath(__file__)) + "/../../") + "/"
save_subdir = "" # empty to disable
rnn = model(config_file=config_path, task="planning", epoch=-1) # initial load

# Planner
max_plan_length = rnn.max_timesteps * 2
static_plan_steps = rnn.max_timesteps // 2 # start shifting after this
joints = 2
signals = 3 # signals are appended to the joints
# Set goal randomly
rgoals = 10
rgoals_range = [0.2, 0.8]
goals = []
train_data = np.load(basedir + data_path)
for _ in range(rgoals):
    rnd_gx = random.uniform(*rgoals_range)
    while rnd_gx in train_data[:, 0, -3]:
        rnd_gx = random.uniform(*rgoals_range) # re-randomize
    goals.append([rnd_gx, 1.0])

print("Testing random untrained goals", goals)

# goal is set per process
goal_offset = -0.1
goal_intermediates = [[0.25, 0.5], [0.75, 0.5]]
goal_intermediate = None
unmask = [0, 2] # dimension range to unmask (not counting signals)
goal_mask = [2, 4] # dimension range containing the goal signal (passive signals omitted)
init_joints = [0.5, 0.1] # initial position
# goal_signals = [goal[0], goal[1], 0.0] # input signals [goal_x, goal_y, stop]
planner_max_epochs = 1000 # internally the RNN will run for rnn.max_epochs for each planner epoch
save_only_improved = False
best_loss = sys.float_info.max
plan_update_threshold = 100.0
barrier_box = [0.35, 0.475, 0.3, 0.05] # define box: [leftcorner_x, leftcorner_y, width, height]. None to disable
project_position_from = [0.0, 0.5] # decode predicted position from angle and distance

stop_signal = -1 # dimension in robot_signal where stop signal goes high to trigger a stop (None to disable)
stop_signal_threshold = 0.1
stop_signal_startscan = 2 # ignore early signals due to initial instability
stop_offset = 1
sm_stop_temp = 0.02

plan_start_offset = 0

### Plan sampling
plan_sampling_count = 10
plan_sampling_random = False # set to true to use random RNG seeds during sampling, or false to use a sequence starting from 0

def plan_sample(config_path, joints, init_joints, goal, goal_offset, unmask, goal_mask, plan_seed, barrier_box=None, push_back_len=0.01, averaged_plan_count=3, project_position_from=[0.0, 0.5], stop_signal=-1, stop_signal_threshold=0.1, stop_signal_startscan=2):
    ## Generates a plan and checks its final distance to the goal
    with stdout_redirected():
        __rnn = model(config_file=config_path, task="planning", epoch=-1, rng_seed=plan_seed)

    if averaged_plan_count > 0:
        averaged_plan_update = True # weighted integration of plan update (reduce jitters)
        update_weight = np.expand_dims(np.array([0.0, 0.2, 0.4, 0.6, 0.8]), axis=0).T # blend averaged plan and raw plan


    # RJA contains window history
    robot_joint_angles = np.zeros(shape=[__rnn.max_timesteps, joints])
    robot_joint_angles[0, 0] = math.atan2(init_joints[1]-project_position_from[1], init_joints[0]-project_position_from[0]) # bearing
    robot_joint_angles[0, 1] = np.linalg.norm(np.asarray(init_joints)-np.asarray(project_position_from)) # distance
    last_generated_plan = []
    active_plan = np.zeros(shape=[__rnn.max_timesteps, joints+signals])
    max_plan_length = __rnn.max_timesteps * 2
    final_plan = np.zeros(shape=[max_plan_length, (joints+signals)])
    plan_window_moving = False

    # Init agent controller
    agent = AgentController(np.array(init_joints), 0)
    agent.active = False # PID and limiters disabled

    # Initial data
    robot_signals = np.zeros(shape=[__rnn.max_timesteps, signals])
    goal_signals = [goal[0], goal[1], 0.0] # input signals [goal_x, goal_y, stop]
    robot_signals[:, :] = np.array(goal_signals)
    current_robot_joint_angles = np.array([math.atan2(agent.pos[1]-project_position_from[1], agent.pos[0]-project_position_from[0]), np.linalg.norm(np.asarray(agent.pos)-np.asarray(project_position_from))]) # bearing, distance
    data_in = np.zeros((__rnn.max_timesteps, __rnn.dims)) # window length
    # FIXME: no well defined API for this op
    data_in[:,joints:] = robot_signals[:, :] # goal signal
    __rnn.er_data = list(data_in) # set future signals
    mask = np.zeros((__rnn.max_timesteps, __rnn.dims*__rnn.softmax_quant))
    mask[0,unmask[0]*__rnn.softmax_quant:unmask[1]*__rnn.softmax_quant] = 1 # unmask initial frame
    mask[:,goal_mask[0]*__rnn.softmax_quant:goal_mask[1]*__rnn.softmax_quant] = 1 # unmask goal signal
    # Generate initial plan
    loss = __rnn.plan(data_in=np.hstack([current_robot_joint_angles, data_in[-1,joints:]]), mask=mask, sub_dir="", output_suffix="", dynamic=False)
    if not all([math.isfinite(l) for l in loss.values()]):
        raise RuntimeError("Loss is not finite!")
    generated_plan = np.asarray(__rnn.getPlanOutput())
    final_plan[0, :] = np.asarray(generated_plan[0])
    plan_start_offset = 0
    steps = 1
    stop_at = -1

    # Main loop
    for t in range(1, max_plan_length):
        if (t < static_plan_steps):
            dynamic = False
            mask[t,unmask[0]*__rnn.softmax_quant:unmask[1]*__rnn.softmax_quant] = 1 # update mask
        else:
            dynamic = True

        # Generate plan
        # FIXME: no well defined API for this op
        loss = __rnn.plan(data_in=np.hstack([current_robot_joint_angles, data_in[-1,joints:]]), mask=mask, sub_dir="", output_suffix="", dynamic=dynamic)
        if not all([math.isfinite(l) for l in loss.values()]):
            raise RuntimeError("Loss is not finite!")

        # src = __rnn.getPosteriorMyuSigma()
        # generated_src = np.concatenate((np.asarray(src["myu"]), np.asarray(src["sigma"])), axis=1).tolist()
        generated_plan = __rnn.getPlanOutput()
        if dynamic or t-static_plan_steps > 0:
            plan_start_offset += 1
            plan_window_moving = True

        if averaged_plan_update:
            if len(last_generated_plan) == averaged_plan_count:
                last_generated_plan.pop(0) # remove oldest
            last_generated_plan.append(generated_plan) # add newest plan
            if plan_window_moving:
                # Strip off first timesteps and pad with newest entry
                for i in range(len(last_generated_plan)-1):
                    last_generated_plan[i] = last_generated_plan[i][1:]
                    last_generated_plan[i].append(last_generated_plan[-1][-1])

            # weighted integration of averaged and new plan
            weighted_range = [t-plan_start_offset, min(t-plan_start_offset+update_weight.size, __rnn.max_timesteps-averaged_plan_count)]
            active_plan[weighted_range[0]:weighted_range[1], :] = ((1.0-update_weight[:weighted_range[1]-weighted_range[0]]) * np.mean(last_generated_plan, axis=0)[weighted_range[0]:weighted_range[1], :]) + (update_weight[:weighted_range[1]-weighted_range[0]] * np.asarray(generated_plan[weighted_range[0]:weighted_range[1]]))
            active_plan[weighted_range[1]:, :] = np.asarray(generated_plan[weighted_range[1]:]) # outside of blended area
        else:
            active_plan = np.asarray(generated_plan)

        final_plan[t, :] = active_plan[t-plan_start_offset] # save history
        # internal = np.asarray(generated_src)
        # zug = np.ones_like(internal, dtype=np.float32)
        # zug[:, :plot_z_units] = 0.00
        # zi = [0.0] * len(__rnn.z_units)
        # for l in range(len(__rnn.z_units)):
        #     for a in range(__rnn.z_units[l]):
        #         zi[l] += np.sum(-0.5*(np.log(internal[:, a+plot_z_units]) - internal[:, a+plot_z_units] - np.square(internal[:, a]) + 1.0)) # KLD
        #     # normalize
        #     zi[l] /= __rnn.z_units[l]
        #     zi[l] /= __rnn.max_timesteps # wnd length
        # zinfo.append(zi)

        # Move sim agent
        last_pos = agent.pos.copy()
        # Decode position and let the controller move
        dec_pos = np.array([math.cos(final_plan[t, 0])*final_plan[t, 1], 0.5+math.sin(final_plan[t, 0])*final_plan[t, 1]])
        # print("Agent at", np.array([math.cos(agent.pos[0])*agent.pos[1], 0.5+math.sin(agent.pos[0])*agent.pos[1]]), "going to", dec_pos)
        agent.move_pos(dec_pos)

        if barrier_box is not None:
            # Check for collision
            if agent.pos[0] > barrier_box[0] and agent.pos[1] > barrier_box[1] and agent.pos[0] < barrier_box[0]+barrier_box[2] and agent.pos[1] < barrier_box[1]+barrier_box[3]:
                # print("Agent collision!")
                # Push agent back outside
                rev_hdg = math.pi + agent.hdg
                while agent.pos[0] > barrier_box[0] and agent.pos[1] > barrier_box[1] and agent.pos[0] < barrier_box[0]+barrier_box[2] and agent.pos[1] < barrier_box[1]+barrier_box[3]:
                    # agent.active = False
                    agent.move(rev_hdg, push_back_len)
                    agent.vel = 0.0
                    # agent.active = True
                    # print("Agent pushed back to [" + str(agent.pos) + "]")
            else:
                # Check for agent passing through barrier
                check_x = np.linspace(agent.pos[0], last_pos[0], 10)
                check_y = np.linspace(agent.pos[1], last_pos[1], 10)
                for n in range(len(check_x)):
                    if check_x[n] > barrier_box[0] and check_y[n] > barrier_box[1] and check_x[n] < barrier_box[0]+barrier_box[2] and check_y[n] < barrier_box[1]+barrier_box[3]:
                        # print("Agent collision!")
                        agent.teleport([check_x[n-1], check_y[n-1]])
                        agent.vel = 0.0
                        # print("Agent stopped at [" + str(tgt_pos) + "]")
                        break

        # Update past
        current_robot_joint_angles = np.array([math.atan2(agent.pos[1]-project_position_from[1], agent.pos[0]-project_position_from[0]), np.linalg.norm(np.asarray(agent.pos)-np.asarray(project_position_from))]) # bearing, distance
        robot_joint_angles[t-plan_start_offset, :] = current_robot_joint_angles

        # Check for goal reached
        if stop_signal is not None:
            # First convert the low level signal into a softmax distribution
            sm_stop_signal = softmax(active_plan[stop_signal_startscan:, stop_signal]/sm_stop_temp)
            active_plan[stop_signal_startscan:, stop_signal] = sm_stop_signal # overwrite
            old_stop_at = stop_at
            # Then decode softmax into scalar value
            stop_at = int(math.ceil(sm_dec(sm_stop_signal, max=__rnn.max_timesteps-1))) + stop_offset + plan_start_offset
            if stop_at < stop_signal_startscan or np.amax(sm_stop_signal[stop_signal_startscan:]) < stop_signal_threshold:
                stop_at = old_stop_at
            if stop_at is not None and t >= stop_at:
                # print("RNN signalled stop at", stop_at)
                break
        steps += 1

    return (loss["total_batch_reconstruction_loss"], loss["total_batch_regularization_loss"], steps, agent.pos[0], agent.pos[1], goal[0], goal[1]+goal_offset, np.linalg.norm(np.asarray(agent.pos)-np.asarray([goal[0], goal[1]+goal_offset])))

if __name__ == "__main__":
    # Main process to kick off sampling
    print("Sampling", plan_sampling_count, "plans...")
    if plan_sampling_random:
        plan_seeds = [random.randint(0, 2**32) for _ in range(plan_sampling_count)]
    else:
        plan_seeds = [s for s in range(plan_sampling_count)]
    with Pool(plan_sampling_count) as p:
        results = [p.apply_async(plan_sample, args=(config_path, joints, init_joints, goals[idx], goal_offset, unmask, goal_mask, _ps, barrier_box)) for idx, _ps in enumerate(plan_seeds)]
        print("Waiting for samples...")
        p.close()
        p.join()
    print("Saving...")
    # Decode results
    results = [r.get() for r in results]
    # Format: rec_loss, reg_loss, plan_steps, final_agent_pos_x, final_agent_pos_y, goal_x, goal_y, goal_deviation
    np.savetxt(output_subdir + "/plan_goal_error" + exp_suffix + exp_runmod + ".csv", np.asarray(results), delimiter=',', header="rec_loss,reg_loss,plan_steps,final_pos_x,final_pos_y,goal_x,goal_y,goal_deviation", comments='')
    print("Done!")
