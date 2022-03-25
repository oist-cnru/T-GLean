'''
Anticipate goals and future actions for a saved 2D agent trajectory
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
from planner_utils import AgentController, sm_dec, sm_enc, project_pos, project_pos_plan
import numpy as np
# np.set_printoptions(suppress=True)
from scipy.special import softmax
import math, random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from multiprocessing import Pool
import threading
import time
import os, sys
import traceback

## Setup RNN
exp_suffix = "_1bc"
exp_runmod = ""
config_path =  "configs/glean/2dagent" + exp_suffix + ".cfg"
output_subdir = "results/2dagent" + exp_suffix + exp_runmod +"/planning_output"
data_path = "datasets/2dagent/2dagent_observe-ccw.npy"
basedir = os.path.relpath(os.path.dirname(os.path.abspath(__file__)) + "/../../") + "/"
save_subdir = "" # empty to disable
save_only_init_er = True # stop saving after initial plan
rnn = model(config_file=config_path, task="planning", epoch=-1)

# Planner
max_plan_length = rnn.max_timesteps * 2
static_plan_steps = rnn.max_timesteps // 2 # start shifting after this
joints = 2
signals = 3 # signals are appended to the joints
# No goal set!
unmask = [0, 2] # dimension range to unmask (not counting signals)
goal_mask = None # dimension range containing the goal signal (passive signals omitted)
# goal_signals = [goal[0], goal[1], 0.0] # input signals [goal_x, goal_y, stop]
planner_max_epochs = 1000 # internally the RNN will run for rnn.max_epochs for each planner epoch
plan_update_threshold = 100.0
plan_save_data = [] # save planner output to npz
save_only_improved = False

# Shared memory between planner and controller
run_planner = False
generated_plan = []
generated_src = []
full_loss = dict()
best_loss = sys.float_info.max
last_plan = []
current_timestep = 0
epoch = 0

barrier_box = [0.35, 0.475, 0.3, 0.05] # define box: [leftcorner_x, leftcorner_y, width, height]. None to disable
push_back_len = 0.01
project_position_from = [0.0, 0.5] # decode predicted position from angle and distance

input_theta = True # generate angle and velocity instead of position
input_rpos = False # input relative position (to start) instead of absolute position
input_source_file = np.squeeze(np.load(data_path)) # Agent input comes from a file instead of the RNN!
if input_source_file is None:
    init_joints = [0.5, 0.1] # initial position
else:
    init_joints = project_pos(project_position_from, input_source_file[0, 0], input_source_file[0, 1])
    max_plan_length = input_source_file.shape[0]

# RJA contains window history
robot_joint_angles = np.zeros(shape=[rnn.max_timesteps, joints])
if input_theta:
    robot_joint_angles[0, 0] = math.atan2(init_joints[1]-project_position_from[1], init_joints[0]-project_position_from[0]) # bearing
    robot_joint_angles[0, 1] = np.linalg.norm(np.asarray(init_joints)-np.asarray(project_position_from)) # distance
else:
    if not input_rpos:
        robot_joint_angles[0, :] = np.array(init_joints) # initial pose
    else:
        robot_joint_angles[0, :] = np.zeros_like(init_joints)
robot_signals = np.zeros(shape=[rnn.max_timesteps, signals])
# robot_signals[:, :] = np.array(goal_signals)  # constant goal signal initially
current_robot_joint_angles = robot_joint_angles[0, :]
current_robot_signals = robot_signals[0, :]

# Store real history
robot_joint_history = np.zeros(shape=[max_plan_length, joints])
robot_joint_history[0, :] = robot_joint_angles[0, :]
robot_signal_history = np.zeros(shape=[max_plan_length, signals])

wait_for_plan_update = True # run robot synchronously with the planner
init_frame_only = False # behave like offline planner
stop_signal = None # dimension in robot_signal where stop signal goes high to trigger a stop (None to disable)
stop_offset = 1
sm_stop_temp = 0.02
stop_signal_threshold = 0.1
stop_signal_startscan = 2 # ignore early signals due to initial instability
stop_signal_getmax = True
sm_goal_bins = 3
sm_goal_temp = 0.33
sm_goal_range = (-1.0, 1.0)
goal_signal_threshold = 0.9

plan_start_offset = 0

# Let the planner run in the background
def thread_planner():
    global run_planner
    global generated_plan, generated_src, epoch, current_timestep, plan_start_offset, save_subdir
    global full_loss
    if not run_planner:
        print("Waiting to run planner")
    while not run_planner:
        time.sleep(0.001) # spin

    # run_planner
    # Initial data
    data_in = np.zeros((rnn.max_timesteps, rnn.dims)) # window length
    data_in[:,:joints] = robot_joint_angles # initial frame
    data_in[:,joints:] = robot_signals[:, :] # goal signal
    rnn.er_data = list(data_in) # set signals
    mask = np.zeros((rnn.max_timesteps, rnn.dims*rnn.softmax_quant))
    mask[0,unmask[0]*rnn.softmax_quant:unmask[1]*rnn.softmax_quant] = 1 # unmask initial frame
    if goal_mask is not None:
        mask[:,goal_mask[0]*rnn.softmax_quant:goal_mask[1]*rnn.softmax_quant] = 1 # unmask goal signal

    print("Beginning planning loop")
    start_time = time.time()
    for epoch in range(0, planner_max_epochs):
        if not run_planner:
            print("Planner disabled!")
            return
        # Update with current step sensory data
        step = 0 if (epoch == 0 or init_frame_only) else current_timestep

        # data_in[step,:joints] = robot_joint_angles[step,:]
        # data_in[step:,joints:] = robot_signals[step:, :]
        if (step < static_plan_steps):
            dynamic = False
            mask[step,unmask[0]*rnn.softmax_quant:unmask[1]*rnn.softmax_quant] = 1 # update mask
        else:
            dynamic = True

        while wait_for_plan_update and rnn.epoch > 0 and (last_plan is None or not np.array_equal(last_plan, generated_plan)):
            time.sleep(0.01)

        # Generate plan
        # FIXME: data_in has crja + signals
        loss = rnn.plan(data_in=np.hstack([current_robot_joint_angles, data_in[-1,joints:]]), mask=mask, sub_dir=save_subdir, output_suffix=str(epoch), dynamic=dynamic)
        if save_only_init_er:
            save_subdir = ""
        loss_rec = loss["total_batch_reconstruction_loss"]
        loss_reg = loss["total_batch_regularization_loss"]
        loss_total = loss_rec + loss_reg
        print("Epoch %d  time %.0f / %.3f  loss_total %.3f  loss_rec %.3f  loss_reg %.3f" % (epoch*rnn.max_epochs, time.time() - start_time, (time.time() - start_time)/(epoch*rnn.max_epochs+1), loss_total, loss_rec, loss_reg))
        if not save_only_improved or (save_only_improved and best_loss > loss_total):
            if loss_total <= plan_update_threshold:
                src = rnn.getPosteriorMyuSigma()
                generated_plan = rnn.getPlanOutput()
                generated_src = np.concatenate((np.asarray(src["myu"]), np.asarray(src["sigma"])), axis=1).tolist()
                if dynamic or step-static_plan_steps > 0:
                    plan_start_offset += 1
                full_loss = rnn.getPlanLoss(weighted_kld=False)
                best_loss = loss_rec + loss_reg
                print("Updated plan")
            else:
                print("Skipped plan update")
    print("End of planning loop")

# Visualization
live_dashboard = True # display output

if live_dashboard:
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    resolution = (1800, 600)
    fps = 10.0
    cv2.startWindowThread()
    cv2.namedWindow("GLean planner dashboard")
    if not os.path.exists(basedir + output_subdir):
        os.makedirs(basedir + output_subdir)
    dash_avi = cv2.VideoWriter(basedir + output_subdir + "/output.avi", fourcc, fps, resolution, True)

## Start planner
run_planner = True
online_planner = threading.Thread(target=thread_planner)
online_planner.daemon = True
online_planner.start()

final_plan = np.zeros(shape=[max_plan_length, (joints+signals)])
evidence_fe = []
estimated_fe = []
zinfo = []
plot_z_units = np.sum(rnn.z_units)

averaged_plan_update = False # weighted integration of plan update (reduce jitters)
averaged_plan_count = 3 # average over past plans
update_weight = np.expand_dims(np.array([0.0, 0.2, 0.4, 0.6, 0.8]), axis=0).T # blend averaged plan and raw plan
last_generated_plan = []
active_plan = np.zeros(shape=[rnn.max_timesteps, joints+signals])
plan_window_moving = False

try:
    # Wait until planner writes first plan
    while len(generated_plan) == 0:
        print("Waiting for planner...")
        time.sleep(1)

    final_plan[0, :] = np.asarray(generated_plan[0])
    cur_hdg = 0 # north
    cur_pos = np.array(init_joints)
    agent = AgentController(cur_pos, cur_hdg)
    # agent.active = False # PID and limiters disabled
    # Main loop
    for t in range(1, max_plan_length):
        plan_window_moving = True if plan_start_offset > 0 else False
        vel_mul = 1.0
        if not run_planner:
            break
        while wait_for_plan_update and epoch != planner_max_epochs-1 and np.array_equal(last_plan, generated_plan):
            time.sleep(0.01) # wait until the plan is updated

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
            weighted_range = [current_timestep-plan_start_offset, min(current_timestep-plan_start_offset+update_weight.size, rnn.max_timesteps-averaged_plan_count)]
            active_plan[weighted_range[0]:weighted_range[1], :] = ((1.0-update_weight[:weighted_range[1]-weighted_range[0]]) * np.mean(last_generated_plan, axis=0)[weighted_range[0]:weighted_range[1], :]) + (update_weight[:weighted_range[1]-weighted_range[0]] * np.asarray(generated_plan[weighted_range[0]:weighted_range[1]]))
            active_plan[weighted_range[1]:, :] = np.asarray(generated_plan[weighted_range[1]:]) # outside of blended area
        else:
            active_plan = np.asarray(generated_plan)

        final_plan[t, :] = active_plan[t-plan_start_offset] # save history
        internal = np.asarray(generated_src)
        zug = np.ones_like(internal, dtype=np.float32)
        zug[:, :plot_z_units] = 0.00
        zi = [0.0] * len(rnn.z_units)
        for l in range(len(rnn.z_units)):
            for a in range(rnn.z_units[l]):
                zi[l] += np.sum(-0.5*(np.log(internal[:, a+plot_z_units]) - internal[:, a+plot_z_units] - np.square(internal[:, a]) + 1.0)) # KLD
            # normalize
            zi[l] /= rnn.z_units[l]
            zi[l] /= rnn.max_timesteps # wnd length
        zinfo.append(zi)

        if plan_save_data is not None:
            plan_save_data.append([active_plan, np.asarray(generated_plan), robot_joint_history[t, :], robot_signal_history[t, :], internal, np.asarray(full_loss["batch_reconstruction_loss"]), np.asarray(full_loss["batch_regularization_loss"])])

        # Sim agent
        last_pos = agent.pos.copy()
        if input_theta:
            # Decode position and let the controller move
            if input_source_file is None:
                dec_pos = np.array(project_pos(project_position_from, final_plan[t, 0], final_plan[t, 1]))
            else:
                dec_pos = np.array(project_pos(project_position_from, input_source_file[t, 0], input_source_file[t, 1]))
            agent.move_pos(dec_pos)
        else:
            if input_source_file is None:
                tgt_pos = final_plan[t, :2]
            else:
                tgt_pos = input_source_file[t, :2]
            if not input_rpos:
                agent.move_pos(tgt_pos)
            else:
                agent.move_rpos(tgt_pos)

        if barrier_box is not None:
            # Check for collision
            if agent.pos[0] > barrier_box[0] and agent.pos[1] > barrier_box[1] and agent.pos[0] < barrier_box[0]+barrier_box[2] and agent.pos[1] < barrier_box[1]+barrier_box[3]:
                print("Agent collision!")
                # Push agent back outside
                rev_hdg = math.pi + agent.hdg
                while agent.pos[0] > barrier_box[0] and agent.pos[1] > barrier_box[1] and agent.pos[0] < barrier_box[0]+barrier_box[2] and agent.pos[1] < barrier_box[1]+barrier_box[3]:
                    agent.active = False
                    agent.move(rev_hdg, push_back_len)
                    agent.vel = 0.0
                    agent.active = True
                    # print("Agent pushed back to [" + str(agent.pos) + "]")
            else:
                # Check for agent passing through barrier
                check_x = np.linspace(agent.pos[0], last_pos[0], 10)
                check_y = np.linspace(agent.pos[1], last_pos[1], 10)
                for n in range(len(check_x)):
                    if check_x[n] > barrier_box[0] and check_y[n] > barrier_box[1] and check_x[n] < barrier_box[0]+barrier_box[2] and check_y[n] < barrier_box[1]+barrier_box[3]:
                        print("Agent collision!")
                        agent.teleport([check_x[n-1], check_y[n-1]])
                        agent.vel = 0.0
                        # print("Agent stopped at [" + str(tgt_pos) + "]")
                        break

        # Update past
        if input_theta:
            # Encode position
            current_robot_joint_angles[0] = math.atan2(agent.pos[1]-project_position_from[1], agent.pos[0]-project_position_from[0]) # bearing
            current_robot_joint_angles[1] = np.linalg.norm(np.asarray(agent.pos)-np.asarray(project_position_from)) # distance
        else:
            current_robot_joint_angles = agent.pos if not input_rpos else agent.rpos
        robot_joint_angles[t-plan_start_offset, :] = current_robot_joint_angles
        last_plan = np.asarray(generated_plan)
        plan_end = rnn.max_timesteps + plan_start_offset
        plan_end += 1 if plan_window_moving else 0 # strange hack
        copy_offset = max(robot_joint_angles.shape[0] - robot_joint_history.shape[0] + plan_start_offset, 0)
        robot_joint_history[t, :] = robot_joint_angles[t-plan_start_offset, :]
        robot_signal_history[t, :] = robot_signals[t-plan_start_offset, :]

        if input_rpos:
            robot_joint_history[plan_start_offset:plan_end, :] += np.array(init_joints)
        cur_hdg = agent.hdg
        cur_pos = agent.pos
        current_timestep += 1 # for planner
        print("Agent at t=" + str(t) + " is at " + str(cur_pos))

        ## Per step output for animation
        goal_radius = 200
        fig, _ = plt.subplots(2, 3, figsize=(12, 4))
        fig.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
        
        # XY plot
        ax = plt.subplot(131)
        agentcolors = ["tab:orange", "red", "green", "blue", "silver", "gray"]
        # Plot plan
        if input_theta:
            # plot projected positions
            dec_plan = np.asarray([np.cos(last_plan[:, 0])*last_plan[:, 1], 0.5+np.sin(last_plan[:, 0])*last_plan[:, 1]])
            x_plan = np.squeeze(dec_plan[0, :])
            y_plan = np.squeeze(dec_plan[1, :])
        else:
            # directly plot future plan
            x_plan = np.squeeze(last_plan[:, 0])
            y_plan = np.squeeze(last_plan[:, 1])
        if input_rpos:
            x_plan += init_joints[0]
            y_plan += init_joints[1]
        if input_source_file is None:
            ax.scatter(x_plan, y_plan) # plan
            ax.plot(x_plan, y_plan) # plan
        else:
            # recolor the plan based on the goal signal
            # convert analog signal to softmax bins
            for i in range(last_plan.shape[0]):
                task_bins = sm_enc(last_plan[i, -2], sm_bins=sm_goal_bins, sm_temp=sm_goal_temp, min=sm_goal_range[0], max=sm_goal_range[1])
                if task_bins[2] > goal_signal_threshold:   # goal
                    gcolor = agentcolors[2]
                elif task_bins[1] > goal_signal_threshold: # cw
                    gcolor = agentcolors[1]
                elif task_bins[0] > goal_signal_threshold: # ccw
                    gcolor = agentcolors[3]
                else: # unknown
                    gcolor = agentcolors[-2]
                opacity = 1.0 if i > min(t, static_plan_steps) else (i+1)/(min(t, static_plan_steps)+1)
                ax.scatter(x_plan[i], y_plan[i], color=gcolor, alpha=opacity, s=3) # plan
            # plt.plot(x_plan, y_plan) # plan
            if gcolor == agentcolors[2]: # plot goal if existing
                ax.scatter(last_plan[-1, -3], 0.9, s=goal_radius, facecolors="none", edgecolors="green")
        # Plot history
        if input_theta:
            # plot projected positions
            dec_robot = np.asarray(project_pos_plan(project_position_from, robot_joint_history[:plan_end]))
            x_robot = np.squeeze(dec_robot[0, :t+1])
            y_robot = np.squeeze(dec_robot[1, :t+1])
        else:
            # directly plot future plan
            x_robot = np.squeeze(robot_joint_history[:t+1, 0])
            y_robot = np.squeeze(robot_joint_history[:t+1, 1])

        if input_source_file is None:
            ax.scatter(x_robot, y_robot, facecolors=agentcolors[0]) # agent traj    
            ax.plot(x_robot, y_robot) # agent traj
            # plt.scatter(goal[0], goal[1], s=goal_radius, facecolors="none", edgecolors="green") # goal
        else:
            cur_task = sm_enc(last_plan[min(t, static_plan_steps), -2], sm_bins=sm_goal_bins, sm_temp=sm_goal_temp, min=sm_goal_range[0], max=sm_goal_range[1])
            if cur_task[2] > goal_signal_threshold:   # goal
                acolor = agentcolors[2]
            elif cur_task[1] > goal_signal_threshold: # cw
                acolor = agentcolors[1]
            elif cur_task[0] > goal_signal_threshold: # ccw
                acolor = agentcolors[3]
            else: # unknown
                acolor = agentcolors[-1]
            ax.plot(x_robot, y_robot, color=agentcolors[-1]) # agent traj
            ax.plot(x_robot[t], y_robot[t], marker=(3, 0, math.degrees(agent.hdg)), markersize=12, color=agentcolors[-1], mfc=acolor) # highlight current position
        if barrier_box is not None:
            ax.add_patch(patches.Rectangle((barrier_box[0], barrier_box[1]), barrier_box[2], barrier_box[3], edgecolor="black", facecolor="None", fill=False))
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

        # Show current goal signal condition
        gbinax = [None] * (sm_goal_bins+1)
        gbinax[0] = ax.inset_axes([0.925, 0.07, 0.05, 0.1])
        gbinax[0].set_xlabel("↺")
        gbinax[1] = ax.inset_axes([0.85, 0.07, 0.05, 0.1])
        gbinax[1].set_xlabel("↻")
        gbinax[2] = ax.inset_axes([0.775, 0.07, 0.05, 0.1])
        gbinax[2].set_xlabel("↸")
        gbinax[3] = ax.inset_axes([0.55, 0.07, 0.2, 0.04])
        gbinax[3].set_xlabel("Goal x")
        for a in gbinax:
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_xticks([])
            a.set_yticks([])
            a.set_ylim(bottom=0.0, top=1.0)
            a.set_xlim(left=0.0, right=1.0)
        gbinax[0].bar(0, [cur_task[0]], 1.0, align="edge", color=agentcolors[3] if cur_task[0] > goal_signal_threshold else agentcolors[-1]) # CCW
        gbinax[1].bar(0, [cur_task[1]], 1.0, align="edge", color=agentcolors[1] if cur_task[1] > goal_signal_threshold else agentcolors[-1]) # CW
        gbinax[2].bar(0, [cur_task[2]], 1.0, align="edge", color=agentcolors[2] if cur_task[2] > goal_signal_threshold else agentcolors[-1]) # GR
        gbinax[3].barh(0, [last_plan[min(t, static_plan_steps), -3]], 1.0, align="edge", color=agentcolors[2] if cur_task[2] > goal_signal_threshold else agentcolors[-1]) # Goal X

        ax.set_title("XY trajectory")
        # Determine goal step FIXME: this should be independent of plotting
        if stop_signal is not None:
            try:
                if not stop_signal_getmax:
                    stop_at = next(i for i,v in enumerate(last_plan[stop_signal_startscan:,stop_signal]) if v > stop_signal_threshold) + stop_offset + plan_start_offset
                else:
                    # First convert the low level signal into a softmax distribution
                    sm_stop_signal = softmax(last_plan[stop_signal_startscan:, stop_signal]/sm_stop_temp)
                    last_plan[stop_signal_startscan:, stop_signal] = sm_stop_signal # overwrite
                    # Then decode softmax into scalar value
                    stop_at = int(math.ceil(sm_dec(sm_stop_signal, max=rnn.max_timesteps-1))) + stop_offset + plan_start_offset
                    # if stop_at < stop_signal_startscan or sm_stop_signal[stop_at-stop_offset-plan_start_offset-stop_signal_startscan+1] < stop_signal_threshold:
                    if stop_at < stop_signal_startscan or np.amax(sm_stop_signal[stop_signal_startscan:]) < stop_signal_threshold:
                        stop_at = "--"
            except StopIteration:
                stop_at = "--"
            ax.text(1.07, 1.04, "Current step / Expected goal step: " + str(t) + " / " + str(stop_at), horizontalalignment="right")
            if stop_at != "--" and t >= stop_at:
                run_planner = False
                print("RNN signalled stop at", stop_at, "with probability", sm_stop_signal[stop_at-stop_offset-plan_start_offset-stop_signal_startscan+1])
                input("Press Enter to exit\n")

        # Plan output
        ax = plt.subplot(232)
        ax.set_xlim((plan_start_offset, last_plan.shape[0]+plan_start_offset+1))
        ax.set_ylim((-1.5, 1.5))
        ax.set_xticklabels([])
        x_plot = np.linspace(1+plan_start_offset, last_plan.shape[0]+plan_start_offset, last_plan.shape[0]) # timesteps on X
        for d in range(last_plan.shape[1]):
            y_plot = last_plan[:, d]
            if input_rpos and (d == 0 or d == 1): ## hack for XY
                y_plot -= init_joints[d]
            ax.plot(np.squeeze(x_plot), np.squeeze(y_plot))
        ax.axvline(x=t+1, linewidth=0.5, color="black") # current timestep
        ax.set_title("Plan")

        # Agent sense
        ax = plt.subplot(235)
        ax.set_xlim((plan_start_offset, plan_end+1))
        ax.set_ylim((-1.5, 1.5))
        # ax.set_xticklabels([])
        x_plot = np.linspace(1+plan_start_offset, t+1, t-plan_start_offset+1) # timesteps on X
        for d in range(robot_joint_history.shape[1]):
            y_plot = robot_joint_history[plan_start_offset:t+1, d]
            ax.plot(np.squeeze(x_plot), np.squeeze(y_plot))
        
        for d in range(signals):
            y_plot = robot_signal_history[plan_start_offset:t+1, d]
            ax.plot(np.squeeze(x_plot), np.squeeze(y_plot))
        plan_end_offset = max((plan_end-t) - (max_plan_length-t), 0)
        x_plot2 = np.linspace(t+1, min(plan_end-plan_end_offset, max_plan_length),  min(plan_end-plan_end_offset, max_plan_length)-t) # remaining timesteps
        # for d in range(signals):
        #     y_plot2 = robot_signal_history[t:plan_end, d]
        #     color2 = "tab:green" if d == 0 else "tab:red"
        #     plt.plot(np.squeeze(x_plot2), np.squeeze(y_plot2), linestyle=':', color=color2) # goal signal
        ax.axvline(x=t+1, linewidth=0.5, color="black") # current timestep

        ax.set_title("Proprioception")

        # Z activity
        ax = plt.subplot(236)
        ax.set_xlim((plan_start_offset, last_plan.shape[0]+plan_start_offset+1))
        ax.set_ylim((-0.1, 1.1))
        # ax.set_xticklabels([])
        x_plot = np.linspace(0, t, num=t) # timesteps on X
        # x_plot = np.linspace(1, last_plan.shape[0], last_plan.shape[0]) # timesteps on X
        opacity = 1.0
        for a in range(rnn.n_layers-1):
            y_plot = np.asarray(zinfo)[:, a]
            # if (len(generated_src) == rnn.max_timesteps):
            #     gsc = np.asarray(generated_src)[:,a]
            # else:
            #     print(generated_src)
            #     print(np.shape(generated_src))
            #     gsc[rnn.max_timesteps-len(generated_src):] = np.asarray(generated_src)[:-(rnn.max_timesteps-len(generated_src)),a]
            # y_plot = gsc
            plt.plot(x_plot, y_plot, color=(0.8, 0.5, 0.8, opacity))
            opacity /= 2.0
        ax.axvline(x=t+1, linewidth=0.5, color="black") # current timestep
        ax.set_title("Z information")

        # Free Energy
        ax = plt.subplot(233)
        lstep = current_timestep + 1
        evidence_window_size = min(t, static_plan_steps+1)
        plan_window_size = rnn.max_timesteps - evidence_window_size
        xloss = np.asarray(full_loss["batch_reconstruction_loss"])
        zloss = np.asarray(full_loss["batch_regularization_loss"])
        evfe = ((np.sum(xloss[:lstep]) + np.sum(zloss[:, :lstep-plan_start_offset])) * rnn.max_timesteps) / evidence_window_size # re-normalize
        esfe = ((np.sum(xloss[lstep:]) + np.sum(zloss[:, lstep-plan_start_offset:])) * rnn.max_timesteps) / plan_window_size
        evidence_fe.append(evfe)
        estimated_fe.append(esfe)
        ax.set_xlim((plan_start_offset, last_plan.shape[0]+plan_start_offset+1))
        ax.set_xticklabels([])
        x_plot = np.linspace(0, t, num=t) # timesteps on X
        y_plot = np.asarray(evidence_fe)
        ax.plot(x_plot, y_plot, color="tab:purple")
        ax.set_ylim(bottom=-0.1, top=6.1)
        ax.set_ylabel("Evidence Free Energy", color="tab:purple")
        ax2 = ax.twinx()
        y2_plot = np.asarray(estimated_fe)
        ax2.plot(x_plot, y2_plot, color="tab:olive")
        ax2.set_ylim(bottom=-0.1, top=6.1)
        ax2.set_ylabel("Estimated Free Energy", color="tab:olive")
        ax.axvline(x=t+1, linewidth=0.5, color="black") # current timestep
        ax.set_title("Free Energy")

        # Save
        if not os.path.exists(basedir + output_subdir):
            os.makedirs(basedir + output_subdir)
        # fig.tight_layout()
        fig.savefig(basedir + output_subdir + "/output_" + str(current_timestep) + ".png", dpi=150)
        plt.close(fig)

        if live_dashboard:
            frame = cv2.imread(basedir + output_subdir + "/output_" + str(current_timestep) + ".png")
            cv2.imshow("GLean planner dashboard", frame)
            cv2.waitKey(1)
            dash_avi.write(frame)
    
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    run_planner = False

    if plan_save_data is not None:
        # save_data is [epoch, save_arr, steps, dims] EXCEPT zloss is [epoch, dims, steps] for some reason
        # First slice by save_arr
        all_data = np.asarray(plan_save_data)
        all_plan = np.asarray(all_data[:, 0]) # active plan
        all_rplan = np.asarray(all_data[:, 1]) # raw generated plan
        all_pos = np.asarray(all_data[:, 2]) # actual agent positions
        all_sig = np.asarray(all_data[:, 3]) # actual signals
        all_internal = np.asarray(all_data[:, 4]) # Z activity
        all_xloss = np.asarray(all_data[:, 5]) # rec loss
        all_zloss = np.asarray(all_data[:, 6]) # reg loss

        # Then turn them into real ndarrays
        save_plan = np.array([np.array(x) for x in all_plan])
        save_rplan = np.array([np.array(x) for x in all_rplan])
        save_pos = np.array([np.array(x) for x in all_pos])
        save_sig = np.array([np.array(x) for x in all_sig])
        save_internal = np.array([np.array(x) for x in all_internal])
        save_xloss = np.array([np.array(x) for x in all_xloss])
        save_zloss = np.array([np.array(x) for x in all_zloss])
        save_zloss = np.array([z.T for z in save_zloss]) # zloss needs to be reordered
        np.savez_compressed("2dagent_plan_data.npz", plan=save_plan, rnn_plan=save_rplan, pos=save_pos, sig=save_sig, internal=save_internal, xloss=save_xloss, zloss=save_zloss)
        print("Saved", save_plan.shape, save_rplan.shape, save_pos.shape, save_sig.shape, save_internal.shape, save_xloss.shape, save_zloss.shape)

except Exception as e:
    print("Exception occurred in robot control!")
    traceback.print_exc()
finally:
    run_planner = False
    dash_avi.release()
    cv2.destroyAllWindows()
    time.sleep(1)
