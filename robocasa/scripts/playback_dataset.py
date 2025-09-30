import argparse
import json
import os
import random
import time

import h5py
import imageio
import numpy as np
import robosuite
from termcolor import colored
import numpy as np
from scipy.spatial.transform import Rotation as R
import robocasa
import pdb
from scipy.spatial.transform import Slerp


def playback_trajectory_with_env(
    env,
    initial_state,
    states,
    actions=None,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    first=False,
    verbose=False,
    camera_height=512,
    camera_width=512,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state.
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    write_video = video_writer is not None
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    ## this reset call doesn't seem necessary.
    ## seems ok to remove but haven't fully tested it.
    ## removing for now
    # env.reset()

    if verbose:
        ep_meta = json.loads(initial_state["ep_meta"])
        lang = ep_meta.get("lang", None)
        if lang is not None:
            print(colored(f"Instruction: {lang}", "green"))
        print(colored("Spawning environment...", "yellow"))
    reset_to(env, initial_state)

    traj_len = states.shape[0]
    action_playback = actions is not None
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    if render is False:
        print(colored("Running episode...", "yellow"))

    # sped_up_actions = # Slow down the trajectory by a factor of 5
    # actions = smoothe_trajectory(actions, cutoff_ratio=0.1)
    # pdb.set_trace()
    arm_actions = actions[:, :7]
    # pdb.set_trace()
    slowed_trajectory = slow_down_trajectory(arm_actions, factor=2)
    # stack [0,0,0,0,-1] for the dimension of the new slowed trajectory
    zero_base = np.stack([[0, 0, 0, 0, -1]] * slowed_trajectory.shape[0], axis=0)
    slowed_trajectory = np.concatenate([slowed_trajectory, zero_base], axis=1)  # concatenate the zero base to the slowed trajectory
    actions = slowed_trajectory
    traj_len = actions.shape[0]  # update trajectory length after slowing down
    cumulative_xyz = np.zeros(3)
    for i in range(traj_len):
        start = time.time()
        #

        if action_playback:
            env.step(actions[i])
            cumulative_xyz += actions[i][:3]
            print("cumulative_xyz", cumulative_xyz)
            if i < traj_len - 1:
                # check whether the actions deterministically lead to the same recorded states
                state_playback = np.array(env.sim.get_state().flatten())
                # if not np.all(np.equal(states[i + 1], state_playback)):
                #     err = np.linalg.norm(states[i + 1] - state_playback)
                #     if verbose or i == traj_len - 2:
                #         print(
                #             colored(
                #                 "warning: playback diverged by {} at step {}".format(
                #                     err, i
                #                 ),
                #                 "yellow",
                #             )
                #         )
        else:
            reset_to(env, {"states": states[i]})

        # on-screen render
        if render:
            if env.viewer is None:
                env.initialize_renderer()

            # so that mujoco viewer renders
            env.viewer.update()

            max_fr = 60
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    im = env.sim.render(
                        height=camera_height, width=camera_width, camera_name=cam_name
                    )[::-1]
                    video_img.append(im)
                video_img = np.concatenate(
                    video_img, axis=1
                )  # concatenate horizontally
                video_writer.append_data(video_img)

            video_count += 1

        if first:
            break

    if render:
        env.viewer.close()
        env.viewer = None


def playback_trajectory_with_obs(
    traj_grp,
    video_writer,
    video_skip=5,
    image_names=None,
    first=False,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert (
        image_names is not None
    ), "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["obs/{}".format(image_names[0] + "_image")].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k + "_image")][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break


def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        # pdb.set_trace()
        # if "env_args" in f["data"].attrs:
        #     env_meta = json.loads(f["data"].attrs["env_args"])
        # else:
        #     print()
        env_meta = json.loads(f["data"].attrs["env_args"])
    else:
        raise ValueError
    f.close()
    return env_meta


class ObservationKeyToModalityDict(dict):
    """
    Custom dictionary class with the sole additional purpose of automatically registering new "keys" at runtime
    without breaking. This is mainly for backwards compatibility, where certain keys such as "latent", "actions", etc.
    are used automatically by certain models (e.g.: VAEs) but were never specified by the user externally in their
    config. Thus, this dictionary will automatically handle those keys by implicitly associating them with the low_dim
    modality.
    """

    def __getitem__(self, item):
        # If a key doesn't already exist, warn the user and add default mapping
        if item not in self.keys():
            print(
                f"ObservationKeyToModalityDict: {item} not found,"
                f" adding {item} to mapping with assumed low_dim modality!"
            )
            self.__setitem__(item, "low_dim")
        return super(ObservationKeyToModalityDict, self).__getitem__(item)


def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        # import pdb; pdb.set_trace()
        env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    # if should_ret:
    #     # only return obs if we've done a forward call - otherwise the observations will be garbage
    return env._get_observations()
    # return None


# Helper function to interpolate between two values
def linear_interpolate(start, end, t):
    return start + (end - start) * t


# Slerp interpolation for quaternions
def slerp(quat1, quat2, t):
    # Perform Slerp (Spherical Linear Interpolation) for quaternions
    r1 = R.from_quat([quat1, quat2])
    r2 = R.from_quat(quat2)
    from scipy.spatial.transform import Slerp

    # pdb.set_trace()
    times = np.array([0, 1])
    # keyrots = np.array([r1, r2])
    # setup key rots and a scipy.transform.rotation object
    # keyrots = np.array([r1.as_quat(), r2.as_quat()])
    keyrots = R.from_quat([quat1, quat2])

    slerp_result = Slerp(times, keyrots)
    interp_times = np.arange(0, 1, 1 / t)
    interp_rots = slerp_result(interp_times)
    # convert to rotation vector
    interp_rots = interp_rots.as_rotvec()
    return interp_rots


# Manual Slerp interpolation for quaternions
def manual_slerp(quat1, quat2, t):
    # Convert quaternions to Rotation objects
    r1 = R.from_quat(quat1)
    r2 = R.from_quat(quat2)

    # Compute the dot product between the two quaternions
    dot_product = np.dot(r1.as_quat(), r2.as_quat())

    # If dot product is negative, negate one quaternion to get the shortest path
    if dot_product < 0.0:
        r2 = R.from_quat(-r2.as_quat())
        dot_product = -dot_product

    # If the dot product is very close to 1, use linear interpolation
    if dot_product > 0.9995:
        return R.slerp(t, [r1, r2]).as_quat()[0]

    # Compute the angle between the quaternions
    theta_0 = np.arccos(dot_product)
    theta = theta_0 * t

    # Compute the slerp
    q3 = r2.as_quat() - r1.as_quat() * dot_product
    q3 /= np.linalg.norm(q3)

    return r1.as_quat() * np.cos(theta) + q3 * np.sin(theta)


# Function to interpolate between two relative joint actions
def interpolate_actions(action1, action2, t):
    # Extract the components of each action (Position, Rotation, Gripper Pose)
    # pdb.set_trace()
    position1, rotation1, gripper1 = action1[:3], action1[3:6], action1[6]
    position2, rotation2, gripper2 = action2[:3], action2[3:6], action2[6]

    # Interpolate for position (linear interpolation)
    # interpolated_positions = [linear_interpolate(position1[i], position2[i], t) for i in range(3)]

    # linear interpolation between positions for a factor of t
    t_values = np.linspace(0, 1, t)

    # Interpolate linearly for each value of t
    interpolated_positions = np.array(
        [position1 + t * (position2 - position1) for t in t_values]
    )

    # Instead, interpolate by dividing by the number of steps

    # convert 3d rotation to quaternions
    quat1 = R.from_rotvec(rotation1).as_quat()
    quat2 = R.from_rotvec(rotation2).as_quat()

    # Interpolate for rotation (Slerp interpolation for quaternions)
    interpolated_rotations = slerp(quat1, quat2, t)

    # convert back to rotation vector
    # interpolated_rotations = R.from_quat(interpolated_rotations).as_rotvec()

    # Interpolate for rotation (Slerp interpolation for quaternions)
    interpolated_gripper = np.array(
        [gripper1 + t * (gripper2 - gripper1) for t in t_values]
    )

    # interpolated_positions = np.array(interpolated_positions)
    interpolated_gripper = np.expand_dims(
        interpolated_gripper, axis=1
    )  # make it a column vector
    # pdb.set_trace()
    # Return the interpolated action
    return np.concatenate(
        [interpolated_positions, interpolated_rotations, interpolated_gripper], axis=1
    )


# Slow down trajectory by inserting interpolated actions between each pair of consecutive actions
def slow_down_trajectory_old(actions, factor=2):
    new_trajectory = []

    for i in range(len(actions) - 1):
        # Get the two consecutive actions
        action1 = actions[i]
        action2 = actions[i + 1]

        # Insert the original action
        # new_trajectory.append(action1)

        # Insert interpolated actions to slow down (factor defines the number of interpolated steps)
        # for t in range(1, factor):  # Interpolate `factor-1` times between two actions
        interpolated_action = interpolate_actions(action1, action2, factor)
        new_trajectory.extend(list(interpolated_action))

    # Append the last action
    new_trajectory.append(actions[-1])

    # Compute relative actions (difference between consecutive steps)
    # new_trajectory = [new_trajectory[i+1] - new_trajectory[i] for i in range(len(new_trajectory) - 1)]

    # pdb.set_trace()

    return np.array(new_trajectory)


def interpolate_relative_rotations(quaternions, num_steps_between=10):
    """
    Perform slerp interpolation on a list of relative rotation quaternions.

    Args:
        quaternions (list or np.ndarray): List of quaternions (Nx4).
        num_steps_between (int): Number of interpolation steps between original quaternions.

    Returns:
        List of interpolated quaternions including originals.
    """
    key_times = np.arange(len(quaternions))
    rotations = R.from_quat(quaternions)

    # Interpolation times
    interp_times = np.linspace(
        0, len(quaternions) - 1, num=(len(quaternions) - 1) * num_steps_between + 1
    )

    # Slerp interpolator
    slerp = Slerp(key_times, rotations)
    interp_rots = slerp(interp_times)

    return interp_rots.as_rotvec()


def slow_down_trajectory(actions, factor=2):
    new_trajectory = []
    cumulative_xyz = np.zeros(3)
    for i in range(len(actions)):
        # Get the two consecutive actions
        action1 = actions[i]
        # action2 = actions[i + 1]

        # Insert the original action
        # new_trajectory.append(action1)

        # Insert interpolated actions to slow down (factor defines the number of interpolated steps)
        # for t in range(1, factor):  # Interpolate `factor-1` times between two actions
        # interpolated_action = interpolate_actions(action1, action2, factor)
        # just append to new_trajectory the actions divided by factor
        interpolated_action = action1 / factor

        # for _ in range(factor - 1):
        #     # Insert the interpolated action
        #     new_trajectory.append(interpolated_action)
        # new_trajectory.append(list(interpolated_action))

        # quat1 = R.from_rotvec(action1[3:6]).as_quat()
        # # quat2 = R.from_rotvec(action2[3:6]).as_quat()
        # identity = R.from_quat([[0, 0, 0, 1]])
        # quat1 = R.from_quat(quat1)
        # # pdb.set_trace()
        # slerp = Slerp([0, 1], R.concatenate([identity, quat1]))
        # partial_rot = slerp([1 / factor])
        # scaled_quat = partial_rot.as_rotvec()[0]
        # interpolated_action[
        #     3:6
        # ] = scaled_quat  # set the rotation part to the scaled quaternion
        # interpolated_quats = interpolate_relative_rotations([quat1, quat2], num_steps_between=factor)
        # convert back to rotation vector
        # interpolated_quats = R.from_quat(interpolated_quats).as_rotvec
        # # convert to numpy array
        # interpolated_quats = np.array(interpolated_quats)
        # pdb.set_trace()
        cumulative_action = np.zeros_like(interpolated_action)
        for _ in range(factor):  # Insert (factor-1) interpolated actions
            # tmp_interpolated_action = interpolated_action
            # # set the rotation part to the interpolated quaternion
            # tmp_interpolated_action[3:6] = interpolated_quats[_]
            if sum(np.abs(interpolated_action[:3])) < 0.01:
                # add bit forward motion to x and y
                tmp_interpolated_action = np.copy(interpolated_action)
                tmp_interpolated_action[0] += 0.1
                tmp_interpolated_action[1] += 0.1
                interpolated_action = tmp_interpolated_action
            # new_trajectory.append(tmp_interpolated_action)
            if cumulative_xyz[2] < -26 and i < len(actions)/2:
                import copy
                tmp_interpolated_action = copy.deepcopy(interpolated_action)
                tmp_interpolated_action[2] = 0
                new_trajectory.append(tmp_interpolated_action)
                cumulative_action += tmp_interpolated_action
                cumulative_xyz += tmp_interpolated_action[:3]
            else:
                new_trajectory.append(interpolated_action)
                cumulative_action += interpolated_action
                cumulative_xyz += interpolated_action[:3]
            print("cumulative_xyz", cumulative_xyz)
            # Identity rotation (no rotation)
        # the final relative action added should be the initial action minus the cumulative action to account for the numerical error
        # new_trajectory.append(action1 - cumulative_action)

        # pdb.set_trace()
        # new_trajectory.extend(list(interpolated_quats))

    # Append the last action
    # new_trajectory.append(actions[-1])

    # Compute relative actions (difference between consecutive steps)
    # new_trajectory = [new_trajectory[i+1] - new_trajectory[i] for i in range(len(new_trajectory) - 1)]

    # pdb.set_trace()

    return np.array(new_trajectory)


def fft_smooth_trajectory(relative_xyz, cutoff_ratio=0.1):
    """
    Smooths a trajectory of relative xyz actions using FFT filtering.

    Args:
        relative_xyz (np.ndarray): Array of shape (T, 3) with relative xyz actions.
        cutoff_ratio (float): Ratio of frequencies to keep (0 < ratio < 1). Lower = smoother.

    Returns:
        np.ndarray: Smoothed relative xyz actions (same shape).
    """
    T, D = relative_xyz.shape
    smoothed = np.zeros_like(relative_xyz)

    # For each dimension (x, y, z)
    # for d in range(D):
    #     signal = relative_xyz[:, d]
    #     freq = np.fft.fft(signal)

    #     # Zero out high-frequency components
    #     cutoff = int(T * cutoff_ratio)
    #     freq[cutoff:T - cutoff] = 0  # keep low frequencies only

    #     # Inverse FFT to get smoothed signal
    #     smoothed[:, d] = np.fft.ifft(freq).real
    for d in range(D):
        signal = relative_xyz[:, d]
        freq = np.fft.fft(signal)
        magnitude = np.abs(freq)

        # Compute threshold
        std = np.std(magnitude)
        threshold = 2 * std

        # Zero out low-magnitude frequency components
        freq_filtered = np.where(magnitude > threshold, freq, 0)

        # Inverse FFT
        smoothed[:, d] = np.fft.ifft(freq_filtered).real

    return smoothed


def smoothe_trajectory_old(actions, cutoff_ratio=0.1):
    xyz = actions[:, :3]  # extract xyz positions
    smoothed_xyz = fft_smooth_trajectory(
        xyz, cutoff_ratio=cutoff_ratio
    )  # smooth xyz positions
    new_actions = np.concatenate(
        [smoothed_xyz, actions[:, 3:]], axis=1
    )  # concatenate smoothed xyz with other action parts

    return new_actions

def smoothe_trajectory(actions, cutoff_ratio=0.1):
    xyz = actions[:, :3]  # extract xyz positions
    absolute_xyz = np.cumsum(xyz, axis=0)
    from scipy.signal import savgol_filter

    # window_length must be odd and > polyorder
    smoothed_xyz = savgol_filter(absolute_xyz, window_length=11, polyorder=3, axis=0)
    # convert back to relative actions
    smoothed_relative_xyz = np.diff(smoothed_xyz, axis=0, prepend=smoothed_xyz[0:1, :])

    new_actions = np.concatenate(
        [smoothed_relative_xyz, actions[:, 3:]], axis=1
    )  # concatenate smoothed xyz with other action parts


    return new_actions


def playback_dataset(args):
    # some arg checking
    write_video = args.render is not True
    if args.video_path is None:
        args.video_path = args.dataset.split(".hdf5")[0] + ".mp4"
        if args.use_actions:
            args.video_path = args.dataset.split(".hdf5")[0] + "_use_actions.mp4"
        elif args.use_abs_actions:
            args.video_path = args.dataset.split(".hdf5")[0] + "_use_abs_actions.mp4"
    assert not (args.render and write_video)  # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        args.render_image_names = "robot0_agentview_center"

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert (
            not args.use_actions and not args.use_abs_actions
        ), "playback with observations is offline and does not support action playback"

    env = None

    # create environment only if not playing back with observations
    if not args.use_obs:
        # # need to make sure ObsUtils knows which observations are images, but it doesn't matter
        # # for playback since observations are unused. Pass a dummy spec here.
        # dummy_spec = dict(
        #     obs=dict(
        #             low_dim=["robot0_eef_pos"],
        #             rgb=[],
        #         ),
        # )
        # initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        if args.use_abs_actions:
            env_meta["env_kwargs"]["controller_configs"][
                "control_delta"
            ] = False  # absolute action space

        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = True
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = True
        env_kwargs["use_camera_obs"] = False
        # env_kwargs["control_freq"] = 5

        if args.verbose:
            print(
                colored(
                    "Initializing environment for {}...".format(env_kwargs["env_name"]),
                    "yellow",
                )
            )

        env = robosuite.make(**env_kwargs)
        # import pdb; pdb.set_trace()

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    elif "data" in f.keys():
        demos = list(f["data"].keys())

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        random.shuffle(demos)
        demos = demos[: args.n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    for ind in range(len(demos)):
        ep = demos[ind]
        print(colored("\nPlaying back episode: {}".format(ep), "yellow"))

        if args.use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)],
                video_writer=video_writer,
                video_skip=args.video_skip,
                image_names=args.render_image_names,
                first=args.first,
            )
            continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)
        # pdb.set_trace()
        if args.extend_states:
            states = np.concatenate((states, [states[-1]] * 50))

        # supply actions if using open-loop action playback
        actions = None
        assert not (
            args.use_actions and args.use_abs_actions
        )  # cannot use both relative and absolute actions
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]
        elif args.use_abs_actions:
            actions = f["data/{}/actions_abs".format(ep)][()]  # absolute actions

        playback_trajectory_with_env(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
            verbose=args.verbose,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
        )

    f.close()
    if write_video:
        print(colored(f"Saved video to {args.video_path}", "green"))
        video_writer.close()

    if env is not None:
        env.close()


def get_env_from_dataset(dataset):
    # args = get_playback_args()
    # # some arg checking
    # write_video = args.render is not True
    # if args.video_path is None:
    #     args.video_path = args.dataset.split(".hdf5")[0] + ".mp4"
    #     if args.use_actions:
    #         args.video_path = args.dataset.split(".hdf5")[0] + "_use_actions.mp4"
    #     elif args.use_abs_actions:
    #         args.video_path = args.dataset.split(".hdf5")[0] + "_use_abs_actions.mp4"
    # assert not (args.render and write_video)  # either on-screen or video but not both

    # # Auto-fill camera rendering info if not specified
    # if args.render_image_names is None:
    #     # We fill in the automatic values
    #     env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
    #     args.render_image_names = "robot0_agentview_center"

    # if args.render:
    #     # on-screen rendering can only support one camera
    #     assert len(args.render_image_names) == 1

    # if args.use_obs:
    #     assert write_video, "playback with observations can only write to video"
    #     assert (
    #         not args.use_actions and not args.use_abs_actions
    #     ), "playback with observations is offline and does not support action playback"

    env = None

    # create environment only if not playing back with observations
    # if not args.use_obs:

    env_meta = get_env_metadata_from_dataset(dataset_path=dataset)
    # if args.use_abs_actions:
    env_meta["env_kwargs"]["controller_configs"][
        "control_delta"
    ] = False  # absolute action space

    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False

    # if args.verbose:
    #     print(
    #         colored(
    #             "Initializing environment for {}...".format(env_kwargs["env_name"]),
    #             "yellow",
    #         )
    #     )

    env = robosuite.make(**env_kwargs)
    return env


def get_playback_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action="store_true",
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action="store_true",
        help="use open-loop action playback instead of loading sim states",
    )

    # Playback stored dataset absolute actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-abs-actions",
        action="store_true",
        help="use open-loop action playback with absolute position actions instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action="store_true",
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs="+",
        default=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
        "None, which corresponds to a predefined camera for each env type",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action="store_true",
        help="use first frame of each episode",
    )

    parser.add_argument(
        "--extend_states",
        action="store_true",
        help="play last step of episodes for 50 extra frames",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log additional information",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=512,
        help="(optional, for offscreen rendering) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=512,
        help="(optional, for offscreen rendering) width of image observations",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_playback_args()
    playback_dataset(args)
