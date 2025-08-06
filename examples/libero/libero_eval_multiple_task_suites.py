import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import datetime, sys, os
from typing_extensions import Annotated
from typing import List

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    # task_suite_name: str = (
    #     "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    # )
    task_suite_names: Annotated[
        List[str],
        "List of task suites to evaluate, e.g., libero_10, libero_goal, libero_spatial. Pass one or more."
    ] = dataclasses.field(default_factory=list)
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    log_file_path: str = "data/libero/libero_eval.log"  # Path to log file

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(task_suite_name: str, args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Thuan: tracking individual task success rate within this task suite
    per_task_results = {}

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        # Thuan: recording this task's success rate 
        per_task_results[task_description] = float(task_successes) / float(task_episodes)

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    # Thuan: returning this task suite's results for logging into a log file
    this_task_results = {
        'task_suite': task_suite_name,
        'total_episodes': total_episodes,
        'total_successes': total_successes,
        'task_suite_success_rate': float(total_successes) / float(total_episodes),
        'per_task_success_rate': per_task_results
    }

    return this_task_results


# Thuan's addition to evaluate multiple task suites
def eval_multiple_task_suites(task_suite_names: List[str], args: Args) -> dict:
    """This function will loop through `task_suite_names` (an array of 'libero_spatial', 'libero_object', etc.) in order to call `eval_libero()` on each of them, outputing the results to both `stdout` and a log file."""
    all_results = {}

    for task_suite_name in task_suite_names:
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting evaluation on {task_suite_name}")
        logging.info(f"{'='*50}")

        try:
            results = eval_libero(task_suite_name, args)
            all_results[task_suite_name] = results

            # completed, and now logging the results for this task suite
            logging.info(f"{'='*50}")
            logging.info(f"Completed the entire task suite: {task_suite_name}")
            for k, v in results.items():
                if isinstance(v, dict):
                    logging.info("    Per-task results:")
                    for sub_k, sub_v in v.items():
                        logging.info(f"        {sub_k}: {sub_v}")
                else:
                    logging.info(f"    {k}: {v}")
            logging.info(f"{'='*50}")
        except Exception as e:
            logging.error(f"Failed on {task_suite_name}: {e}")
            all_results[task_suite_name] = {'error': str(e)}

    return all_results

        


# Thuan: set up logging to file as well and not just stdout
def setup_logging(log_file_path: str):
    """Set up logging to file as well and not just stdout"""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s - %(message)s'
    )

    # File handler and console handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure the root logger
    logging.basicConfig(
        handlers=[file_handler, console_handler],
        level=logging.INFO,
    )


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    # tyro.cli(eval_libero)

    args = tyro.cli(Args)

    # Set up logging to file
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = os.path.dirname(args.log_file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    setup_logging(args.log_file_path)

    all_results = eval_multiple_task_suites(args.task_suite_names, args)


    logging.info("Completed evaluating all task suites on LIBERO...")
    for task_suite_name, task_suite_results in all_results.items():
        logging.info(f"{'='*50}")
        logging.info(f"Summary for task suite: {task_suite_name}")
        for k, v in task_suite_results.items():
            if isinstance(v, dict):
                logging.info("    Per-task results:")
                for sub_k, sub_v in v.items():
                    logging.info(f"        {sub_k}: {sub_v}")
            else:
                logging.info(f"    {k}: {v}")
