import os
import random

from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.gaifo import GAIfO
from tf2rl.envs.dummy_env import DummyEnv
from tf2rl.experiments.custom_trainer import CustomTrainer
from tf2rl.experiments.utils import load_csv_dataset
import tensorflow as tf
import pandas as pd
import numpy as np

FILENAME="/home/user/repos/masters/processed_data/train_datasets/train-003430811ff20c35ccd5.csv"
OUTPUT="results/generated/generated-100k.csv"
cdir = os.path.dirname("results/20220409T123851.835836_DDPG_GAIfO")
#cdir = "/home/user/repos/tf2rl/results/20220409T132003.050239_DDPG_GAIfO"
latest = tf.train.latest_checkpoint(cdir)


def collect_episode(initial: np.ndarray, policy: DDPG, env: DummyEnv):
    obses = []
    obs = env.reset(initial)
    done = False
    while not done:
        obses.append(obs)
        action = policy.get_action(obs)
        obs, _, done, __ = env.step(action)
    return np.stack(obses)

def get_initial_states(train_set):
    all_states = train_set["obses"]
    selection = random.sample(range(len(all_states)), 50)
    return all_states[selection]

def generate_trajectories(initial_states, model, env):
    trajectories = []
    for init in initial_states:
        ep = collect_episode(init, model, env)
        trajectories.append(ep)
    data = np.concatenate(trajectories)
    df = pd.DataFrame(data=data, columns=["x", "y", "z", "rx", "ry", "rz", "rw", "rel", "tx", "ty", "tz"])
    return df



if __name__ == "__main__":
    parser = CustomTrainer.get_argument()
    parser = GAIfO.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="DummyEnv")
    parser.add_argument('--max-steps', type=str, default=100000)
    args = parser.parse_args()

    units = [400, 300]

    env = DummyEnv()
    test_env = DummyEnv()
    policy = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        max_action=env.action_space.high[0],
        gpu=args.gpu,
        lr_actor=10e-5,
        lr_critic=10e-5,
        actor_units=units,
        critic_units=units,
        n_warmup=1000,
        batch_size=100)

    expert_trajs = load_csv_dataset(FILENAME)
    # YOU DUMB FUCK, THE DDPG CLASS IS NOT INHERITING FROM MDOEL
    # COMPOSITION not INHERITANCE
    # In this case, it has two models - actor and critic
    # AS per the actor/critic model, the actor is the actual policy,
    # the ciritic provides an estimate of the expected reward at train
    # time.
    checkpoint = tf.train.Checkpoint(actor=policy.actor, critic=policy.critic)
    checkpoint.restore(latest)


    initial_states = get_initial_states(expert_trajs)
    df = generate_trajectories(initial_states, policy, env)
    df.to_csv(OUTPUT)