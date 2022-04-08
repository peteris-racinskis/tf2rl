import os
from re import S

from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.gaifo import GAIfO
from tf2rl.envs.dummy_env import DummyEnv
from tf2rl.experiments.custom_trainer import CustomTrainer
from tf2rl.experiments.utils import load_csv_dataset
import tensorflow as tf


FILENAME="/home/user/repos/masters/processed_data/train_datasets/train-003430811ff20c35ccd5.csv"
cdir = os.path.dirname("results/20220408T111648.999757_DDPG_GAIfO/ckpt-10")
latest = tf.train.latest_checkpoint(cdir)


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
    irl = GAIfO(
        state_shape=env.observation_space.shape,
        units=units,
        enable_sn=args.enable_sn,
        batch_size=32,
        gpu=args.gpu)

    expert_trajs = load_csv_dataset(FILENAME)
    initial_states = expert_trajs["obses"]
    # YOU DUMB FUCK, THE DDPG CLASS IS NOT INHERITING FROM MDOEL
    # COMPOSITION not INHERITANCE
    # In this case, it has two models - actor and critic
    # AS per the actor/critic model, the actor is the actual policy,
    # the ciritic provides an estimate of the expected reward at train
    # time.
    checkpoint = tf.train.Checkpoint(actor=policy.actor, critic=policy.critic)
    checkpoint.restore(latest)
    x = env.reset(initial_states)
    policy.get_action(env.observation_space.sample())
    pass