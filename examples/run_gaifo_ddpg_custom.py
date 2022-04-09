from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.gaifo import GAIfO
from tf2rl.experiments.custom_trainer import CustomTrainer
from tf2rl.experiments.utils import load_csv_dataset
from tf2rl.envs.dummy_env import DummyEnv

FILENAME="/home/user/repos/masters/processed_data/train_datasets/train-003430811ff20c35ccd5.csv"


if __name__ == '__main__':
    parser = CustomTrainer.get_argument()
    parser = GAIfO.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="DummyEnv")
    parser.add_argument('--max-steps', type=str, default=1000000)
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
        n_warmup=0,
        batch_size=100)
    irl = GAIfO(
        state_shape=env.observation_space.shape,
        units=units,
        enable_sn=args.enable_sn,
        batch_size=32,
        gpu=args.gpu)
    expert_trajs = load_csv_dataset(FILENAME)
    initial_states = expert_trajs["obses"]
    trainer = CustomTrainer(initial_states, policy, env, args, irl, expert_trajs["obses"],
                         expert_trajs["next_obses"], expert_trajs["acts"], test_env)
    trainer()
