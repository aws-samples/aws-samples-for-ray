import argparse
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig

from sagemaker_ray_helper import RayHelper

# store checkpoints to local file directory path
CHECKPOINT_ROOT = f"/opt/ml/checkpoints"


def my_train_fn(configuration, reporter):
    """Training function for RL model. Here we use the Cart Pole Example. We pull the environment from the gym
    library. We use the PPO algorithm to execute iterations.

    :param configuration: (dict) Configuration dictionary for PPO
    :param reporter: (tune.report)
    :return:
    """
    iterations = configuration.pop("train-iterations", 10)

    ppo_config = PPOConfig().update_from_dict(configuration).environment("CartPole-v1")
    
    # set/add constant values
    # can include these values as constants or add as hyperparameters if desired
    # if uncommented these values will give you an episode_reward_mean of 500
#     ppo_config["lr"] = 0.001
#     ppo_config["gamma"] = 0.99
#     ppo_config["kl_coeff"] = 1.0
#     ppo_config["num_sgd_iter"] = 20
#     ppo_config["sgd_minibatch_size"] = 1000
#     ppo_config["train_batch_size"] = 25000

    agent = ppo_config.build()
    for i in range(iterations):
        result = agent.train()
        reporter(**result)
        # create custom logic to save checkpoint and check for stopping condition
        # every 5 iterations
        if i % 5 == 0:
            state = agent.save(CHECKPOINT_ROOT)
        # setup custom logic for stopping condition
        # stop iterations if reward is greater than 450
        if result["episode_reward_mean"] >= 450:
            break
    state = agent.save(CHECKPOINT_ROOT)
    agent.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="tf",
        help="The DL framework specifier.",
    )
    parser.add_argument(
        "--train-iterations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=95,
    )
    parser.add_argument(
        "--model_dir",
        type=str
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99
    ),
    parser.add_argument(
        "--kl_coeff",
        type=float,
        default=1.0
    ),
    parser.add_argument(
        "--num_sgd_iter",
        type=int,
        default=20
    )

    args = parser.parse_args()

    ray_helper = RayHelper()
    ray_helper.start_ray()

    cluster_resources = ray.cluster_resources()
    print(f"all cluster resources = {cluster_resources}")

    config = {
        "train-iterations": args.train_iterations,
        "num_workers": args.num_workers,
        "framework": args.framework,
        "lr": args.lr,
        "gamma": args.gamma,
        "kl_coeff": args.kl_coeff,
        "num_sgd_iter": args.num_sgd_iter
    }

    resources = PPO.default_resource_request(config)
    tuner = tune.Tuner(
        tune.with_resources(my_train_fn, resources=resources), param_space=config
    )
    tuner.fit()
