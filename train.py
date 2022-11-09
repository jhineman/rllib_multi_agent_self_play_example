import os
import sys

from griddly import gd
from griddly.util.rllib.callbacks import VideoCallbacks
from griddly.util.rllib.environment.core import RLlibEnv, RLlibMultiAgentWrapper
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune import register_env, tune

from rllib_multi_agent_self_play_example.gap_agent import GAPAgent
from rllib_multi_agent_self_play_example.simple_conv_agent import SimpleConvAgent

# You have to put this here so that rllib can find griddly libraries when it starts new workers
sep = os.pathsep
os.environ["PYTHONPATH"] = sep.join(sys.path)

environment_name = "TestEnvironment"
environment_yaml = "Multi-Agent/foragers.yaml"
model_name = "SimpleConvAgent"

# Register the environment with RLlib
register_env(environment_name, lambda config: RLlibMultiAgentWrapper(RLlibEnv(config)))

model_class = None
if model_name == "SimpleConvAgent":
    model_class = SimpleConvAgent
elif model_name == "GlobalAveragePoolingAgent":
    model_class = GAPAgent

# Register the model with RLlib
ModelCatalog.register_custom_model(model_name, model_class)

test_dir = f"./results/{environment_name}"
video_dir = f"videos"

config = (
    PPOConfig()
    .rollouts(num_rollout_workers=8, rollout_fragment_length=512)
    .callbacks(VideoCallbacks)
    .training(
        model={
            "custom_model": model_name
        },
        train_batch_size=2048,
        lr=2e-5,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
        grad_clip=None,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        sgd_minibatch_size=512,
        num_sgd_iter=4,
    )
    .environment(
        env_config={
            # A video every 50 iterations
            'record_video_config': {
                'frequency': 10000,
                'directory': video_dir,

                # Will record a video of the global observations
                'include_global': True,

                # Will record a video of the agent's perspective
                'include_agents': True,
            },
            'random_level_on_reset': True,
            'yaml_file': environment_yaml,
            'global_observer_type': gd.ObserverType.SPRITE_2D,
            'player_observer_type': gd.ObserverType.VECTOR,
            'max_steps': 1000,
        },
        env=environment_name, clip_actions=True)
    .debugging(log_level="ERROR")
    .framework(framework="torch")
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
)

result = tune.run(
    "PPO",
    name="PPO",
    stop={"timesteps_total": 1000000},
    local_dir=test_dir,
    config=config.to_dict(),
)
