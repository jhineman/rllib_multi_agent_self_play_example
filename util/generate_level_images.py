
from griddly import GymWrapperFactory, gd, GymWrapper
from griddly.RenderTools import RenderToFile



if __name__ == "__main__":
    wrapper = GymWrapperFactory()

    image_renderer = RenderToFile()

    for renderer in ["GlobalSpriteObserver", "GlobalBlockObserver"]:
        for level in [0,1,2]:
            env = GymWrapper(
                yaml_file="../gridman/gridman_multiagent.yaml",
                player_observer_type="Vector",
                global_observer_type=renderer,
            )

            env.reset(level_id=level)
            obs = env.render(observer="global", mode="rgb_array")
            image_renderer.render(obs, f"{renderer}_{level}.png")

            env.close()



