import gym
import gym_platform


def main():
    env = gym.make("Platform-v0")
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action=action)
        print(f"{reward=}", f"{done=}")
        env.render("rgb")


if __name__ == "__main__":
    main()
