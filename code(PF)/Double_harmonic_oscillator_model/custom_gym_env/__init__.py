import gym as gym

gym.register(
    id="DoubleHarmonicOscillatorEnv-v0",
    entry_point="custom_gym_env.envs.double_quantum_environment:DoubleHarmonicOscillatorEnv",
)
