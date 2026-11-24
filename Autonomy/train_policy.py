"""Train the obstacle-avoiding flight policy with reinforcement learning (PPO) in the training world.

    python train_policy.py                    # ~6 M steps, about 20-40 minutes on 10 CPU cores
    python train_policy.py --steps 2000000    # shorter
    python train_policy.py --evaluate         # only evaluate the saved policy

The policy learns by trial and error: +reward for progress toward the goal and for reaching it,
-reward for flying near obstacles, leaving the target height and jerky commands, and a large
penalty for a crash. Writes models/nav_policy.zip and models/training_report.json.
"""
import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "nav_policy.zip")


def make_env(seed):
    def _init():
        from train_world import ForestWorld
        return ForestWorld(seed=seed)
    return _init


def evaluate(model, episodes=60, seed=10_000, straight=False):
    """Fly fresh, unseen forests (with the safety shield, as in real flight). Returns crash rate, goals
    reached and shield interventions per episode."""
    from train_world import ForestWorld
    crashes = goals = steps = interventions = 0
    for e in range(episodes):
        env = ForestWorld(seed=seed + e)
        obs, _ = env.reset(seed=seed + e)
        done = False
        while not done:
            if straight:  # baseline: fly straight at the goal
                gb = np.arctan2(obs[32], obs[33])
                action = np.array([1.0, np.clip(gb * 2, -1, 1), np.clip(-obs[35] * 2, -1, 1)])
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            steps += 1
            done = term or trunc
        crashes += info["crashed"]
        goals += info["goals"]
        interventions += info["interventions"]
    return {"episodes": episodes, "crash_rate": crashes / episodes, "goals_per_episode": goals / episodes,
            "shield_interventions_per_1000_steps": round(1000 * interventions / max(steps, 1), 1), "steps": steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6_000_000)
    parser.add_argument("--envs", type=int, default=10)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    if not args.evaluate:
        env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(args.envs)]))
        model = PPO("MlpPolicy", env, n_steps=1024, batch_size=4096, n_epochs=8, learning_rate=3e-4,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.003,
                    policy_kwargs={"net_arch": [256, 256]}, device="cpu", verbose=0, seed=0)
        started, done = time.time(), 0
        chunk = 500_000
        while done < args.steps:
            model.learn(total_timesteps=min(chunk, args.steps - done), reset_num_timesteps=False)
            done = model.num_timesteps
            rewards = [e["r"] for e in model.ep_info_buffer] or [0]
            print(f"{done:>9,} steps | {time.time() - started:6.0f} s | mean episode reward {np.mean(rewards):7.1f}",
                  flush=True)
            model.save(MODEL)
        env.close()
    model = PPO.load(MODEL, device="cpu")
    report = {"policy": evaluate(model), "straight_to_goal_baseline": evaluate(None, straight=True),
              "trained_steps": int(model.num_timesteps)}
    json.dump(report, open(os.path.join(HERE, "models", "training_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
