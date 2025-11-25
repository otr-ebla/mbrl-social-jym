import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

# I tuoi import
from gym_env.wrapper_gym_lasernav import LaserNavGym
from socialjym.utils.rewards.lasernav_rewards.reward1_lasernav import Reward1LaserNav
from socialjym.utils.aux_functions import animate_trajectory


def main():

    # ============================
    # 1. CONFIGURAZIONE AMBIENTE
    # ============================
    env_params = {
        'n_stack': 5,
        'lidar_num_rays': 100,
        'lidar_angular_range': 2*np.pi,
        'lidar_max_dist': 10.,
        'n_humans': 7,
        'n_obstacles': 5,
        'robot_radius': 0.3,
        'robot_dt': 0.25,
        'humans_dt': 0.01,
        'robot_visible': False,
        'scenario': 'hybrid_scenario',
        'reward_function': Reward1LaserNav(robot_radius=0.3),
        'kinematics': 'unicycle',
    }

    # ============================
    # 2. CREAZIONE VEC-ENV
    # ============================
    N_ENVS = 16
    TOTAL_TIMESTEPS = 10_000_000

    print(f"Creazione di {N_ENVS} ambienti paralleli sb3...")
    vec_env = make_vec_env(
        LaserNavGym,
        n_envs=N_ENVS,
        env_kwargs={'env_params': env_params},
    )
    vec_env = VecMonitor(vec_env)

    # ============================
    # 3. TRAINING PPO (SB3)
    # ============================
    print("Inizio training SB3 PPO...")

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        device="cuda",   # usa GPU per il training se disponibile
        batch_size=64,
        n_steps=2048,
        n_epochs=10,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.0,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    print("Salvataggio modello SB3...")
    model.save("ppo_lasernav_parallel")
    vec_env.close()

    # ============================
    # 4. TEST SU AMBIENTE SINGOLO
    # ============================
    print("\nAvvio test di visualizzazione su ambiente singolo...")

    eval_env = LaserNavGym(env_params)
    obs, info = eval_env.reset()

    all_states = np.array([eval_env._state])
    all_observations = np.array([obs])

    for i in range(60):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)

        all_states = np.vstack((all_states, [eval_env._state]))
        all_observations = np.vstack((all_observations, [obs]))

        if terminated or truncated:
            print(f"Episodio terminato al passo {i+1}")
            break

    # ============================
    # 5. PREPARA LIDAR
    # ============================
    print("Preparazione dati per animazione...")

    robot_yaws = all_states[:, -1, 4]

    angles = np.linspace(
        -env_params['lidar_angular_range']/2,
        env_params['lidar_angular_range']/2,
        env_params['lidar_num_rays']
    )[None, :] + robot_yaws[:, None]

    lidar_dists = all_observations[:, 0, 6:]

    lidar_measurements = np.stack([lidar_dists, angles], axis=-1)

    # ============================
    # 6. ANIMAZIONE
    # ============================
    print("Apertura finestra grafica...")
    animate_trajectory(
        states=all_states,
        humans_radiuses=info['humans_parameters'][:, 0],
        robot_radius=env_params['robot_radius'],
        humans_policy='hsfm',
        robot_goal=info['robot_goal'],
        scenario=info['current_scenario'],
        static_obstacles=info['static_obstacles'][-1],
        robot_dt=env_params['robot_dt'],
        lidar_measurements=lidar_measurements,
        kinematics=env_params['kinematics'],
        figsize=(10, 10)
    )


if __name__ == "__main__":
    main()
