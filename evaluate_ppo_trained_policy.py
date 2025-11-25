import jax.numpy as jnp
import numpy as np
import jax
from jax import vmap

# --- CAMBIAMENTO QUI ---
# Invece di stable_baselines3.PPO, usiamo sbx.PPO
# Nota: VecMonitor e make_vec_env si prendono ancora da SB3, sono compatibili
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

# I tuoi import
from gym_env.wrapper_gym_lasernav import LaserNavGym
from socialjym.utils.rewards.lasernav_rewards.reward1_lasernav import Reward1LaserNav
from socialjym.utils.aux_functions import animate_trajectory

def main():
    # 1. CONFIGURAZIONE
    env_params = {
        'n_stack': 5, 
        'lidar_num_rays': 100, 
        'lidar_angular_range': 2*jnp.pi,
        'lidar_max_dist': 10., 
        'n_humans': 7, 
        'n_obstacles': 5,
        'robot_radius': 0.3, 
        'robot_dt': 0.25, 
        'humans_dt': 0.01,
        'robot_visible': False, 
        'scenario': 'hybrid_scenario',
        # Assicurati di usare la tua nuova Reward1LaserNav
        'reward_function': Reward1LaserNav(robot_radius=0.3), 
        'kinematics': 'unicycle',
    }

    eval_env = LaserNavGym(env_params=env_params)
    obs, info = eval_env.reset()

    all_states = np.array([eval_env._state])
    all_observations = np.array([obs])

    model = PPO.load("ppo_lasernav_parallel")

    print("\nAvvio test di visualizzazione su ambiente singolo...\n")

    for i in range(60): # Massimo 60 step di test
        # Predici l'azione usando il modello addestrato (deterministic=True rimuove la casualità)
        action, _ = model.predict(obs, deterministic=True)
        
        # Esegui lo step
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        # Salva stato e osservazione
        all_states = np.vstack((all_states, [eval_env._state]))
        all_observations = np.vstack((all_observations, [obs]))
        
        if terminated or truncated:
            print(f"Episodio terminato al passo {i+1}")
            break

    robot_yaws = all_states[:, -1, 4]

    def get_angles(yaw):
        return jnp.linspace(
            yaw - env_params['lidar_angular_range']/2, 
            yaw + env_params['lidar_angular_range']/2, 
            env_params['lidar_num_rays']
        )
    angles = vmap(get_angles)(robot_yaws)

    # Estrae le distanze lidar dalle osservazioni (dal 6° indice in poi)
    lidar_dists = all_observations[:, 0, 6:] 
    
    # Combina distanze e angoli per il plotter: shape (T, 100, 2)
    lidar_measurements = vmap(lambda d, a: jnp.stack((d, a), axis=-1))(lidar_dists, angles)

    # ==========================================
    # 5. LANCIO ANIMAZIONE
    # ==========================================
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