import numpy as np
import csv
import os
import pdb
import pandas as pd

"""
    Introdction:
    This file contains the code to evaluate the performance of the training result
    - 
    
"""


class EpisodeEvaluator:
    """
        An EpisodeEvaluator class is defined to record the data of each episode
    """
    def __init__(self,env,id,output_dir, time_limit, env_seed):  
        self._env = env
        self.id = id
        self.output_dir = output_dir
        self.time = 0
        self.velocity_list = [0]
        self.position_list = [[0,0]]
        self.reward_list = [0]
        self.collision_list = [False]
        self.distance_list = [0]
        self.time_to_collision = -1
        self.distance_to_collision = -1
        self.time_limit = time_limit
        self.env_seed = env_seed

    def record(self,timestep):
        """
            Record the data of each timestep
        """
        # self.velocity_list.append(np.linalg.norm(self.env.task._car.observables.body_vel_2d(self.env.physics)))
        self.velocity_list.append(np.linalg.norm(timestep.observation['car/body_vel_2d']))
        self.position_list.append(timestep.observation['car/body_pose_2d'][:2])
        self.reward_list.append(timestep.reward)
        self.distance_list.append(np.linalg.norm(np.copy(self.position_list[-1]) - np.copy(self.position_list[-2])))
        self.time += 1
        
        if self._env.task.collided:
            self.collision_list.append(True)
            # reset the collision flag
            self._env.task.collided = False
            # record time & distance until first collision
            if self.time_to_collision == -1:
                self.time_to_collision = self.time
                self.distance_to_collision = np.sum(self.distance_list)
        else:
            self.collision_list.append(False)

    def get_summary_data(self):
        """
            Get the summary data of an episode
        """
        summary_data = {}
        summary_data["env_seed"] = self.env_seed
        summary_data["episode_id"] = self.id
        summary_data["total_timesteps"] = self.time
        summary_data["stuck"] = self.time < self.time_limit
        summary_data["mean_velocity"] = np.mean(self.velocity_list)
        summary_data["total_distance"] = np.sum(self.distance_list)
        summary_data["collision_times"] = np.sum(self.collision_list)
        summary_data["time_to_first_collision"] = self.time_to_collision
        summary_data["distance_to_first_collision"] = self.distance_to_collision
        return summary_data

    # def export_to_csv(self):
    #     """
    #         Export the recorded data of an episode to a csv file
    #     """
    #     # prepare the data
    #     velocity_list = np.copy(self.velocity_list)
    #     x_pos_list = np.copy(self.position_list)[:,0]
    #     y_pos_list = np.copy(self.position_list)[:,1]
    #     distance_list = np.copy(self.distance_list)
    #     reward_list = np.copy(self.reward_list)
    #     collision_list = np.copy(self.collision_list)
    #     time_list = np.copy(self.time_list)

    #     # Combine the data
    #     result = np.vstack([time_list,x_pos_list,y_pos_list,velocity_list,reward_list,collision_list])

    #     # Add headers
    #     headers = ["Time", "Velocity","Pos_X","Pos_Y", "Reward", "Collision"]


    #     # Ensure the output directory exists
    #     os.makedirs(os.path.join(self.output_dir, "output"), exist_ok=True)

    #     # Define the output file path
    #     output_file = os.path.join(self.output_dir, "output", f"{self.id}.csv")

    #     # Write the data
    #     with open(output_file, mode="w", newline="") as file:
    #         writer = csv.writer(file)
    #         writer.writerow(headers)  # Write the header row
    #         writer.writerows(result.T)  # Write each row of the transposed result

    #     print(f"CSV exported to {output_file}")





class Evaluator:
    """
        An Evaluator class is defined to manage the whole evaluation process
    """
    def __init__(self, env, output_dir, episodes, prefix, time_limit = 1000):
        self.output_dir = output_dir
        self.episodes = episodes
        self.current_episode = 0
        self.time_limit = time_limit
        self._data = []
        self.prefix = prefix
        
        self.folder = os.path.join("evaluation_results")
        # Create the folder
        # try:
        #     os.mkdir(self.folder)
        #     print(f"Folder '{self.folder}' created successfully!")
        # except FileExistsError:
        #     self.folder = os.path.join(self.folder, "new_run")
        #     os.makedirs(self.folder) 
        #     print(f"***WARNNING***: Folder already exists.")
        try:
            os.mkdir(self.folder)
            print(f"Folder '{self.folder}' created successfully!")
        except FileExistsError:
            pass  
        self.set_environment(env,env_seed=0)
            
    def step(self,timestep):
        self.episode_evaluator.record(timestep)
        if self._env.task.timesteps == self.time_limit or self.check_static():    
            print(f"Episode {self.current_episode} finished at time {self._env.task.timesteps}")
            self._env.task.initialize_episode(self._env.physics,np.zeros(3))
            self._data.append(self.episode_evaluator.get_summary_data())
            self.reinitialize()

    def check_finish(self):
        """
            check if the episode has reached its limit
        """
        if self.current_episode == self.episodes:
            print("Evaluation for environment finished")   
            return True
        return False

    def reinitialize(self):
        """
            Reinitialize a new EpisodeEvaluator after the current episode is finished
        """
        self.current_episode += 1
        seed = self.episode_evaluator.env_seed
        self.episode_evaluator = EpisodeEvaluator(env = self._env, id = self.current_episode, 
                                                  output_dir = self.folder, time_limit=self.time_limit, env_seed=seed)


    def check_static(self):
        """
            Check if the car is static for the last 10 timesteps,
            if so, reinitialize the episode
        """
        recent_vel = self.episode_evaluator.distance_list[-10:]

        if len(self.episode_evaluator.distance_list) >= 5 and np.all(np.copy(recent_vel) < 1e-3):

            print("Car is stuck, reinitializing the episode")
            # self.env.task.initialize_episode(self.env.physics,np.zeros(3))
            # # self.episode_evaluator.export_to_csv()
            # self._data.append(self.episode_evaluator.get_summary_data())
            # self.reinitialize()
            return True
        return False
    
    def export_csv(self):
        df = pd.DataFrame(self._data)
        df.to_csv(os.path.join(self.folder, f"{self.prefix}_summary.csv"), index=False)
        print(f"Summary data exported to {os.path.join(self.folder, f'{self.prefix}_summary.csv')}")

    def set_environment(self,env,env_seed):
        self._env = env
        self.current_episode = 0
        self.episode_evaluator = EpisodeEvaluator(env = self._env, id = self.current_episode, 
                                                  output_dir = self.folder, time_limit=self.time_limit, env_seed=env_seed)
   
  