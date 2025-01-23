import math
import random
import pygame
from modules.base_bt_nodes import BTNodeList, Status, Node, Sequence, Fallback, SyncAction, LocalSensingNode, DecisionMakingNode

# BT Node List
BTNodeList.ACTION_NODES.append('TaskExecutingNode')
BTNodeList.ACTION_NODES.append('ExplorationNode')
BTNodeList.ACTION_NODES.append('ReturnToBaseNode')

BTNodeList.ACTION_NODES.append('SeparateSlaveNode')
BTNodeList.ACTION_NODES.append('GroupMakingNode')
BTNodeList.ACTION_NODES.append('GroupCheckingNode')
BTNodeList.ACTION_NODES.append('GroupReleasingNode')
BTNodeList.ACTION_NODES.append('FollowLeaderNode')



# Scenario-specific Action/Condition Nodes
from modules.utils import config
target_arrive_threshold = config['tasks']['threshold_done_by_arrival']
task_locations = config['tasks']['locations']
sampling_freq = config['simulation']['sampling_freq']
sampling_time = 1.0 / sampling_freq  # in seconds
agent_max_random_movement_duration = config.get('agents', {}).get('random_exploration_duration', None)


# Task executing node
class TaskExecutingNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._execute_task)

    def _execute_task(self, agent, blackboard):        
        assigned_task_id = blackboard.get('assigned_task_id')        
        if assigned_task_id is not None:
            agent_position = agent.position
            next_waypoint = agent.tasks_info[assigned_task_id].position
            # Calculate norm2 distance
            distance = math.sqrt((next_waypoint[0] - agent_position[0])**2 + (next_waypoint[1] - agent_position[1])**2)
            
            assigned_task_id = blackboard.get('assigned_task_id')
            if distance < agent.tasks_info[assigned_task_id].radius + target_arrive_threshold: # Agent reached the task position                                
                if agent.tasks_info[assigned_task_id].completed:  # 이렇게 먼저 해줘야 중복해서 task_amount_done이 올라가지 않는다.                  
                    return Status.SUCCESS
                agent.tasks_info[assigned_task_id].reduce_amount(agent.work_rate)
                agent.update_task_amount_done(agent.work_rate)  # Update the amount of task done                

            # Move towards the task position
            agent.follow(next_waypoint)

        return Status.RUNNING


# Exploration node
class ExplorationNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._random_explore)
        self.random_move_time = float('inf')
        self.random_waypoint = (0, 0)

    def _random_explore(self, agent, blackboard):
        # Move towards a random position
        if self.random_move_time > agent_max_random_movement_duration:
            self.random_waypoint = self.get_random_position(task_locations['x_min'], task_locations['x_max'], task_locations['y_min'], task_locations['y_max'])
            self.random_move_time = 0 # Initialisation
        
        blackboard['random_waypoint'] = self.random_waypoint        
        self.random_move_time += sampling_time   
        agent.follow(self.random_waypoint)         
        return Status.RUNNING
        
    def get_random_position(self, x_min, x_max, y_min, y_max):
        pos = (random.randint(x_min, x_max),
                random.randint(y_min, y_max))
        return pos
    

class ReturnToBaseNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._return_to_base)
        self.return_to_base_mode = False
        self.depot_pos = pygame.Vector2(700,500)

    def _return_to_base(self, agent, blackboard):
        # Check if the assigned task is completed
        if agent.assigned_task_id is not None and agent.tasks_info[agent.assigned_task_id].completed:
            self.return_to_base_mode = True

        # Move to the base if the task is completed
        if self.return_to_base_mode:
            distance_to_base = (self.depot_pos - agent.position).length()
            if distance_to_base > target_arrive_threshold:
                agent.follow(self.depot_pos)
                return Status.SUCCESS

            self.return_to_base_mode = False

        # If the task is not completed, return ``FAILURE`` to allow the rest of the BT to continue
        return Status.FAILURE


class SeparateSlaveNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._check_agent)

    def _check_agent(self, agent, blackboard):
        # circular import 막기 위해 여기에 import
        from scenarios.simple.agent import Agent, LeaderAgent  # 나중에 경로 수정 필요... simple base로 만들어졌기 때문에 지금은 simple 폴더임.
        
        # Check if the agent is a LeaderAgent or just an Agent
        if isinstance(agent, LeaderAgent):
            return Status.SUCCESS
        elif isinstance(agent, Agent):
            return Status.FAILURE

# class SeperateSlaveNode(SyncAction):
#     def __init__(self, name, agent):
#         super().__init__(name, self._check_agent)

#     def _check_agent(self, agent, blackboard):
#         return Status.SUCCESS
   
class GroupMakingNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._make_group_test)
    def _make_group_test(self, agent, blackboard):
        return Status.SUCCESS

class GroupCheckingNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._check_group)

    def _check_group(self, agent, blackboard):
        return Status.SUCCESS #FAILURE

class GroupReleasingNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._release_group)

    def _release_group(self, agent, blackboard):
        return Status.SUCCESS

class FollowLeaderNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._follow_leader)

    def _follow_leader(self, agent, blackboard):
        return Status.SUCCESS


