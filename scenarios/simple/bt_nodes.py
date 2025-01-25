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
        from scenarios.simple.agent import LeaderAgent, SlaveAgent  # 나중에 경로 수정 필요... simple base로 만들어졌기 때문에 지금은 simple 폴더임.
        
        # leader인지 slave인지에 따라 성공/실패 적용하여 BT 분리
        if isinstance(agent, LeaderAgent):
            return Status.SUCCESS
        elif isinstance(agent, SlaveAgent):
            return Status.FAILURE


class GroupMakingNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._make_group)
    
    def _make_group(self, agent, blackboard):
        # circular import 막기 위해 여기에 import
        from scenarios.simple.agent import SlaveAgent  # 나중에 경로 수정 필요... simple base로 만들어졌기 때문에 지금은 simple 폴더임.

        # 주변 에이전트 가져오기
        nearby_agents = agent.get_agents_nearby()
        
        for nearby_agent in nearby_agents:
            # nearby_agent 중 slave인 agent에게만 다음 로직을 적용
            if not isinstance(nearby_agent, SlaveAgent):
                continue
            
            # 이미 다른 leader를 가진 slave에게만 다음 로직을 적용
            if nearby_agent.leader_id is not None:
                continue
            
            # leader의 id를 slave의 leader_id에 저장
            nearby_agent.leader_id = agent.agent_id
            
            # leader의 slave_list에 slave의 id 추가
            if nearby_agent.agent_id not in agent.slave_list:
                agent.slave_list.append(nearby_agent.agent_id)

        return Status.SUCCESS


class GroupCheckingNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._check_group)

    def _check_group(self, agent, blackboard):

        #TODO: task에 맞게 slave 개수를 지정하는 로직으로 업데이트 필요
        #TODO: slave들이 leader와 일정 거리 내에 있을 때 task 수행 시작할 수 있도록 업데이트 필요

        # leader의 slave_list 확인  
        if len(agent.slave_list) > 0:
            return Status.SUCCESS  # slave가 하나 이상 있으면 성공
        else:
            return Status.FAILURE  # slave가 없으면 실패


class FollowLeaderNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._follow_leader)

    def _follow_leader(self, agent, blackboard):
        # leader의 GroupMakingNode에서 slave에게 저장했던 leader id 호출
        leader_id = agent.leader_id
        
        if not leader_id:
            return Status.FAILURE  # leader가 없으면 실패
        
        # nearby_agents 중에서 leader id와 일치하는 id를 가진 agent를 찾아 leader_agent로 지정
        nearby_agents = agent.get_agents_nearby()
        
        leader_agent = next((a for a in nearby_agents if a.agent_id == leader_id), None)

        if not leader_agent:
            return Status.FAILURE   # 일치하는 leader_agent가 없으면 실패
        
        # leader의 task 가져와서 assign하기
        assigned_task_id = leader_agent.assigned_task_id

        if assigned_task_id is None:
            return Status.FAILURE  # task가 없으면 실패
        
        agent.assigned_task_id = assigned_task_id
        
        # assign한 task로 이동
        task_position = agent.tasks_info[assigned_task_id].position
        agent.follow(task_position)
        
        return Status.RUNNING


class GroupReleasingNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._release_group)

    def _release_group(self, agent, blackboard):
        # circular import 막기 위해 여기에 import
        from scenarios.simple.agent import SlaveAgent  # 나중에 경로 수정 필요... simple base로 만들어졌기 때문에 지금은 simple 폴더임.

        # slave_list에 있는 모든 slave들의 leader_id 초기화
        for slave_id in agent.slave_list:
            slave_agent = next((a for a in agent.agents_info if a.agent_id == slave_id), None)
            if slave_agent and isinstance(slave_agent, SlaveAgent):
                slave_agent.leader_id = None

        # leader의 slave_list 초기화
        agent.slave_list.clear()

        return Status.SUCCESS

        




