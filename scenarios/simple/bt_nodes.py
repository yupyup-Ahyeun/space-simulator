import math
import random
import pygame
from modules.base_bt_nodes import BTNodeList, Status, Node, Sequence, Fallback, SyncAction, LocalSensingNode, DecisionMakingNode

# BT Node List
BTNodeList.ACTION_NODES.append('TaskExecutingNode')
BTNodeList.ACTION_NODES.append('ExplorationNode')
BTNodeList.ACTION_NODES.append('ReturnToBaseNode')
BTNodeList.ACTION_NODES.append('GroupMakingNode')
BTNodeList.ACTION_NODES.append('GroupReleasingNode')
# BTNodeList.ACTION_NODES.append('GroupCheckingNode')


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
    

class GroupMakingNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._make_group)
        # 에이전트의 초기 상태
        self.group_leader_priority = random.randint(1, 5)  # 1~5 범위의 랜덤 값
        self.leader = True  # 초기 상태는 모두 리더
        self.group_members = []  # 그룹에 속한 멤버 (리더일 경우만 사용)

    def _make_group(self, agent, blackboard):
        # 인접 에이전트를 감지
        nearby_agents = agent.get_agents_nearby()
        
        for other_agent in nearby_agents:
            # 그룹 리더 우선순위를 비교
            if self.group_leader_priority > other_agent.group_leader_priority:
                other_agent.leader = False  # 다른 에이전트를 slave로 전환
                other_agent.group_leader_priority = 0  # slave의 리더 우선순위 제거
                if self.leader:  # 현재 에이전트가 리더라면 멤버 추가
                    if other_agent not in self.group_members:   # 다른 leader의 slave가 아닐 때만 slave로 전환
                        self.group_members.append(other_agent)
            elif self.group_leader_priority < other_agent.group_leader_priority:
                # 현재 에이전트가 slave로 전환
                self.leader = False
                self.group_leader_priority = 0
                # 멤버 정보는 리더에게 종속되므로 초기화
                self.group_members = []
                return Status.SUCCESS

        # 리더는 slave 에이전트를 통제
        if self.leader:
            for member in self.group_members:
                # 리더의 움직임과 작업을 멤버들에게 전달
                member.follow(agent.position)
                member.assigned_task_id = agent.assigned_task_id

        return Status.RUNNING



# class GroupCheckingNode(SyncAction):
#     def __init__(self, name, agent):
#         super().__init__(name, self._group_checking)
    
#     def _group_checking(self):
#         return Status.SUCCESS

class GroupReleaseNode(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._release_group)

    def _release_group(self, agent, blackboard):
        # 그룹 해제를 위해 우선 slave로 설정된 모든 에이전트를 초기화
        if not agent.leader:
            # 에이전트가 slave 상태라면 초기화
            agent.group_leader_priority = random.randint(1, 5)
            agent.leader = True  # 리더로 초기화
            return Status.SUCCESS

        # 리더인 경우 그룹 멤버 초기화
        if agent.leader:
            for member in agent.group_members:
                member.group_leader_priority = random.randint(1, 5)
                member.leader = True  # 멤버를 리더로 초기화

            # 그룹 해제 후 멤버 목록 초기화
            agent.group_members = []

        # 에이전트의 그룹 관련 속성 초기화
        agent.group_leader_priority = random.randint(1, 5)
        agent.leader = True

        return Status.SUCCESS
