# %%
import sys
import random
import datetime
import math
import numpy as np
import time
import json
import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from dummydata_1 import dummydata
from geopy.geocoders import Nominatim
import pandas as pd
from pathlib import Path
import yaml
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommendation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    return {
        "weights_file": "coupon_rl_weights.json",
        "time_weights_file": "coupon_rl_time_weights.json",
        "model_dir": "models",
        "learning_rate": 0.01,
        "mcts_iterations": 200,
        "ucb_constant": 1.4,
        "num_actions_per_node": 3,
        "time_preferences": {
            "weekday": {
                "preferred_hours": [17, 18, 19, 20, 21],
                "bonus": 0.3
            },
            "weekend": {
                "preferred_hours": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "bonus": 0.4
            }
        },
        "exploration_rate": 0.3,
        "random_noise_std": 0.1
    }

CONFIG = load_config()

@dataclass
class Event:
    title: str
    date: datetime.date
    time: str
    category: str
    description: str = ""
    location: str = ""
    brand: str = ""
    type: str = ""
    from_location: str = "" 
    to_location: str = ""   
    details: str = "" 
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Event':
        if isinstance(data.get("date"), datetime.date):
            date = data["date"]
        else:
            date = datetime.datetime.strptime(data["date"], "%Y-%m-%d").date()
        title = data.get("title", "")
        if not title and "item" in data:
            title = data.get("item", "")
        return cls(
            title=title,
            date=date,
            time=data.get("time", ""),
            category=data.get("category", ""),
            description=data.get("description", ""),
            location=data.get("location", ""),
            brand=data.get("brand", ""),
            type=data.get("type", ""),
            from_location=data.get("from_location", ""),
            to_location=data.get("to_location", ""),
            details=data.get("details", "")
        )

class DeepRLModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class FeatureExtractor:
    def __init__(self):
        self.location_cache = {}
        self.geolocator = Nominatim(user_agent="my_agent")
        self.word_vectors = {}
        self.vector_size = 10
        
    def _get_word_vector(self, word: str) -> List[float]:
        if word not in self.word_vectors:
            word_hash = hash(word)
            random.seed(word_hash)
            vector = [random.uniform(-1, 1) for _ in range(self.vector_size)]
            magnitude = math.sqrt(sum(x*x for x in vector))
            if magnitude > 0:
                vector = [x/magnitude for x in vector]
            self.word_vectors[word] = vector
        return self.word_vectors[word]
    
    def _text_to_vector(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.vector_size
            
        words = text.lower().split()
        if not words:
            return [0.0] * self.vector_size
            
        vectors = [self._get_word_vector(word) for word in words]
        avg_vector = [sum(x)/len(vectors) for x in zip(*vectors)]
        return avg_vector
        
    def extract_features(self, date: datetime.date, time: int, 
                        event: Event, fixed_events: List[Event]) -> torch.Tensor:
        features = []
        
        features.extend([
            date.weekday() / 6.0,
            time / 23.0,
            1 if date.weekday() >= 5 else 0 
        ])
        
        min_interval = float('inf')
        for fixed_event in fixed_events:
            interval = abs((date - fixed_event.date).days)
            min_interval = min(min_interval, interval)
        features.append(1.0 / (1.0 + min_interval))
        
        if event.location and event.location != "N/A":
            try:
                if event.location not in self.location_cache:
                    location = self.geolocator.geocode(event.location)
                    if location:
                        self.location_cache[event.location] = (location.latitude, location.longitude)
                coords = self.location_cache.get(event.location)
                if coords:
                    features.extend([coords[0] / 90.0, coords[1] / 180.0])
                else:
                    features.extend([0.0, 0.0])
            except Exception as e:
                logger.warning(f"위치 정보 추출 실패: {e}")
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
            
        brand = getattr(event, 'brand', '')
        brand_vector = self._text_to_vector(brand)
        features.extend(brand_vector)
        
        type_str = getattr(event, 'type', '')
        type_vector = self._text_to_vector(type_str)
        features.extend(type_vector)
        
        title_vector = self._text_to_vector(event.title)
        features.extend(title_vector)
            
        return torch.tensor(features, dtype=torch.float32)

class MCTSNode:
    def __init__(self, state: Dict, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = []
        self.ucb_score = 0.0
        
    def add_child(self, child_node: 'MCTSNode'):
        self.children.append(child_node)
        child_node.parent = self
        
    def update(self, reward: float):
        self.visits += 1
        self.total_reward += reward
        self.ucb_score = self.calculate_ucb()
        
    def calculate_ucb(self, c_param: float = CONFIG["ucb_constant"]) -> float:
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            return self.total_reward / self.visits
        if self.parent.visits == 0 or self.visits == 0:
            return float('inf')
        return (self.total_reward / self.visits) + c_param * math.sqrt(
            math.log(max(1, self.parent.visits)) / max(1, self.visits)
        )

class ImprovedPMCTS:
    def __init__(self, agent, feature_extractor: FeatureExtractor):
        self.agent = agent
        self.feature_extractor = feature_extractor
        self.stats = defaultdict(list) 

    def search(self, root_state: Dict, valid_coupons: List[Event],
               fixed_events: List[Event], iterations: int,
               start_date: datetime.date, end_date: datetime.date) -> MCTSNode:
        root = MCTSNode(root_state)
        best_sim_state = None
        best_reward = -float('inf')
        
        for i in range(iterations):
            node = self.tree_policy(root, valid_coupons, fixed_events, start_date, end_date)
            sim_state, reward = self.simulate(node, valid_coupons, fixed_events, start_date, end_date)
            self.backpropagate(node, reward)
            
            self.stats["rewards"].append(reward)
            self.stats["tree_depth"].append(self.get_tree_depth(node))
            
            if reward > best_reward:
                best_reward = reward
                best_sim_state = sim_state
            if i % 50 == 0:
                self.prune_tree(root)
        
        if best_sim_state is not None:
            return MCTSNode(best_sim_state)
        else:
            return self.best_child(root)
    
    def tree_policy(self, node: MCTSNode, valid_coupons: List[Event],
                    fixed_events: List[Event],
                    start_date: datetime.date, end_date: datetime.date) -> MCTSNode:
        while not self.is_terminal(node, valid_coupons, start_date, end_date):
            if not node.children:
                return self.expand(node, valid_coupons, fixed_events, start_date, end_date)
            node = self.best_uct(node)
        return node
    
    def is_terminal(self, node: MCTSNode, valid_coupons: List[Event],
                    start_date: datetime.date, end_date: datetime.date) -> bool:
        if node.state.get("finished", False):
            return True
        available = [(i, coupon) for i, coupon in enumerate(valid_coupons)
                     if i not in node.state["used_coupons"] and start_date <= coupon.date <= end_date]
        return len(available) == 0
    
    def expand(self, node: MCTSNode, valid_coupons: List[Event],
               fixed_events: List[Event],
               start_date: datetime.date, end_date: datetime.date) -> MCTSNode:
        available = [(i, coupon) for i, coupon in enumerate(valid_coupons)
                     if i not in node.state["used_coupons"] and start_date <= coupon.date <= end_date]
        new_nodes = []
        for coupon_idx, coupon in available:
            days_until_expiry = (coupon.date - start_date).days
            possible_dates = [start_date + datetime.timedelta(days=i)
                              for i in range(1, days_until_expiry + 1)
                              if start_date + datetime.timedelta(days=i) <= end_date]
            if not possible_dates:
                continue
            num_dates = min(CONFIG["num_actions_per_node"], len(possible_dates))
            selected_dates = random.sample(possible_dates, num_dates)
            
            for rec_date in selected_dates:
                new_state = copy.deepcopy(node.state)
                decision_exists = any(
                    d["coupon_index"] == coupon_idx and d["recommended_date"] == rec_date
                    for d in new_state["schedule"]
                )
                if decision_exists:
                    continue
                
                if rec_date.weekday() >= 5:  # 주말
                    possible_times = CONFIG["time_preferences"]["weekend"]["preferred_hours"]
                else:  # 평일
                    possible_times = CONFIG["time_preferences"]["weekday"]["preferred_hours"]
                rec_time = random.choice(possible_times)
                features = self.feature_extractor.extract_features(
                    rec_date, rec_time, coupon, fixed_events
                )
                
                with torch.no_grad():
                    action_value = self.agent.model(features)
                    action_value = action_value.item() + random.gauss(0, CONFIG["random_noise_std"])
                
                decision = {
                    "title": coupon.title,
                    "recommended_date": rec_date,
                    "recommended_time": f"{rec_time:02d}:00",
                    "value": action_value,
                    "features": features,
                    "coupon_index": coupon_idx
                }
                new_state["schedule"].append(decision)
                new_state["used_coupons"].add(coupon_idx)
                new_state["finished"] = False
                new_node = MCTSNode(new_state, parent=node, action=decision)
                new_nodes.append(new_node)
        
        finish_state = copy.deepcopy(node.state)
        finish_state["finished"] = True
        finish_node = MCTSNode(finish_state, parent=node, action={"action": "finish"})
        new_nodes.append(finish_node)
        
        if not new_nodes:
            finish_state = copy.deepcopy(node.state)
            finish_state["finished"] = True
            new_node = MCTSNode(finish_state, parent=node)
            return new_node
        
        node.children.extend(new_nodes)
        return random.choice(new_nodes)
    
    def simulate(self, node: MCTSNode, valid_coupons: List[Event],
                 fixed_events: List[Event],
                 start_date: datetime.date, end_date: datetime.date) -> Tuple[Dict, float]:
        state = copy.deepcopy(node.state)
        available = [
            (i, coupon) for i, coupon in enumerate(valid_coupons)
            if i not in state["used_coupons"] and start_date <= coupon.date <= end_date
        ]
        n = len(available)
        decisions_added = 0

        def conflict_exists(schedule, decision):
            return any(d["recommended_date"] == decision["recommended_date"] and
                       d["recommended_time"] == decision["recommended_time"]
                       for d in schedule)

        if n == 0:
            reward = self.evaluate_schedule(state["schedule"], valid_coupons, fixed_events)
            return state, reward
        elif n == 1:
            coupon_idx, coupon = available[0]
            attempt = 0
            decision = None
            while attempt < 5:
                decision = self._simulate_add_coupon(coupon_idx, coupon, fixed_events, start_date, end_date)
                if decision is None:
                    break
                if conflict_exists(state["schedule"], decision):
                    attempt += 1
                else:
                    break
            if decision is not None and not conflict_exists(state["schedule"], decision):
                state["schedule"].append(decision)
                state["used_coupons"].add(coupon_idx)
                decisions_added += 1
        else:
            k_target = random.choice(range(1, n + 1))
            remaining = available.copy()
            for i in range(k_target):
                chosen_pair = random.choice(remaining)
                coupon_idx, coupon = chosen_pair
                remaining.remove(chosen_pair)
                attempt = 0
                decision = None
                while attempt < 5:
                    decision = self._simulate_add_coupon(coupon_idx, coupon, fixed_events, start_date, end_date)
                    if decision is None:
                        break
                    if conflict_exists(state["schedule"], decision):
                        attempt += 1
                    else:
                        break
                if decision is not None and not conflict_exists(state["schedule"], decision):
                    state["schedule"].append(decision)
                    state["used_coupons"].add(coupon_idx)
                    decisions_added += 1

        if not state["schedule"] and available:
            for coupon_idx, coupon in available:
                attempt = 0
                decision = None
                while attempt < 5:
                    decision = self._simulate_add_coupon(coupon_idx, coupon, fixed_events, start_date, end_date)
                    if decision is None:
                        break
                    if conflict_exists(state["schedule"], decision):
                        attempt += 1
                    else:
                        break
                if decision is not None and not conflict_exists(state["schedule"], decision):
                    state["schedule"].append(decision)
                    state["used_coupons"].add(coupon_idx)
                    decisions_added += 1
                    break

        reward = self.evaluate_schedule(state["schedule"], valid_coupons, fixed_events)
        return state, reward
    
    def _simulate_add_coupon(self, coupon_idx: int, coupon: Event,
                             fixed_events: List[Event],
                             start_date: datetime.date, end_date: datetime.date) -> Optional[Dict]:
        days_until_expiry = (coupon.date - start_date).days
        possible_dates = [start_date + datetime.timedelta(days=i)
                          for i in range(1, days_until_expiry + 1)
                          if start_date + datetime.timedelta(days=i) <= end_date]
        if not possible_dates:
            return None
        rec_date = random.choice(possible_dates)
        if rec_date.weekday() >= 5:
            possible_times = CONFIG["time_preferences"]["weekend"]["preferred_hours"]
        else:
            possible_times = CONFIG["time_preferences"]["weekday"]["preferred_hours"]
        rec_time = random.choice(possible_times)
        features = self.feature_extractor.extract_features(rec_date, rec_time, coupon, fixed_events)
        with torch.no_grad():
            action_value = self.agent.model(features)
        decision = {
            "title": coupon.title,
            "recommended_date": rec_date,
            "recommended_time": f"{rec_time:02d}:00",
            "value": action_value.item() + random.gauss(0, CONFIG["random_noise_std"]),
            "features": features,
            "coupon_index": coupon_idx
        }
        return decision
    
    def backpropagate(self, node: MCTSNode, reward: float):
        while node is not None:
            node.update(reward)
            node = node.parent
    
    def best_child(self, node: MCTSNode) -> MCTSNode:
        return max(node.children, key=lambda n: n.visits)
    
    def best_uct(self, node: MCTSNode) -> MCTSNode:
        return max(node.children, key=lambda n: n.ucb_score)
    
    def get_tree_depth(self, node: MCTSNode) -> int:
        depth = 0
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth
    
    def prune_tree(self, root: MCTSNode, max_depth: int = 10):
        """메모리 관리를 위한 트리 가지치기"""
        def _prune(node: MCTSNode, current_depth: int):
            if current_depth > max_depth:
                node.children = []
                return
            for child in node.children:
                _prune(child, current_depth + 1)
        _prune(root, 0)
    
    def evaluate_schedule(self, schedule: List[Dict], valid_coupons: List[Event],
                        fixed_events: List[Event]) -> float:
        if not schedule:
            return 0.0
            
        rewards = []
        
        for i in range(len(schedule) - 1):
            current = schedule[i]
            next_event = schedule[i + 1]
            interval = abs((next_event["recommended_date"] - current["recommended_date"]).days)
            interval_reward = 1.0 / (1.0 + interval)
            rewards.append(interval_reward)
        
        for decision in schedule:
            conflicts = sum(1 for f in fixed_events if f.date == decision["recommended_date"])
            conflict_penalty = 1.0 / (1.0 + conflicts)
            rewards.append(conflict_penalty)
        
        for decision in schedule:
            rec_date = decision["recommended_date"]
            rec_time = int(decision["recommended_time"].split(":")[0])
            
            if rec_date.weekday() >= 5:
                preferred = rec_time in CONFIG["time_preferences"]["weekend"]["preferred_hours"]
                bonus = CONFIG["time_preferences"]["weekend"]["bonus"] if preferred else 0
            else:
                preferred = rec_time in CONFIG["time_preferences"]["weekday"]["preferred_hours"]
                bonus = CONFIG["time_preferences"]["weekday"]["bonus"] if preferred else 0
            
            rewards.append(decision["value"] + bonus)
        
        return np.mean(rewards)

class ImprovedRecommendationAgent:
    def __init__(self, input_size: int):
        self.model = DeepRLModel(input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=CONFIG["learning_rate"])
        self.feature_extractor = FeatureExtractor()
        self.mcts = ImprovedPMCTS(self, self.feature_extractor)
        self.memory = [] 
    
    def recommend(self, valid_coupons: List[Event], fixed_events: List[Event],
                  start_date: datetime.date, end_date: datetime.date) -> List[Dict]:
        root_state = {"used_coupons": set(), "finished": False, "schedule": []}
        best_node = self.mcts.search(
            root_state, valid_coupons, fixed_events,
            CONFIG["mcts_iterations"], start_date, end_date
        )
        schedule = best_node.state.get("schedule", [])
        if not schedule:
            available = [(i, coupon) for i, coupon in enumerate(valid_coupons)
                         if start_date <= coupon.date <= end_date]
            if available:
                coupon_idx, coupon = random.choice(available)
                decision = self.mcts._simulate_add_coupon(coupon_idx, coupon, fixed_events, start_date, end_date)
                if decision is not None:
                    schedule.append(decision)
        return schedule
    
    def train(self, experience: Tuple[torch.Tensor, float]):
        self.memory.append(experience)
        if len(self.memory) >= 32:
            self._update_model()
            
    def _update_model(self):
        batch = random.sample(self.memory, min(32, len(self.memory)))
        features, rewards = zip(*batch)
        
        features = torch.stack(features)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        self.optimizer.zero_grad()
        values = self.model(features).squeeze()
        loss = nn.MSELoss()(values, rewards)
        loss.backward()
        self.optimizer.step()
        
        if len(self.memory) > 1000:
            self.memory = self.memory[-1000:]
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(state_dict, path, pickle_protocol=4)
    
    def load(self, path: str):
        if Path(path).exists():
            try:
                state_dict = torch.load(path, map_location='cpu', weights_only=False) 
                self.model.load_state_dict(state_dict['model_state_dict'])
                self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
                logger.info("기존 모델을 로드했습니다.")
            except Exception as e:
                logger.error(f"모델 로드 실패: {e}")
                logger.info("새로운 모델로 시작합니다.")

def main():
    logger.info("추천 시스템 시작")
    
    try:
        dataset_num = int(input("데이터셋 번호 (1-3): "))
        mode = int(input("모드 선택 (1: 훈련, 2: 추천): "))
        days = int(input("추천 기간 (일): "))
    except ValueError:
        logger.warning("잘못된 입력. 기본값 사용")
        dataset_num = 1
        mode = 2
        days = 7
    
    base_date = datetime.date.today()
    end_date = base_date + datetime.timedelta(days=days)
    
    logger.info(f"데이터셋 {dataset_num} 로드 중...")
    data = dummydata(dataset_num)[0]
    fixed_events = []
    for category in ["교통", "엔터테인먼트", "약속"]:
        for event_data in data.get(category, []):
            try:
                event = Event.from_dict(event_data)
                fixed_events.append(event)
                logger.debug(f"고정 이벤트 추가: {event.title}")
            except Exception as e:
                logger.error(f"이벤트 변환 실패: {e}")
    
    valid_coupons = []
    logger.info("쿠폰 데이터 처리 중...")
    for coupon_data in data.get("쿠폰", []):
        try:
            coupon = Event.from_dict(coupon_data)
            if base_date <= coupon.date <= end_date:
                valid_coupons.append(coupon)
                logger.info(f"유효한 쿠폰 추가: {coupon.title}, 만료일: {coupon.date}")
        except Exception as e:
            logger.error(f"쿠폰 변환 실패: {e}")
    
    fixed_events = [event for event in fixed_events if base_date <= event.date <= end_date]
    if not valid_coupons:
        logger.warning("추천할 쿠폰이 없습니다.")
        return
    
    logger.info(f"총 {len(valid_coupons)}개의 유효한 쿠폰과 {len(fixed_events)}개의 고정 이벤트를 찾았습니다.")
    
    feature_extractor = FeatureExtractor()
    sample_features = feature_extractor.extract_features(
        base_date, 0, valid_coupons[0], fixed_events
    )
    agent = ImprovedRecommendationAgent(input_size=len(sample_features))
    
    model_dir = Path(CONFIG["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "recommendation_model.pth"
    if model_path.exists():
        try:
            agent.load(str(model_path))
            logger.info("기존 모델을 로드했습니다.")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            logger.info("새로운 모델로 시작합니다.")
    
    if mode == 1:  # 훈련 모드
        try:
            iterations = int(input("훈련 반복 횟수: "))
            if iterations <= 0:
                raise ValueError("반복 횟수는 양수여야 합니다.")
        except ValueError as e:
            iterations = 5
            logger.warning(f"잘못된 입력 ({e}). 기본값 5회 사용")
        
        for i in range(iterations):
            logger.info(f"훈련 반복 {i+1}/{iterations}")
            
            try:
                schedule = agent.recommend(valid_coupons, fixed_events, base_date, end_date)
                print_schedule(schedule, valid_coupons, fixed_events, i+1, show_evaluation_criteria=True)
                
                try:
                    rating = float(input("평가 점수 (1-5): "))
                    if not 1 <= rating <= 5:
                        raise ValueError("평가 점수는 1에서 5 사이여야 합니다.")
                except ValueError as e:
                    rating = 3
                    logger.warning(f"잘못된 평가 ({e}). 기본값 3점 사용")
                
                # 경험 수집 및 학습
                for decision in schedule:
                    agent.train((decision["features"], rating))
                
                logger.info(f"반복 {i+1} 완료: 평가 점수 {rating}")
                
            except Exception as e:
                logger.error(f"훈련 중 오류 발생: {e}")
                continue
        
        # 모델 저장
        try:
            agent.save(str(model_path))
            logger.info("모델이 저장되었습니다.")
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
        
    else:  # 추천 모드
        try:
            schedule = agent.recommend(valid_coupons, fixed_events, base_date, end_date)
            print_schedule(schedule, valid_coupons, fixed_events, show_evaluation_criteria=False)
            logger.info("추천이 완료되었습니다.")
        except Exception as e:
            logger.error(f"추천 중 오류 발생: {e}")

def print_schedule(schedule: List[Dict], valid_coupons: List[Event], fixed_events: List[Event], 
                   iteration: Optional[int] = None, show_evaluation_criteria: bool = True):
    WEEKDAY_MAP = {
        0: "월요일",
        1: "화요일",
        2: "수요일",
        3: "목요일",
        4: "금요일",
        5: "토요일",
        6: "일요일"
    }
    
    if iteration is not None:
        print(f"\n=== 추천 일정 (반복 {iteration}) ===")
    else:
        print("\n=== 최종 추천 일정 ===")
    
    all_events = defaultdict(list)

    for item in schedule:
        rec_date = item["recommended_date"]
        if isinstance(rec_date, str):
            rec_date = datetime.datetime.strptime(rec_date, "%Y-%m-%d").date()
        all_events[rec_date].append(("추천", item))

    for event in fixed_events:
        all_events[event.date].append(("고정", event))
    
    for date in sorted(all_events.keys()):
        weekday = WEEKDAY_MAP.get(date.weekday(), "")
        print(f"\n{date} ({weekday}):")
        
        fixed = [e for t, e in all_events[date] if t == "고정"]
        if fixed:
            print("  [고정 일정]")
            for idx, event in enumerate(fixed, 1):
                print(f"    {idx}. [{event.category}] {event.title if event.title else 'N/A'}")
                print(f"       시간: {event.time}")
                if event.category == "교통":
                    if event.from_location:
                        print(f"       출발지: {event.from_location}")
                    if event.to_location:
                        print(f"       도착지: {event.to_location}")
                if event.category == "약속":
                    if event.details:
                        print(f"       세부사항: {event.details}")
                if event.location:
                    print(f"       장소: {event.location}")
                if event.brand:
                    print(f"       브랜드: {event.brand}")
                if event.type:
                    print(f"       유형: {event.type}")
                if event.description:
                    print(f"       설명: {event.description}")
        
        recommended = [e for t, e in all_events[date] if t == "추천"]
        if recommended:
            print("  [추천 일정]")
            for idx, item in enumerate(recommended, 1):
                coupon = next((c for i, c in enumerate(valid_coupons) if i == item["coupon_index"]), None)
                print(f"    {idx}. [쿠폰] {item['title'] if item['title'] else 'N/A'}")
                print(f"       추천 시간: {item.get('recommended_time', 'N/A')}")
                if coupon:
                    print(f"       원래 만료일: {coupon.date} ({WEEKDAY_MAP.get(coupon.date.weekday(), '')})")
                    if coupon.brand:
                        print(f"       브랜드: {coupon.brand}")
                    if coupon.type:
                        print(f"       유형: {coupon.type}")
                    if coupon.description:
                        print(f"       설명: {coupon.description}")
                print(f"       예상 가치: {item.get('value', 'N/A')}")
    
    if show_evaluation_criteria:
        print("\n평가 기준:")
        print("1: 매우 나쁨 (날짜/시간이 매우 부적절)")
        print("2: 나쁨 (개선이 필요함)")
        print("3: 보통 (괜찮지만 개선의 여지가 있음)")
        print("4: 좋음 (대체로 만족스러움)")
        print("5: 매우 좋음 (최적의 추천)")
        print()

if __name__ == "__main__":
    main()

# %% 