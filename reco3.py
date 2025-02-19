# %%
import random
import datetime
import math
import logging
from typing import List, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from geopy.geocoders import Nominatim

from dummydata_1 import dummydata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommendation_grpo_feedback.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    return {
        "window_days": 30,
        "training_epochs": 3,
        "training_period": 30,
        "learning_rate": 0.01,
        "model_dir": "models",
        "batch_size": 32,
        "memory_limit": 1000
    }

CONFIG = load_config()

class Event:
    def __init__(self, data: dict):
        self.data = data
        self.category = data.get("category", "")
        try:
            self.date = datetime.datetime.strptime(data.get("date", ""), "%Y-%m-%d").date()
        except Exception as e:
            self.date = datetime.date.today()
        self.time = data.get("time", "")
        self.title = data.get("title", data.get("item", ""))
        self.description = data.get("description", "")
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(data)

class FeatureExtractor:
    def __init__(self):
        self.location_cache = {}
        self.geolocator = Nominatim(user_agent="my_agent")
        self.word_vectors = {}
        self.vector_size = 10
        self.key_vector_size = 5
        self.fixed_keys = [
            "category", "brand", "type", "title", "item", "code",
            "location", "from_location", "to_location",
            "details", "date", "time"
        ]
        self.output_dim = 57

    def _get_word_vector(self, word: str) -> List[float]:
        if word not in self.word_vectors:
            word_hash = hash(word)
            random.seed(word_hash)
            vector = [random.uniform(-1, 1) for _ in range(self.vector_size)]
            magnitude = math.sqrt(sum(x * x for x in vector))
            if magnitude > 0:
                vector = [x / magnitude for x in vector]
            self.word_vectors[word] = vector
        return self.word_vectors[word]

    def _text_to_vector(self, text: str, vector_size: Optional[int] = None) -> List[float]:
        vector_size = vector_size if vector_size is not None else self.vector_size
        if not text:
            return [0.0] * vector_size
        words = text.lower().split()
        if not words:
            return [0.0] * vector_size
        vectors = [self._get_word_vector(word) for word in words]
        avg_vector = [sum(x) / len(vectors) for x in zip(*vectors)]
        if len(avg_vector) > vector_size:
            avg_vector = avg_vector[:vector_size]
        elif len(avg_vector) < vector_size:
            avg_vector.extend([0.0] * (vector_size - len(avg_vector)))
        return avg_vector

    def _get_location_vector(self, location: str) -> List[float]:
        if not location or location == "N/A":
            return [0.0, 0.0]
        try:
            if location not in self.location_cache:
                loc = self.geolocator.geocode(location)
                if loc:
                    self.location_cache[location] = [loc.latitude / 90.0, loc.longitude / 180.0]
                else:
                    self.location_cache[location] = [0.0, 0.0]
            return self.location_cache[location]
        except Exception as e:
            return [0.0, 0.0]

    def extract_features(self, event: Event) -> List[float]:
        features = []
        for key in self.fixed_keys:
            val = event.data.get(key, "")
            if key == "date":
                try:
                    d = datetime.datetime.strptime(val, "%Y-%m-%d")
                    features.extend([d.year / 3000.0, d.month / 12.0, d.day / 31.0])
                except Exception as e:
                    features.extend([0.0, 0.0, 0.0])
            elif key == "time":
                if val:
                    try:
                        t = datetime.datetime.strptime(val, "%H:%M")
                        features.extend([t.hour / 24.0, t.minute / 60.0])
                    except Exception as e:
                        features.extend([0.0, 0.0])
                else:
                    features.extend([0.0, 0.0])
            elif key in ["location", "from_location", "to_location"]:
                features.extend(self._get_location_vector(val))
            else:
                vec = self._text_to_vector(val, vector_size=self.key_vector_size)
                features.extend(vec)
        if len(features) < self.output_dim:
            features.extend([0.0] * (self.output_dim - len(features)))
        return features

class GRPOModel(nn.Module):
    def __init__(self, total_input_dim=57, hidden_dim=32):
        super(GRPOModel, self).__init__()
        self.coupon_input_dim = 28  
        self.fixed_input_dim = total_input_dim - self.coupon_input_dim  # 29
        
        self.coupon_encoder = nn.Sequential(
            nn.Linear(self.coupon_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        self.fixed_encoder = nn.Sequential(
            nn.Linear(self.fixed_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        self.combined_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        self.out_date = nn.Linear(hidden_dim, 1)
        self.out_time = nn.Linear(hidden_dim, 1)
        self.out_schedule = nn.Linear(hidden_dim, 1)
        self.out_weekday = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x_coupon = x[:, :self.coupon_input_dim]
        x_fixed = x[:, self.coupon_input_dim:]

        coupon_encoded = self.coupon_encoder(x_coupon)
        fixed_encoded = self.fixed_encoder(x_fixed)
        
        combined = torch.cat([coupon_encoded, fixed_encoded], dim=-1)
        combined_feat = self.combined_layer(combined)
        
        out_date = self.out_date(combined_feat)
        out_time = self.out_time(combined_feat)
        out_schedule = self.out_schedule(combined_feat)
        out_weekday = self.out_weekday(combined_feat)
        
        outputs = torch.cat([out_date, out_time, out_schedule, out_weekday], dim=-1)
        return outputs

class GRPORecommendationAgent:
    def __init__(self, window_days=30):
        self.window_days = window_days
        self.model = GRPOModel(total_input_dim=57, hidden_dim=32)
        self.optimizer = optim.Adam(self.model.parameters(), lr=CONFIG["learning_rate"])
        self.feature_extractor = FeatureExtractor()
        self.used_coupons_global = set()
        self.memory = []

    def reset_used_coupons(self):
        self.used_coupons_global = set()

    def compute_recommended_time(self, coupon: Event, fixed_events: List[Event]) -> str:
        start_range = 9 * 60    # 09:00 -> 540분
        end_range = 21 * 60     # 21:00 -> 1260분
        candidate_times = list(range(start_range, end_range + 1, 15))
        
        forbidden_intervals = []
        for event in fixed_events:
            if event.time:
                try:
                    t = datetime.datetime.strptime(event.time, "%H:%M")
                    fixed_minutes = t.hour * 60 + t.minute
                    forbidden_start = max(start_range, fixed_minutes - 120)
                    forbidden_end = min(end_range, fixed_minutes + 120)
                    forbidden_intervals.append((forbidden_start, forbidden_end))
                except Exception:
                    continue
        
        allowed_candidates = []
        for cand in candidate_times:
            conflict = False
            for (fstart, fend) in forbidden_intervals:
                if fstart <= cand <= fend:
                    conflict = True
                    break
            if not conflict:
                allowed_candidates.append(cand)
        
        if allowed_candidates:
            chosen = random.choice(allowed_candidates)
        else:
            chosen = random.choice(candidate_times)
        new_hour = chosen // 60
        new_minute = chosen % 60
        return f"{new_hour:02d}:{new_minute:02d}"

    def recommend(self, valid_coupons: List[Event], fixed_events: List[Event],
                  base_date: datetime.date, end_date: datetime.date) -> List[Dict]:
        recommendations = []
        for coupon in valid_coupons:
            if coupon.title in self.used_coupons_global:
                continue
            days_remaining = (coupon.date - base_date).days
            if days_remaining <= 0:
                continue
            features = self.feature_extractor.extract_features(coupon)
            feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                predicted_values = self.model(feat_tensor).squeeze(0).tolist()
            fixed_for_coupon = [fe for fe in fixed_events if fe.date == coupon.date and fe.time]
            recommended_time = self.compute_recommended_time(coupon, fixed_for_coupon)
            recommendations.append({
                "type": "추천 일정",
                "coupon": coupon.title,
                "time": recommended_time,
                "date": coupon.date,
                "brand": coupon.data.get("brand", ""),
                "coupon_type": coupon.data.get("type", ""),
                "predicted_value": sum(predicted_values) / len(predicted_values),
                "predicted_criteria": predicted_values,
                "days_remaining": days_remaining,
                "features": features
            })
            self.used_coupons_global.add(coupon.title)
        recommendations = sorted(recommendations, key=lambda x: x["predicted_value"], reverse=True)
        return recommendations

    def train(self, features: List[float], target: List[float]):
        self.memory.append((features, target))
        if len(self.memory) > CONFIG["memory_limit"]:
            self.memory.pop(0)
        if len(self.memory) < CONFIG["batch_size"]:
            return
        mini_batch = random.sample(self.memory, CONFIG["batch_size"])
        batch_features = torch.tensor([x for x, _ in mini_batch], dtype=torch.float32)
        batch_targets = torch.tensor([t for _, t in mini_batch], dtype=torch.float32)
        
        self.optimizer.zero_grad()
        outputs = self.model(batch_features)
        exploration_std = 0.01
        noise = torch.randn_like(outputs) * exploration_std
        outputs_noisy = outputs + noise
        weight_tensor = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
        loss = torch.mean(((outputs_noisy - batch_targets) ** 2) * weight_tensor)
        loss.backward()
        self.optimizer.step()
        
        baseline = batch_targets.mean(dim=0)
        advantage = batch_targets - baseline
        rmse = torch.sqrt(torch.mean((outputs - batch_targets) ** 2, dim=0))
        mae = torch.mean(torch.abs(outputs - batch_targets), dim=0)
        overall_rmse = torch.sqrt(torch.mean((outputs - batch_targets) ** 2))
        overall_mae = torch.mean(torch.abs(outputs - batch_targets))
        logger.info(f"Mini-batch update: Loss={loss.item():.4f}, Overall RMSE={overall_rmse.item():.4f}, Overall MAE={overall_mae.item():.4f}")
        logger.info(f"Criterion-wise RMSE: {rmse.tolist()}, MAE: {mae.tolist()}")
        logger.info(f"Baseline: {baseline.tolist()}, Mean Advantage: {advantage.mean(dim=0).tolist()}")

    def save(self, model_path: str):
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path: str):
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict, strict=False)
        logger.info("모델을 불러왔으나, 일부 파라미터는 초기화될 수 있습니다.")

def get_korean_weekday(date_obj: datetime.date) -> str:
    weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    return weekdays[date_obj.weekday()]

def print_schedule(schedule: List[Dict], fixed_events: List[Event],
                   base_date: datetime.date, iteration: Optional[int] = None,
                   show_evaluation_criteria: bool = False):
    grouped = {}
    for ev in fixed_events:
        d = ev.date
        if d not in grouped:
            grouped[d] = {"fixed": [], "recommend": []}
        grouped[d]["fixed"].append(ev)
    for rec in schedule:
        d = rec["date"]
        if d not in grouped:
            grouped[d] = {"fixed": [], "recommend": []}
        grouped[d]["recommend"].append(rec)
    sorted_dates = sorted(grouped.keys())
    
    header = f"[=== 추천 일정 (반복 {iteration}) ===" if iteration is not None else "[=== 추천 일정 ==="
    print("\n" + header + "\n")
    
    for d in sorted_dates:
        day_str = f"{d.strftime('%Y-%m-%d')} ({get_korean_weekday(d)})"
        print(f"{day_str}:")
        day_group = grouped[d]
        if day_group["fixed"]:
            print("  [고정 일정]")
            for idx, event in enumerate(day_group["fixed"], start=1):
                title = event.title if event.title else "N/A"
                print(f"    {idx}. [{event.category}] {title}")
                if event.time:
                    print(f"       시간: {event.time}")
                from_loc = event.data.get("from_location")
                to_loc = event.data.get("to_location")
                loc = event.data.get("location")
                if from_loc:
                    print(f"       출발지: {from_loc}")
                if to_loc:
                    print(f"       도착지: {to_loc}")
                if loc and (not from_loc and not to_loc):
                    print(f"       장소: {loc}")
                event_type = event.data.get("type")
                if event_type:
                    print(f"       유형: {event_type}")
                if event.description:
                    print(f"       설명: {event.description}")

        if day_group["recommend"]:
            print("  [추천 일정]")
            for idx, rec in enumerate(day_group["recommend"], start=1):
                coupon_title = rec.get("coupon", "N/A")
                print(f"    {idx}. [쿠폰] {coupon_title}")
                rec_time = rec.get("time", "")
                if rec_time:
                    print(f"       추천 시간: {rec_time}")
                rec_date = rec.get("date")
                if rec_date and isinstance(rec_date, datetime.date):
                    rec_date_str = f"{rec_date.strftime('%Y-%m-%d')} ({get_korean_weekday(rec_date)})"
                    print(f"       원래 만료일: {rec_date_str}")
                brand = rec.get("brand")
                coupon_type = rec.get("coupon_type")
                if brand:
                    print(f"       브랜드: {brand}")
                if coupon_type:
                    print(f"       유형: {coupon_type}")
        print("")
    print("============================================\n")

def main():
    logger.info("GRPO 기반 쿠폰 추천 시스템 시작")
    try:
        dataset_num = int(input("데이터셋 번호 (예: 1): "))
        mode = int(input("모드 선택 (1: 훈련, 2: 추천): "))
        days = int(input("추천 기간 (일): "))
        if mode == 1:
            num_iterations = int(input("훈련 반복 횟수: "))
            if num_iterations <= 0:
                raise ValueError("반복 횟수는 양수여야 합니다.")
        else:
            num_iterations = 1
    except ValueError as e:
        logger.warning(f"잘못된 입력 ({e}). 기본값 사용")
        dataset_num = 1
        mode = 2
        days = 7
        num_iterations = 5

    logger.info("데이터 로드 중...")
    data, _ = dummydata(dataset_num)
    
    base_date = datetime.date(2025, 2, 15)
    end_date = base_date + datetime.timedelta(days=days)

    fixed_events = []
    for key in ["교통", "약속"]:
        for event_data in data.get(key, []):
            try:
                event = Event.from_dict(event_data)
                if base_date <= event.date <= end_date:
                    fixed_events.append(event)
                    logger.info(f"고정 이벤트 추가: {event.title}")
            except Exception as e:
                logger.error(f"고정 이벤트 변환 실패: {e}")

    valid_coupons = []
    for coupon_data in data.get("쿠폰", []):
        try:
            coupon = Event.from_dict(coupon_data)
            if base_date <= coupon.date <= end_date:
                valid_coupons.append(coupon)
                logger.info(f"유효한 쿠폰 추가: {coupon.title}, 만료일: {coupon.date}")
        except Exception as e:
            logger.error(f"쿠폰 변환 실패: {e}")

    if not valid_coupons:
        logger.warning("추천할 쿠폰이 없습니다.")
        return

    logger.info(f"총 {len(valid_coupons)}개의 유효한 쿠폰과 {len(fixed_events)}개의 고정 이벤트를 찾았습니다.")
    agent = GRPORecommendationAgent(window_days=CONFIG["window_days"])
    agent.reset_used_coupons()
    model_dir = Path(CONFIG["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "grpo_recommendation_model.pth"
    if model_path.exists():
        try:
            agent.load(str(model_path))
            logger.info("기존 모델을 로드했습니다.")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")

    evaluation_criteria = {
        "date_appropriateness": "날짜 적절성",
        "time_appropriateness": "시간 적절성",
        "schedule_relation": "고정 일정과의 관계 적절성",
        "weekday_appropriateness": "요일 적절성"
    }
    criteria_order = ["date_appropriateness", "time_appropriateness", "schedule_relation", "weekday_appropriateness"]

    for iteration in range(num_iterations):
        try:
            if mode == 1:
                logger.info(f"훈련 반복 {iteration+1}/{num_iterations}")
            schedule = agent.recommend(valid_coupons, fixed_events, base_date, end_date)
            print_schedule(schedule, fixed_events, base_date,
                           iteration+1 if mode == 1 else None,
                           show_evaluation_criteria=(mode == 1))
            if mode == 1:
                print("\n=== 평가 기준별 점수 입력 (1-5) ===")
                ratings_vector = []
                for criterion in criteria_order:
                    description = evaluation_criteria[criterion]
                    while True:
                        try:
                            rating = float(input(f"{description}: "))
                            if 1 <= rating <= 5:
                                ratings_vector.append(rating)
                                break
                            else:
                                print("평가 점수는 1에서 5 사이여야 합니다.")
                        except ValueError:
                            print("올바른 숫자를 입력해주세요.")
                for decision in schedule:
                    agent.train(decision["features"], ratings_vector)
                avg_rating = sum(ratings_vector) / len(ratings_vector)
                logger.info(f"반복 {iteration+1} 완료: 평균 평가 점수 {avg_rating:.2f}")
        except Exception as e:
            logger.error(f"{'훈련' if mode == 1 else '추천'} 중 오류 발생: {e}")
            continue
    if mode == 1:
        try:
            agent.save(str(model_path))
            logger.info("모델이 저장되었습니다.")
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")

if __name__ == "__main__":
    main()
# %%