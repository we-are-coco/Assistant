# %%
import datetime
from pathlib import Path

from reco3 import Event, GRPORecommendationAgent, load_config
from dummydata_1 import dummydata


def infer(dataset=1, days=30):
    """
    강화학습으로 훈련된 모델을 활용하여 추천 일정과 고정 일정을 추론하는 함수입니다.

    인자:
        dataset (int): 사용할 데이터셋 번호 (기본값 1)
        days (int): 추천 기간 (일 단위, 기본값 7)

    반환:
        리스트: 추천 일정과 고정 일정 정보가 포함된 리스트
                추천 일정의 경우 다음 형식입니다:
                {
                    "is_reco": True,
                    "id": (원래 데이터의 id),
                    "reco_date": "yyyy-mm-dd",
                    "reco_time": "hh:mm",
                    "item": (원래 데이터의 item)
                }
                고정 일정의 경우 다음 형식입니다:
                {
                    "is_reco": False,
                    "id": (원래 데이터의 id),
                    "fixed_date": "yyyy-mm-dd",
                    "fixed_time": "hh:mm",
                    "description": (원래 데이터의 description)
                }
    """
    data, _ = dummydata(dataset)
    base_date = datetime.date.today()
    end_date = base_date + datetime.timedelta(days=days)

    fixed_events = []
    for key, events in data.items():
        if key in ["쿠폰", "불명", "기타"]:
            continue
        for event_data in events:
            try:
                event = Event.from_dict(event_data)
                if base_date <= event.date <= end_date:
                    fixed_events.append(event)
            except Exception as e:
                continue

    valid_coupons = []
    for coupon_data in data.get("쿠폰", []):
        try:
            coupon = Event.from_dict(coupon_data)
            if base_date <= coupon.date <= end_date:
                valid_coupons.append(coupon)
        except Exception as e:
            continue

    if not valid_coupons:
        return []

    config = load_config()
    agent = GRPORecommendationAgent(window_days=config["window_days"])
    model_dir = Path(config["model_dir"])
    model_path = model_dir / "grpo_recommendation_model.pth"
    if model_path.exists():
        try:
            agent.load(str(model_path))
        except Exception as e:
            pass

    schedule = agent.recommend(valid_coupons, fixed_events, base_date, end_date)

    result = []
    for rec in schedule:
        coupon_title = rec.get("coupon", "")
        id_val = None
        for coupon in valid_coupons:
            if coupon.title == coupon_title:
                id_val = coupon.data.get("id", coupon.data.get("id", None))
                break
        result.append({
            "is_reco": True,
            "id": id_val,
            "reco_date": rec["date"].strftime("%Y-%m-%d") if isinstance(rec["date"], datetime.date) else rec["date"],
            "reco_time": rec["time"],
            "item": coupon_title
        })

    for event in fixed_events:
        id_val = event.data.get("id", event.data.get("code", None))
        result.append({
            "is_reco": False,
            "id": id_val,
            "fixed_date": event.date.strftime("%Y-%m-%d") if isinstance(event.date, datetime.date) else event.date,
            "fixed_time": event.time,
            "description": event.data.get("description", "")
        })

    return result


if __name__ == "__main__":
    recommendations = infer()
    print(recommendations)
    print(len(recommendations))
    for item in recommendations:
        print(item) 
# %%
