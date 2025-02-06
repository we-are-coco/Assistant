# %%
import numpy as np
import re
from functools import lru_cache
import heapq
import json

def vector_search(data, search_term, vector_dim=12, weight=0.5):

    embedding_cache = {}
    non_digit_re = re.compile(r"\D")

    def embed_text(text, d):
        key = (text, d)
        if key in embedding_cache:
            return embedding_cache[key]
        vec = np.zeros(d, dtype='float32')
        for i, c in enumerate(text):
            vec[i % d] += ord(c)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embedding_cache[key] = vec
        return vec

    def composite_query_vector(final_terms):
        vecs = [embed_text(token, vector_dim) for token, token_type in final_terms]
        comp = np.mean(vecs, axis=0)
        norm = np.linalg.norm(comp)
        if norm > 0:
            comp = comp / norm
        return comp

    def doc_vector(doc):
        keys = ['title', 'description', 'type', 'category']
        texts = []
        for key in keys:
            if key in doc and isinstance(doc[key], str):
                texts.append(doc[key])
        if not texts:
            for k, v in doc.items():
                if isinstance(v, str):
                    texts.append(v)
        full_text = " ".join(texts)
        tokens = full_text.split()
        if not tokens:
            return np.zeros(vector_dim, dtype='float32')
        vecs = [embed_text(token, vector_dim) for token in tokens]
        comp = np.mean(vecs, axis=0)
        norm = np.linalg.norm(comp)
        if norm > 0:
            comp = comp / norm
        return comp

    def process_search_terms(terms):
        processed = []
        particles = ["으로부터", "에서", "에게", "께", "까지", "부터", "으로", 
                     "은", "는", "이", "가", "을", "를", "에", "도", "만", "와", "과", "고", "나"]
        for term in terms:
            t = term
            changed = True
            while changed:
                changed = False
                for particle in particles:
                    if t.endswith(particle) and len(t) > len(particle):
                        t = t[:-len(particle)]
                        changed = True
                        break
            processed.extend(t.split())
        return processed

    def clean_token(token):
        if any(marker in token for marker in ["년", "월", "일"]):
            num = non_digit_re.sub("", token)
            return num, "date"
        elif any(marker in token for marker in ["시", "분", "초"]):
            num = non_digit_re.sub("", token)
            return num, "time"
        else:
            return token, "normal"

    @lru_cache(maxsize=None)
    def levenshtein_distance(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    @lru_cache(maxsize=None)
    def normalized_edit_similarity(s1, s2):
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1
        return 1 - (levenshtein_distance(s1, s2) / max_len)

    def score_doc(doc, query, query_vec, token_type):
        best = 1.0  # 낮을수록 매칭이 좋음
        if token_type == "date":
            date_keys = ["date", "departure_date", "valid_until"]
            fields = [(k, v) for k, v in doc.items() if k in date_keys and isinstance(v, str)]
            if not fields:
                fields = [(k, v) for k, v in doc.items() if isinstance(v, str)]
            for key, value in fields:
                segments = value.split("-")
                for seg in segments:
                    if query == seg.strip():
                        return 0
                seg_scores = []
                for seg in segments:
                    seg = seg.strip()
                    seg_emb = embed_text(seg, vector_dim)
                    cos_sim = np.dot(seg_emb, query_vec)
                    cos_score = 1 - cos_sim
                    edit_sim = normalized_edit_similarity(seg.lower(), query.lower())
                    edit_score = 1 - edit_sim
                    seg_scores.append(weight * cos_score + (1 - weight) * edit_score)
                if seg_scores:
                    local_best = min(seg_scores)
                    best = min(best, local_best)
            return best
        elif token_type == "time":
            time_keys = ["time", "departure_time"]
            fields = [(k, v) for k, v in doc.items() if k in time_keys and isinstance(v, str)]
            if not fields:
                fields = [(k, v) for k, v in doc.items() if isinstance(v, str)]
            for key, value in fields:
                segments = value.split(":")
                for seg in segments:
                    if query == seg.strip():
                        return 0
                seg_scores = []
                for seg in segments:
                    seg = seg.strip()
                    seg_emb = embed_text(seg, vector_dim)
                    cos_sim = np.dot(seg_emb, query_vec)
                    cos_score = 1 - cos_sim
                    edit_sim = normalized_edit_similarity(seg.lower(), query.lower())
                    edit_score = 1 - edit_sim
                    seg_scores.append(weight * cos_score + (1 - weight) * edit_score)
                if seg_scores:
                    local_best = min(seg_scores)
                    best = min(best, local_best)
            return best
        else:
            query_lower = query.lower()
            for value in doc.values():
                if isinstance(value, str):
                    if query_lower in value.lower():
                        return 0
                    tokens = value.split()
                    if tokens:
                        token_scores = []
                        for token in tokens:
                            token_emb = embed_text(token, vector_dim)
                            cos_sim = np.dot(token_emb, query_vec)
                            cos_score = 1 - cos_sim
                            edit_sim = normalized_edit_similarity(token.lower(), query_lower)
                            edit_score = 1 - edit_sim
                            token_scores.append(weight * cos_score + (1 - weight) * edit_score)
                        local_best = min(token_scores)
                    else:
                        token_emb = embed_text(value, vector_dim)
                        cos_sim = np.dot(token_emb, query_vec)
                        local_best = 1 - cos_sim
                    best = min(best, local_best)
            return best

    processed_tokens = process_search_terms(search_term)
    final_terms = [clean_token(tok) for tok in processed_tokens]
    original_term_count = len(final_terms)

    documents = [doc for group in data.values() for doc in group]

    first_query, first_type = final_terms[0]
    first_query_vec = embed_text(first_query, vector_dim)
    stage_scores = [(score_doc(doc, first_query, first_query_vec, first_type), idx)
                    for idx, doc in enumerate(documents)]
    threshold_1 = 3 if original_term_count == 1 else 5
    candidate_indices = [idx for _, idx in heapq.nsmallest(threshold_1, stage_scores, key=lambda x: x[0])]

    if original_term_count > 1:
        second_query, second_type = final_terms[1]
        second_query_vec = embed_text(second_query, vector_dim)
        stage_scores = [(score_doc(documents[idx], second_query, second_query_vec, second_type), idx)
                        for idx in candidate_indices]
        threshold_2 = 2 if original_term_count == 2 else 3
        candidate_indices = [idx for _, idx in heapq.nsmallest(threshold_2, stage_scores, key=lambda x: x[0])]

    if original_term_count > 2:
        third_query, third_type = final_terms[2]
        third_query_vec = embed_text(third_query, vector_dim)
        stage_scores = [(score_doc(documents[idx], third_query, third_query_vec, third_type), idx)
                        for idx in candidate_indices]
        candidate_indices = [idx for _, idx in heapq.nsmallest(1, stage_scores, key=lambda x: x[0])]

    def compute_rerank_score(doc, final_terms):
        scores_for_token = []
        for token, token_type in final_terms:
            distances = []
            token_lower = token.lower()
            for key, value in doc.items():
                if isinstance(value, str):
                    sim = normalized_edit_similarity(token_lower, value.lower())
                    dist = (1 - sim) * 100
                    distances.append(dist)
            if not distances:
                token_score = 100
            else:
                distances.sort()
                num = min(4, len(distances))
                token_score = sum(distances[:num]) / num
            scores_for_token.append(token_score)
        return sum(scores_for_token) / len(scores_for_token)

    if original_term_count <= 2:
        candidate_scores = [(compute_rerank_score(documents[idx], final_terms), idx) for idx in candidate_indices]
        candidate_scores.sort(key=lambda x: x[0])
        candidate_indices = [idx for score, idx in candidate_scores]

    all_scores = [compute_rerank_score(doc, final_terms) for doc in documents]
    sorted_scores = sorted(all_scores)
    top_k = 3
    if len(sorted_scores) <= top_k:
        threshold = 100
    else:
        below_top = sorted_scores[top_k:]
        threshold = sum(below_top) / len(below_top)

    best_candidate_score = compute_rerank_score(documents[candidate_indices[0]], final_terms)
    if best_candidate_score > threshold:
        return [{"result": "검색 결과 없음"}]

    final_selection = [documents[idx] for idx in candidate_indices]
    return final_selection

data = {
        "쿠폰": [
            {
                "category": "쿠폰",
                "brand": "ABC마트",
                "type": "식품",
                "item": "커피 할인권",
                "valid_until": "2023-12-31",
                "code": "ABC123XYZ",
                "description": "커피 할인 쿠폰입니다."
            },
            {
                "category": "쿠폰",
                "brand": "XYZ슈퍼",
                "type": "의류",
                "item": "정장 할인권",
                "valid_until": "2024-01-15",
                "code": "XYZ456ABC",
                "description": "정장 할인 쿠폰입니다."
            }
        ],
        "교통": [
            {
                "category": "교통",
                "type": "버스",
                "from_location": "서울역",
                "to_location": "부산역",
                "departure_date": "2023-11-15",
                "departure_time": "08:30",
                "description": "서울에서 부산까지의 버스 티켓입니다."
            },
            {
                "category": "교통",
                "type": "지하철",
                "from_location": "강남역",
                "to_location": "잠실역",
                "departure_date": "2023-12-01",
                "departure_time": "09:15",
                "description": "강남에서 잠실까지의 지하철 티켓입니다."
            },
            {
                "category": "교통",
                "type": "기차",
                "from_location": "대전",
                "to_location": "광주",
                "departure_date": "2023-10-20",
                "departure_time": "07:00",
                "description": "대전에서 광주까지의 기차 티켓입니다."
            }
        ],
        "엔터테인먼트": [
            {
                "category": "엔터테인먼트",
                "type": "영화",
                "title": "영화 티켓 할인",
                "date": "2023-10-01",
                "time": "19:00",
                "location": "CGV 강남",
                "description": "영화 할인 티켓입니다."
            },
            {
                "category": "엔터테인먼트",
                "type": "콘서트",
                "title": "록 페스티벌",
                "date": "2023-09-25",
                "time": "18:30",
                "location": "잠실 실내체육관",
                "description": "록 콘서트 티켓입니다."
            },
            {
                "category": "엔터테인먼트",
                "type": "전시",
                "title": "미술 전시회",
                "date": "2023-11-10",
                "time": "15:00",
                "location": "서울 아트센터",
                "description": "미술 전시회 티켓입니다."
            }
        ],
        "약속": [
            {
                "category": "약속",
                "type": "미팅",
                "date": "2023-09-20",
                "time": "14:00",
                "location": "스타벅스 테헤란로",
                "details": "비즈니스 미팅",
                "description": "비즈니스 미팅 약속입니다."
            },
            {
                "category": "약속",
                "type": "의료",
                "date": "2023-10-05",
                "time": "10:00",
                "location": "서울병원",
                "details": "의사 상담 예약",
                "description": "진료 예약 약속입니다."
            },
            {
                "category": "약속",
                "type": "식당",
                "date": "2023-12-12",
                "time": "19:30",
                "location": "이태원 레스토랑",
                "details": "친구와 만남",
                "description": "식사 예약 약속입니다."
            }
        ],
        "불명": [
            {
                "category": "불명",
                "type": "정보 없음",
                "date": "2023-07-01",
                "time": "12:00",
                "description": "정보가 불명확한 경우입니다."
            },
            {
                "category": "불명",
                "type": "미정",
                "date": "2023-08-15",
                "time": "13:45",
                "description": "추가 정보가 필요합니다."
            },
            {
                "category": "불명",
                "type": "불확실",
                "date": "2023-09-10",
                "time": "11:30",
                "description": "내용이 불확실합니다."
            }
        ],
        "기타": [
            {
                "category": "기타",
                "description": "기타 정보의 예시 데이터입니다."
            },
            {
                "category": "기타",
                "description": "추가 기타 정보 예시입니다."
            },
            {
                "category": "기타",
                "description": "기타 항목의 또 다른 예시입니다."
            }
        ]
    }

test_search_terms = [
    ["강남에서", "영화"],
    ["잠실에서", "지하철"],
    ["친구랑", "미팅"],
    ["정장", "할인"],
    ["대전", "기차", "광주"],
    ["추가", "정보"],
    ["스타벅스", "테헤란로"],
    ["11월", "전시"],
    ["15일에", "부산으로", "버스표"],
    ['7시', '광주']
]

import time

for term in test_search_terms:
    start_time = time.time()
    results = vector_search(data, term)
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("검색어:", term)
    print(f"검색 소요 시간: {execution_time:.4f}초")
    print("검색 결과:")
    print(json.dumps(results[0], ensure_ascii=False, indent=2))
    print("-" * 50)
# %%
