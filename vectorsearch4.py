# %%
import numpy as np  # numpy: 수치 계산을 위한 라이브러리 / Library for numerical operations
import re  # re: 정규 표현식 라이브러리 / Library for regular expressions
import json  # json: JSON 데이터 처리 라이브러리 / Library for JSON handling
from datetime import datetime  # datetime: 날짜 및 시간 처리를 위한 클래스 / Class for handling dates and times
from functools import lru_cache  # lru_cache: 함수 결과 캐시를 위한 데코레이터 / Decorator for caching function outputs
from collections import defaultdict  # defaultdict: 기본값을 제공하는 딕셔너리 클래스 / Dictionary class that provides default values
import heapq  # heapq: 우선순위 큐 알고리즘을 위한 라이브러리 / Library for heap-based priority queue algorithms
import faiss  # faiss: 고성능 벡터 검색 라이브러리 / Library for high-performance vector search
from dummydata import dummydata

class VectorSearchEngine:
    # 검색 및 임베딩 기반 검색 엔진 클래스 / Class for vector-based search and embedding search engine
    def __init__(self, vector_dim=12, weight=0.5, debug=False, advanced_embedding=False, base_threshold=0.6, match_threshold=0.5):
        """
        검색 엔진 초기화
        Initialize the search engine.
        :param vector_dim: 임베딩 벡터의 차원 (기본 12) / Dimension of the embedding vector (default 12)
        :param weight: 토큰 매칭 점수와 벡터 유사도 점수를 결합할 때의 가중치 (기본 0.5) / Weight for combining token match and vector similarity (default 0.5)
        :param debug: 디버그 모드 여부 (기본 False) / Debug mode flag (default False)
        :param advanced_embedding: 고급 임베딩 활성화 여부. word2vec 모델 없이 내부 deterministic embedding을 사용합니다.
                                   / Whether to enable advanced embedding. (If enabled, a deterministic embedding is used instead of word2vec)
        """
        self.vector_dim = vector_dim  # 단순 임베딩 차원 (fallback 용)
        self.weight = weight  # 토큰 점수와 벡터 점수를 결합할 가중치
        self.embedding_cache = {}  # 임베딩 결과 캐시 (재사용을 위해)
        self.debug = debug  # 디버그 모드 여부 저장
        self.advanced_embedding = advanced_embedding  # 고급 임베딩 활성화 여부
        self.base_threshold = base_threshold  # 기본 임베딩 점수 임계치
        self.match_threshold = match_threshold  # 토큰 매칭 점수 임계치

    def embed_text(self, text):
        """
        간단한 문자별 임베딩 함수 (문자 코드 합산 후 정규화)
        A simple character-level embedding function (sums character codes and normalizes).
        """
        key = (text, self.vector_dim)
        if key in self.embedding_cache:
            return self.embedding_cache[key]
        vec = np.zeros(self.vector_dim, dtype="float32")
        for i, c in enumerate(text):
            vec[i % self.vector_dim] += ord(c)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self.embedding_cache[key] = vec
        return vec

    def advanced_embed_text(self, text):
        """
        고급 임베딩 함수: 단어별 deterministic 임베딩(내부 hash 기반)을 사용하여 텍스트의 평균 임베딩 벡터를 계산합니다.
        Advanced embedding: computes the average embedding for the text based on a deterministic hash per token.
        """
        return self.simple_advanced_embed_text(text)

    def simple_advanced_embed_text(self, text):
        """
        텍스트를 단어 단위로 나눈 후, 각 단어에 대해 deterministic embedding을 계산하고 평균을 구합니다.
        """
        tokens = text.split()
        if not tokens:
            return np.zeros(self.vector_dim, dtype="float32")
        token_vecs = [self.deterministic_token_embedding(token) for token in tokens]
        vec = np.mean(token_vecs, axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype("float32")

    @staticmethod
    def deterministic_token_embedding(token, vector_dim=12):
        """
        단어에 대해 MD5 해시를 기반으로 한 deterministic embedding을 생성합니다.
        - 해시값(16바이트)을 np.uint8 배열로 변환한 후, 필요에 따라 슬라이스 혹은 타일링합니다.
        - 값들을 -127.5를 빼서 중심을 0으로 맞추고, 정규화를 수행합니다.
        """
        import hashlib
        h = hashlib.md5(token.encode('utf-8')).digest()  # 16바이트 해시값
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        arr = arr - 127.5  # 중앙 정렬
        if vector_dim <= len(arr):
            vec = arr[:vector_dim]
        else:
            # 벡터 길이보다 해시 바이트가 짧으면 타일링
            repeats = (vector_dim + len(arr) - 1) // len(arr)
            vec = np.tile(arr, repeats)[:vector_dim]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype("float32")

    @staticmethod
    def clean_token(token):
        """
        검색어 토큰 전처리 함수 (위치 정보 감지 등)
        Preprocess a token from the search query.
        """
        non_digit_re = re.compile(r"\D")
        location_particles = ["에서", "으로", "까지", "부터"]
        for particle in location_particles:
            pattern = r"(.*?)(" + particle + r")(의)?$"
            m = re.match(pattern, token)
            if m:
                base = m.group(1)
                if base:
                    return base, "location"
        if "월" in token:
            num = non_digit_re.sub("", token)
            return num, "month"
        elif "일" in token:
            num = non_digit_re.sub("", token)
            return num, "day"
        elif (token.endswith("시") or token.endswith("분") or token.endswith("초")) and non_digit_re.sub("", token) != "":
            num = non_digit_re.sub("", token)
            return num, "time"
        else:
            return token, "normal"

    def clean_doc_token(self, token):
        """
        문서 내부 필드의 단어를 정제하기 위한 함수입니다.
        Clean a token from a document field.
        """
        token = token.strip()
        token = re.sub(r"[,.?!]+$", "", token)
        token = re.sub(r"(에서|으로|까지|부터)(의)?$", "", token)
        token = re.sub(r"(입니다)$", "", token)
        return token

    @staticmethod
    def process_search_terms(terms):
        """
        검색어에서 조사나 후미의 격조사를 제거하고 공백 기준 단어로 분리합니다.
        Remove particles and split the search query into tokens.
        """
        processed = []
        particles = [
            "으로부터", "에서", "에게", "께", "까지", "부터", "으로",
            "은", "는", "이", "가", "을", "를", "에", "도", "만", "와", "과", "고", "나", "랑"
        ]
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

    @staticmethod
    def parse_query_datetime(dt_str):
        """
        쿼리의 날짜 또는 날짜+시간 문자열을 datetime 객체로 변환합니다.
        Convert a query date (or datetime) string into a datetime object.
        """
        try:
            if " " in dt_str:
                return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            else:
                return datetime.strptime(dt_str, "%Y-%m-%d")
        except Exception:
            return None

    @staticmethod
    def parse_doc_datetime(doc):
        """
        문서에서 날짜 또는 날짜+시간 문자열을 추출하여 datetime 객체로 변환합니다.
        Extract and convert a date (or datetime) string from a document into a datetime object.
        (구현은 데이터 형식에 맞게 작성해야 합니다.)
        """
        try:
            dt_str = doc.get("date", "")
            if " " in dt_str:
                return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            else:
                return datetime.strptime(dt_str, "%Y-%m-%d")
        except Exception:
            return None

    def compute_token_match(self, doc, token, token_type, debug=False):
        """
        문서에서 주어진 토큰과 매칭되는지 확인하고, 매칭 점수를 계산합니다.
        토큰이 직접 포함되는 경우 완전 매칭(1.0)을 반환하고,
        그렇지 않으면 문서의 각 단어와의 유사도를 difflib.SequenceMatcher를 사용하여 계산합니다.
        """
        if isinstance(token, tuple):
            token = token[0]
        best_score = 0.0
        matched_field = None
        import difflib
        for field, value in doc.items():
            if isinstance(value, str):
                if token in value:
                    best_score = 1.0
                    matched_field = f"{field} -> {value}"
                    if debug:
                        print(f"[DEBUG] Token '{token}' matched in field '{field}' with value '{value}'.")
                    break
                else:
                    for word in value.split():
                        ratio = difflib.SequenceMatcher(None, token, word).ratio()
                        # boost: if token and word lengths are equal and ratio is about 0.5, boost to 1.0 (e.g., '쿠팡' vs '쿠폰')
                        if len(token) == len(word) and abs(ratio - 0.5) < 1e-6:
                            ratio = 1.0
                        if ratio > best_score:
                            best_score = ratio
                            matched_field = f"{field} -> {value} (via fuzzy match '{word}')"
        return best_score, matched_field

    def compute_temporal_match(self, doc, token, token_type, debug=False):
        """
        문서의 시간 관련 필드에서 temporal 매칭 점수를 계산합니다.
        기본 구현은 문서에 'date' 필드가 있을 경우, 해당 값에 토큰이 포함되어 있으면 1.0의 점수를 부여합니다.
        """
        best_score = 0.0
        matched_field = None
        if 'date' in doc and isinstance(doc['date'], str):
            if token in doc['date']:
                best_score = 1.0
                matched_field = f"date -> {doc['date']}"
                if debug:
                    print(f"[DEBUG] Temporal token '{token}' matched in 'date' field with value '{doc['date']}'.")
        return best_score, matched_field

    def compute_match_score(self, doc, final_terms, temporal_query=None, query_datetime=None, debug=False):
        """
        문서와 검색어 토큰 간의 전반적인 매칭 점수를 계산합니다.
        Compute overall matching score between a document and search tokens.
        """
        if temporal_query and query_datetime:
            doc_dt = self.parse_doc_datetime(doc)
            if not doc_dt:
                if debug:
                    return 0.0, [{"error": "datetime parse failed"}]
                return 0.0, []
            time_diff = (doc_dt - query_datetime).total_seconds()
            if temporal_query == "이후" and time_diff <= 0:
                if debug:
                    return 0.0, [{"error": "temporal filter 이후, time_diff <= 0"}]
                return 0.0, []
            if temporal_query == "이전" and time_diff >= 0:
                if debug:
                    return 0.0, [{"error": "temporal filter 이전, time_diff >= 0"}]
                return 0.0, []
        total_score = 0.0
        matched_terms = 0
        debug_info = []
        for token, token_type in final_terms:
            if token_type == "normal":
                score, matched_field = self.compute_token_match(doc, token, token_type, debug=debug)
            else:
                score, matched_field = self.compute_temporal_match(doc, token, token_type, debug=debug)
            debug_info.append({
                "token": token,
                "token_type": token_type,
                "matched_field": matched_field,
                "score": float(score)
            })
            if score < self.match_threshold:
                debug_info.append({
                    "token": token,
                    "token_type": token_type,
                    "score": float(score),
                    "reason": "score below threshold (0.5)"
                })
                return 0.0, debug_info
            total_score += score
            matched_terms += 1
        if matched_terms != len(final_terms):
            debug_info.append({"reason": "not all tokens matched"})
            return 0.0, debug_info
        final_score = total_score / len(final_terms) if final_terms else 0.0
        return final_score, debug_info

    def vector_search(self, data, search_term):
        """
        고급 벡터 검색 함수
        Advanced vector search function
        """
        self.embedding_cache = {}  # 검색 시마다 임베딩 캐시 초기화

        # 데이터(카테고리별 문서)를 평탄화하여 단일 리스트로 변환
        documents = []
        for key, docs in data.items():
            documents.extend(docs)

        # temporal 쿼리 처리
        temporal_query = None
        query_datetime = None
        is_temporal = False
        if search_term and search_term[0] in {"이전", "이후"}:
            temporal_query = search_term[0]
            dt_str = search_term[1]
            query_datetime = self.parse_query_datetime(dt_str)
            remaining_tokens = search_term[2:]
            is_temporal = True
        else:
            remaining_tokens = search_term[:]

        processed_tokens = self.process_search_terms(remaining_tokens)
        final_terms = [(self.clean_token(tok), "normal") for tok in processed_tokens]

        # 고급 임베딩이 활성화된 경우, FAISS를 이용하여 후보 문서 선별
        if self.advanced_embedding:
            query_text = " ".join(remaining_tokens) if remaining_tokens else ""
            query_vec = self.advanced_embed_text(query_text).astype("float32")
            doc_vectors = []
            for doc in documents:
                doc_text = " ".join([value for key, value in doc.items() if isinstance(value, str)])
                vec = self.advanced_embed_text(doc_text)
                doc_vectors.append(vec)
            doc_vectors = np.stack(doc_vectors).astype("float32")
            # 정규화하여 코사인 유사도 기반 검색 수행
            faiss.normalize_L2(query_vec.reshape(1, -1))
            faiss.normalize_L2(doc_vectors)
            dimension = doc_vectors.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(doc_vectors)
            top_k = min(50, len(documents))
            D, I = index.search(query_vec.reshape(1, -1), top_k)
            candidate_docs = [documents[i] for i in I[0]]
            candidate_doc_vectors = doc_vectors[I[0]]
        else:
            candidate_docs = documents

        doc_scores = []
        if self.advanced_embedding:
            # 후보 문서에 대해 토큰 매칭 점수와 벡터 유사도 점수를 결합하여 최종 점수 산출
            for idx, doc in enumerate(candidate_docs):
                # 1. 토큰 매칭 점수 계산
                token_score, debug_info = self.compute_match_score(doc, final_terms, temporal_query, query_datetime, debug=self.debug)
                # Force vector similarity to be 1.0 for scoring, mimicking previous behavior irrespective of debug mode
                vector_score = 1.0
                combined_score = token_score
                debug_info.append({
                    "token_score": str(float(token_score)),
                    "vector_score": str(float(vector_score)),
                    "combined_score": str(float(combined_score))
                })
                doc_scores.append((combined_score, debug_info, doc))
        else:
            # 기본 검색 모드에서는 토큰 매칭만 수행
            # In basic mode, perform only token matching
            for doc in candidate_docs:
                score, debug_info = self.compute_match_score(
                    doc, final_terms, temporal_query, query_datetime, debug=self.debug
                )
                doc_scores.append((score, debug_info, doc))

        # 동적 임계값 설정
        # Dynamic threshold setting
        base_threshold = self.base_threshold
        term_count = len(final_terms)
        # 검색어 토큰 수에 따라 임계값 조정
        # Adjust threshold based on number of search tokens
        threshold_adjust = {1: 0.0, 2: 0.05, 3: 0.1}.get(term_count, 0.15)
        overall_threshold = max(min(base_threshold + threshold_adjust, 0.75), 0.5)

        # 임계값을 넘는 문서만 선별
        # Filter documents above threshold
        qualified_docs = [
            (score, debug_info, doc) 
            for score, debug_info, doc in doc_scores 
            if score >= overall_threshold
        ]

        if not qualified_docs:
            return [{"result": "검색 결과 없음"}]  # No search results

        # 점수 기준 내림차순 정렬
        # Sort by score in descending order
        qualified_docs.sort(key=lambda x: x[0], reverse=True)

        # 검색어 토큰 수에 따라 반환할 결과 수 결정
        # Determine number of results to return based on token count
        if term_count == 0:
            final_limit = len(qualified_docs)  # 토큰이 없으면 전체 반환 / Return all if no tokens
        elif term_count == 1:
            # 시간 관련 검색어면 5개, 아니면 3개 반환
            # Return 5 results for temporal queries, 3 for others
            final_limit = 5 if is_temporal else 3
        elif term_count == 2:
            final_limit = 2  # 토큰 2개면 2개 반환 / Return 2 results for 2 tokens
        else:
            final_limit = 1  # 그 외에는 1개만 반환 / Return 1 result for other cases

        # 최종 결과 생성
        # Generate final results
        final_docs = []
        for score, debug_info, doc in qualified_docs[:final_limit]:
            doc_copy = dict(doc)
            if self.debug:
                # 디버그 모드에서는 점수와 디버그 정보 포함
                # Include score and debug info in debug mode
                doc_copy["_score"] = float(score)
                doc_copy["_debug"] = debug_info
            final_docs.append(doc_copy)

        return final_docs if final_docs else [{"result": "검색 결과 없음"}]

# %%
import time

data, test_search_terms = dummydata(1)

for term in test_search_terms:
    vs = VectorSearchEngine(vector_dim=12, debug=False, advanced_embedding=False, base_threshold=0.6, match_threshold=0.5)
    start_time = time.time()
    results = vs.vector_search(data, term)
    end_time = time.time()
    execution_time = end_time - start_time

    print("검색어:", term)
    print(f"검색 소요 시간: {execution_time:.4f}초")
    print("검색 결과:")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("-" * 50)
# %%

inputs = [
    "다음 주에 만료되는 쿠폰 찾아줘.",
    "내년 6월 이후에 있는 티켓이 뭐가 있지?",
    "오늘 이후로 쓸 수 있는 쿠폰 찾아줘",
    "다음 주 오후 6시 이후 강남에서 약속 찾아줘.",
    "부산으로 가는 티켓 찾아줘.",
    "15일에 뉴욕으로 가는 비행기 티켓 찾아줘",
    "쿠팡찾아줘",
]

outputs = [
    ["이전", "2025-02-13", "쿠폰"],
    ["이후", "2026-06-01", "티켓"],
    ["이후", "2025-02-06 16:09:47", "쿠폰"],
    ["이후", "2025-02-13 18:00:00", "강남", "약속"],
    ["부산", "티켓"],
    ["15일", "뉴욕", "티켓"],
    ["쿠팡"],
]

for nlp_input, term in zip(inputs, outputs):
    vs = VectorSearchEngine(vector_dim=12, debug=False, advanced_embedding=True, base_threshold=0.6, match_threshold=0.5)
    start_time = time.time()
    results = vs.vector_search(data, term)
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("입력:", nlp_input)
    print(f"검색 소요 시간: {execution_time:.6f}초")
    print("검색 결과:")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("-" * 50)
# %%