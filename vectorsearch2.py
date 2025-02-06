# %%
import numpy as np
import re
from functools import lru_cache
import heapq
import json
"""
lru_cache 사용에 대한 설명 (Explanation for using lru_cache)

영어 (English):
lru_cache (Least Recently Used Cache) is a decorator provided by Python's functools module.
It caches the results of function calls based on their input arguments.
When the same arguments are passed again, the cached result is returned immediately,
eliminating the need for redundant computations.
This is particularly beneficial for functions like levenshtein_distance and normalized_edit_similarity,
which may be called repeatedly with the same string pairs.
By reusing computed results, lru_cache optimizes performance, reduces computational overhead,
and enhances the overall speed of the search operations.

한국어 (Korean):
lru_cache (최소 최근 사용 캐시)는 Python의 functools 모듈에서 제공하는 데코레이터입니다.
함수 호출 결과를 입력 인자값을 기준으로 캐싱합니다.
동일한 인자가 다시 전달되면, 캐시된 결과를 즉시 반환하여 불필요한 재계산을 방지합니다.
이는 특히 levenshtein_distance와 normalized_edit_similarity와 같이 동일한 문자열 쌍에 대해
반복적으로 계산이 이루어지는 함수에서 매우 유용합니다.
이미 계산된 결과를 재사용함으로써 성능을 최적화하고, 계산 오버헤드를 줄이며,
검색 작업의 전체 속도를 개선할 수 있습니다.
"""

# 벡터 검색 함수 정의
def vector_search(data, search_term, vector_dim=12, weight=0.5):
    """
    English:
    A sophisticated vector-based search function that finds relevant documents based on search terms.
    
    Parameters:
    - data: Dictionary containing categorized documents to search through
    - search_term: Search query string (can contain multiple words/terms)
    - vector_dim: Dimension of the embedding vectors (default=12)
    - weight: Balance between cosine similarity and edit distance (default=0.5)
           Higher weight gives more importance to cosine similarity
    
    The function works in multiple stages:
    1. Converts text to vectors using character-based embeddings
    2. Processes search terms by removing Korean particles
    3. Scores documents using both vector similarity and text matching
    4. Returns the most relevant documents
    
    한국어:
    검색어를 기반으로 관련 문서를 찾아주는 고급 벡터 기반 검색 함수입니다.
    
    매개변수:
    - data: 검색할 문서들이 카테고리별로 정리된 딕셔너리
    - search_term: 검색하고자 하는 문자열 (여러 단어/용어 포함 가능)
    - vector_dim: 임베딩 벡터의 차원 (기본값=12)
    - weight: 코사인 유사도와 편집 거리 간의 가중치 (기본값=0.5)
           높은 가중치는 코사인 유사도에 더 큰 중요성을 부여
    
    이 함수는 다음과 같은 단계로 작동합니다:
    1. 문자 기반 임베딩을 사용하여 텍스트를 벡터로 변환
    2. 한국어 조사를 제거하여 검색어 처리
    3. 벡터 유사도와 텍스트 매칭을 모두 사용하여 문서 점수 계산
    4. 가장 관련성 높은 문서들을 반환
    """
    # Global cache to store embedding results to avoid redundant calculations
    # 중복 계산을 피하기 위한 전역 캐시 - 한 번 계산된 임베딩은 재사용됨
    embedding_cache = {}
    
    # Regular expression that matches any non-digit character
    # 숫자가 아닌 모든 문자를 찾기 위한 정규표현식 패턴
    non_digit_re = re.compile(r"\D")

    # -------------------------------------------------
    # 1. embed_text 함수
    # -------------------------------------------------
    
    def embed_text(text, d):
        """
        English:
        Converts text into a fixed-size numerical vector representation.
        
        Process:
        1. First checks if the vector is already in cache to avoid recalculation
        2. Creates a zero vector of size 'd'
        3. For each character in text:
           - Converts it to ASCII value
           - Adds it to vector position using modulo for cycling
        4. Normalizes the final vector to unit length
        
        Parameters:
        - text: Input text to convert to vector
        - d: Dimension of the output vector
        
        Returns: Normalized vector representation of the text
        
        한국어:
        텍스트를 고정된 크기의 수치 벡터로 변환합니다.
        
        처리 과정:
        1. 먼저 캐시를 확인하여 이미 계산된 벡터가 있는지 확인
        2. 'd' 크기의 0으로 채워진 벡터 생성
        3. 텍스트의 각 문자에 대해:
           - ASCII 값으로 변환
           - 모듈로 연산을 사용해 순환하면서 벡터 위치에 더함
        4. 최종 벡터를 단위 길이로 정규화
        
        매개변수:
        - text: 벡터로 변환할 입력 텍스트
        - d: 출력 벡터의 차원
        
        반환값: 텍스트의 정규화된 벡터 표현
        """
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

    # -------------------------------------------------
    # 2. composite_query_vector 함수
    # -------------------------------------------------

    def composite_query_vector(final_terms):
        """
        English:
        Creates a single vector representation for multiple search terms.
        
        Process:
        1. Converts each search term to its vector representation
        2. Calculates the average (mean) of all term vectors
        3. Normalizes the resulting vector
        
        This allows multiple search terms to be compared with documents
        using a single vector comparison.
        
        Parameters:
        - final_terms: List of (token, token_type) tuples
        
        Returns: Normalized composite vector representing all search terms
        
        한국어:
        여러 검색어를 하나의 벡터로 표현합니다.
        
        처리 과정:
        1. 각 검색어를 벡터로 변환
        2. 모든 검색어 벡터의 평균(mean) 계산
        3. 결과 벡터를 정규화
        
        이를 통해 여러 검색어를 하나의 벡터 비교로
        문서들과 비교할 수 있게 됩니다.
        
        매개변수:
        - final_terms: (토큰, 토큰_유형) 튜플의 리스트
        
        반환값: 모든 검색어를 대표하는 정규화된 복합 벡터
        """
        vecs = [embed_text(token, vector_dim) for token, token_type in final_terms]
        comp = np.mean(vecs, axis=0)
        norm = np.linalg.norm(comp)
        if norm > 0:
            comp = comp / norm
        return comp

    # -------------------------------------------------
    # 3. doc_vector 함수
    # -------------------------------------------------

    def doc_vector(doc):
        """
        English:
        Creates a vector representation for a document by combining its text fields.
        
        Process:
        1. First tries to extract text from primary fields (title, description, etc.)
        2. If primary fields are not available, uses all string fields
        3. Combines all text, splits into tokens
        4. Creates vectors for each token and averages them
        
        The function is smart about field selection:
        - Prioritizes important fields like title and description
        - Falls back to all text fields if primary ones are missing
        - Handles empty documents by returning zero vector
        
        Parameters:
        - doc: Document dictionary containing various fields
        
        Returns: Normalized vector representation of the document
        
        한국어:
        문서의 텍스트 필드들을 결합하여 벡터로 표현합니다.
        
        처리 과정:
        1. 먼저 주요 필드(제목, 설명 등)에서 텍스트 추출 시도
        2. 주요 필드가 없다면 모든 문자열 필드 사용
        3. 모든 텍스트를 결합하고 토큰으로 분리
        4. 각 토큰의 벡터를 만들고 평균 계산
        
        이 함수는 다음과 같이 스마트하게 필드를 선택합니다:
        - 제목, 설명과 같은 중요 필드 우선 처리
        - 주요 필드가 없으면 모든 텍스트 필드로 대체
        - 빈 문서는 0 벡터를 반환하여 처리
        
        매개변수:
        - doc: 다양한 필드를 포함하는 문서 딕셔너리
        
        반환값: 문서의 정규화된 벡터 표현
        """
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

    # -------------------------------------------------
    # 4. process_search_terms 함수
    # -------------------------------------------------

    def process_search_terms(terms):
        """
        English:
        Processes and cleans search terms by removing Korean particles and suffixes.
        
        Process:
        1. Maintains a comprehensive list of Korean particles to remove
        2. Iteratively removes particles from the end of each term
        3. Splits remaining terms on spaces for additional granularity
        
        This preprocessing is crucial for Korean language search because:
        - Removes grammatical particles that don't affect meaning
        - Handles compound terms with multiple particles
        - Improves match accuracy by focusing on core words
        
        Parameters:
        - terms: List of original search terms
        
        Returns: List of processed terms with particles removed
        
        한국어:
        한국어 조사와 접미사를 제거하여 검색어를 처리하고 정제합니다.
        
        처리 과정:
        1. 제거할 한국어 조사의 포괄적인 목록 유지
        2. 각 검색어 끝에서 조사를 반복적으로 제거
        3. 남은 검색어를 공백 기준으로 분리하여 더 세밀하게 처리
        
        이 전처리는 다음과 같은 이유로 한국어 검색에 매우 중요합니다:
        - 의미에 영향을 주지 않는 문법적 조사 제거
        - 여러 조사가 결합된 복합 용어 처리
        - 핵심 단어에 집중하여 매칭 정확도 향상
        
        매개변수:
        - terms: 원본 검색어 리스트
        
        반환값: 조사가 제거된 처리된 검색어 리스트
        """
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

    # -------------------------------------------------
    # 5. clean_token 함수
    # -------------------------------------------------
    def clean_token(token):
        """
        English:
        Analyzes and classifies tokens based on their content type.
        
        Process:
        1. Checks if token contains date markers (년, 월, 일)
           - If yes, removes non-digit characters and labels as "date"
        2. Checks if token contains time markers (시, 분, 초)
           - If yes, removes non-digit characters and labels as "time"
        3. If neither date nor time markers are found
           - Labels as "normal" text
        
        This classification is important because:
        - Different token types require different matching strategies
        - Date/time tokens need special numeric comparison
        - Helps improve search accuracy for temporal information
        
        Parameters:
        - token: Input string to be classified
        
        Returns: Tuple of (processed_token, token_type)
        
        한국어:
        토큰의 내용을 분석하여 유형을 분류합니다.
        
        처리 과정:
        1. 날짜 표시자(년, 월, 일) 포함 여부 확인
           - 포함된 경우 숫자가 아닌 문자를 제거하고 "date"로 분류
        2. 시간 표시자(시, 분, 초) 포함 여부 확인
           - 포함된 경우 숫자가 아닌 문자를 제거하고 "time"으로 분류
        3. 날짜나 시간 표시자가 없는 경우
           - "normal" 텍스트로 분류
        
        이 분류 작업이 중요한 이유:
        - 토큰 유형에 따라 다른 매칭 전략 필요
        - 날짜/시간 토큰은 특별한 숫자 비교 필요
        - 시간 정보에 대한 검색 정확도 향상
        
        매개변수:
        - token: 분류할 입력 문자열
        
        반환값: (처리된_토큰, 토큰_유형) 튜플
        """
        if any(marker in token for marker in ["년", "월", "일"]):
            # 날짜 처리를 위해 숫자가 아닌 문자를 제거
            num = non_digit_re.sub("", token)
            return num, "date"
        elif any(marker in token for marker in ["시", "분", "초"]):
            # 시간 처리를 위해 숫자가 아닌 문자를 제거
            num = non_digit_re.sub("", token)
            return num, "time"
        else:
            return token, "normal"

    # -------------------------------------------------
    # 6. Levenshtein 및 normalized_edit_similarity 관련 함수
    # -------------------------------------------------
    @lru_cache(maxsize=None)
    def levenshtein_distance(s1, s2):
        """
        English:
        Calculates the Levenshtein (edit) distance between two strings.
        
        Process:
        1. Creates a dynamic programming matrix
        2. Fills matrix based on character comparisons
        3. Considers three operations:
           - Insertion: Adding a character
           - Deletion: Removing a character
           - Substitution: Replacing a character
        
        The distance represents:
        - Minimum number of single-character edits needed
        - To transform one string into another
        - Lower values indicate more similar strings
        
        Parameters:
        - s1, s2: Two strings to compare
        
        Returns: Minimum edit distance (integer)
        
        한국어:
        두 문자열 간의 레벤슈타인(편집) 거리를 계산합니다.
        
        처리 과정:
        1. 동적 프로그래밍 행렬 생성
        2. 문자 비교를 기반으로 행렬 채우기
        3. 세 가지 연산 고려:
           - 삽입: 문자 추가
           - 삭제: 문자 제거
           - 대체: 문자 교체
        
        이 거리가 의미하는 것:
        - 한 문자열을 다른 문자열로 변환하는데 필요한
        - 최소 단일 문자 편집 횟수
        - 값이 낮을수록 더 유사한 문자열
        
        매개변수:
        - s1, s2: 비교할 두 문자열
        
        반환값: 최소 편집 거리(정수)
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # dp 행렬의 첫 행과 첫 열을 초기화
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # 삭제, 삽입 또는 대체 연산 중 최솟값 선택
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]
    
    @lru_cache(maxsize=None)
    def normalized_edit_similarity(s1, s2):
        """
        English:
        Calculates a normalized similarity score between two strings.
        
        Process:
        1. Gets Levenshtein distance between strings
        2. Normalizes by dividing by max string length
        3. Converts to similarity score (1 - normalized distance)
        
        Features:
        - Returns value between 0 and 1
        - 1 means identical strings
        - 0 means completely different
        - Accounts for string length differences
        
        Parameters:
        - s1, s2: Two strings to compare
        
        Returns: Normalized similarity score (float)
        
        한국어:
        두 문자열 간의 정규화된 유사도 점수를 계산합니다.
        
        처리 과정:
        1. 문자열 간 레벤슈타인 거리 계산
        2. 최대 문자열 길이로 나누어 정규화
        3. 유사도 점수로 변환 (1 - 정규화된 거리)
        
        특징:
        - 0과 1 사이의 값 반환
        - 1은 완전히 동일한 문자열
        - 0은 완전히 다른 문자열
        - 문자열 길이 차이를 고려
        
        매개변수:
        - s1, s2: 비교할 두 문자열
        
        반환값: 정규화된 유사도 점수(실수)
        """
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1
        return 1 - (levenshtein_distance(s1, s2) / max_len)
    
    # -------------------------------------------------
    # 7. score_doc 함수
    # -------------------------------------------------
    def score_doc(doc, query, query_vec, token_type):
        """
        English:
        Evaluates how well a document matches a search query token.
        
        Process:
        1. Different scoring strategies for different token types:
           - Date tokens: Checks date-related fields first
           - Time tokens: Checks time-related fields first
           - Normal tokens: Checks all text fields
        2. For each field:
           - Checks for exact matches (returns 0 for perfect match)
           - Calculates cosine similarity with query vector
           - Calculates normalized edit distance
           - Combines scores using weighted average
        
        Features:
        - Intelligent field selection based on token type
        - Handles missing fields gracefully
        - Returns scores where lower is better
        
        Parameters:
        - doc: Document to score
        - query: Search token
        - query_vec: Vector representation of query
        - token_type: Type of token (date/time/normal)
        
        Returns: Score (float, lower is better)
        
        한국어:
        문서가 검색어 토큰과 얼마나 잘 일치하는지 평가합니다.
        
        처리 과정:
        1. 토큰 유형별 다른 점수 계산 전략:
           - 날짜 토큰: 날짜 관련 필드 우선 확인
           - 시간 토큰: 시간 관련 필드 우선 확인
           - 일반 토큰: 모든 텍스트 필드 확인
        2. 각 필드에 대해:
           - 정확한 일치 확인 (완벽 일치시 0 반환)
           - 쿼리 벡터와의 코사인 유사도 계산
           - 정규화된 편집 거리 계산
           - 가중 평균으로 점수 결합
        
        특징:
        - 토큰 유형에 따른 지능적 필드 선택
        - 누락된 필드 처리 가능
        - 낮을수록 좋은 점수 반환
        
        매개변수:
        - doc: 점수를 매길 문서
        - query: 검색 토큰
        - query_vec: 쿼리의 벡터 표현
        - token_type: 토큰 유형 (날짜/시간/일반)
        
        반환값: 점수 (실수, 낮을수록 좋음)
        """
        best = 1.0  # 최악의 점수(1.0)로 초기화
        
        # 날짜 처리
        if token_type == "date":
            date_keys = ["date", "departure_date", "valid_until"]
            # 문서 내에서 날짜 관련 필드 우선 탐색
            fields = [(k, v) for k, v in doc.items() if k in date_keys and isinstance(v, str)]
            if not fields:
                # 날짜 필드가 없으면, 모든 문자열 필드 사용
                fields = [(k, v) for k, v in doc.items() if isinstance(v, str)]
            for key, value in fields:
                # '-' 기준으로 분리된 날짜 부분을 검사
                segments = value.split("-")
                for seg in segments:
                    if query == seg.strip():
                        # 완벽한 일치시 0 반환
                        return 0
                seg_scores = []
                # 각 날짜 세그먼트에 대해 점수 산출 (코사인 유사도와 편집 유사도를 가중 평균)
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
        
        # 시간 처리
        elif token_type == "time":
            time_keys = ["time", "departure_time"]
            # 문서 내에서 시간 관련 필드 우선 탐색
            fields = [(k, v) for k, v in doc.items() if k in time_keys and isinstance(v, str)]
            if not fields:
                # 시간 필드가 없으면 모든 문자열 필드 사용
                fields = [(k, v) for k, v in doc.items() if isinstance(v, str)]
            for key, value in fields:
                # ':' 기준으로 분리된 시간 부분을 검사
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
            # 일반 텍스트 처리
            query_lower = query.lower()
            for value in doc.values():
                if isinstance(value, str):
                    # 검색어가 부분 문자열로 포함되어 있으면 즉시 0 반환
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
    
    # Preprocessing term: eliminate the suffix and clean the token
    # 검색어 전처리: 조사 제거 및 토큰 정제
    processed_tokens = process_search_terms(search_term)
    final_terms = [clean_token(tok) for tok in processed_tokens]
    original_term_count = len(final_terms)
    
    # Combine all documents from all categories into a single list
    # 모든 카테고리의 문서를 하나의 리스트로 통합
    documents = [doc for group in data.values() for doc in group]
    
    # -------------------------------------------------
    # Stage 1: 첫 번째 검색어 토큰을 이용한 후보군 선별
    # -------------------------------------------------
    first_query, first_type = final_terms[0]
    first_query_vec = embed_text(first_query, vector_dim)
    stage_scores = [(score_doc(doc, first_query, first_query_vec, first_type), idx)
                    for idx, doc in enumerate(documents)]
    threshold_1 = 3 if original_term_count == 1 else 5  # 토큰 개수에 따라 후보 갯수 결정
    candidate_indices = [idx for _, idx in heapq.nsmallest(threshold_1, stage_scores, key=lambda x: x[0])]
    
    # -------------------------------------------------
    # Stage 2: 두 번째 검색어 토큰이 있을 경우 후보군 재필터링
    # -------------------------------------------------
    if original_term_count > 1:
        second_query, second_type = final_terms[1]
        second_query_vec = embed_text(second_query, vector_dim)
        stage_scores = [(score_doc(documents[idx], second_query, second_query_vec, second_type), idx)
                        for idx in candidate_indices]
        threshold_2 = 2 if original_term_count == 2 else 3
        candidate_indices = [idx for _, idx in heapq.nsmallest(threshold_2, stage_scores, key=lambda x: x[0])]
    
    # -------------------------------------------------
    # Stage 3: 세 번째 검색어 토큰이 있을 경우 최종 후보군 재필터링
    # -------------------------------------------------
    if original_term_count > 2:
        third_query, third_type = final_terms[2]
        third_query_vec = embed_text(third_query, vector_dim)
        stage_scores = [(score_doc(documents[idx], third_query, third_query_vec, third_type), idx)
                        for idx in candidate_indices]
        candidate_indices = [idx for _, idx in heapq.nsmallest(1, stage_scores, key=lambda x: x[0])]
    
    # -------------------------------------------------
    # compute_rerank_score 함수: 문서 재정렬 점수 계산
    # -------------------------------------------------
    def compute_rerank_score(doc, final_terms):
        """
        English:
        Computes a re-ranking score for a document by evaluating normalized edit distances for
        each token against the document's text fields and averaging the best few distances.
        
        한국어:
        각 검색어 토큰에 대해 문서 내 문자열 필드와의 정규화된 편집 거리를 평가하고,
        상위 몇 개의 값을 평균 내어 재정렬 점수를 산출합니다.
        """
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
    
    # -------------------------------------------------
    # 후보군 재정렬 (검색어 토큰이 2개 이하일 경우)
    # -------------------------------------------------
    if original_term_count <= 2:
        candidate_scores = [(compute_rerank_score(documents[idx], final_terms), idx) for idx in candidate_indices]
        candidate_scores.sort(key=lambda x: x[0])
        candidate_indices = [idx for score, idx in candidate_scores]
    
    # Calculate threshold based on rerank scores
    # 전체 문서의 재정렬 점수를 기반으로 임계값(threshold) 산출
    all_scores = [compute_rerank_score(doc, final_terms) for doc in documents]
    sorted_scores = sorted(all_scores)
    top_k = 3
    if len(sorted_scores) <= top_k:
        threshold = 100
    else:
        below_top = sorted_scores[top_k:]
        threshold = sum(below_top) / len(below_top)
    
    best_candidate_score = compute_rerank_score(documents[candidate_indices[0]], final_terms)
    # Return "검색 결과 없음" if the best candidate score is higher than the threshold
    # 최적 후보 점수가 threshold보다 높으면 "검색 결과 없음" 반환
    if best_candidate_score > threshold:
        return [{"result": "검색 결과 없음"}]
    
    # Return the best candidate document
    # 최적 후보 문서 반환
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
