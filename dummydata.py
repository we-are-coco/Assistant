def dummydata(dataset = 1):
    data1 = {
        "쿠폰": [
            {
                "category": "쿠폰",
                "brand": "ABC마트",
                "type": "식품",
                "item": "커피 할인권",
                "date": "2025-02-11",
                "time": "12:00",
                "code": "ABC123XYZ",
                "description": "커피 할인 쿠폰입니다."
            },
            {
                "category": "쿠폰",
                "brand": "XYZ슈퍼",
                "type": "의류",
                "item": "정장 할인권",
                "date": "2025-03-15",
                "time": "12:00",
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
                "date": "2025-03-15",
                "time": "08:30",
                "description": "서울에서 부산까지의 버스 티켓입니다."
            },
            {
                "category": "교통",
                "type": "기차",
                "from_location": "서울역",
                "to_location": "대구역",
                "date": "2026-07-01",
                "time": "09:15",
                "description": "서울에서 대구까지의 기차 티켓입니다."
            },
            {
                "category": "교통",
                "type": "비행기",
                "from_location": "인천",
                "to_location": "뉴욕",
                "date": "2025-06-15",
                "time": "07:00",
                "description": "인천에서 뉴욕까지의 비행기 티켓입니다."
            }
        ],
        "엔터테인먼트": [
            {
                "category": "엔터테인먼트",
                "type": "영화",
                "title": "영화 티켓 할인",
                "date": "2025-02-07",
                "time": "19:00",
                "location": "CGV 강남",
                "description": "영화 할인 티켓입니다."
            },
            {
                "category": "엔터테인먼트",
                "type": "콘서트",
                "title": "록 페스티벌",
                "date": "2025-09-25",
                "time": "18:30",
                "location": "잠실 실내체육관",
                "description": "록 콘서트 티켓입니다."
            },
            {
                "category": "엔터테인먼트",
                "type": "전시",
                "title": "미술 전시회",
                "date": "2025-02-01",
                "time": "15:00",
                "location": "서울 아트센터",
                "description": "미술 전시회 티켓입니다."
            }
        ],
        "약속": [
            {
                "category": "약속",
                "type": "미팅",
                "date": "2025-02-14",
                "time": "19:00",
                "location": "강남 스타벅스",
                "details": "비즈니스 미팅",
                "description": "비즈니스 미팅 약속입니다."
            },
            {
                "category": "약속",
                "type": "의료",
                "date": "2025-01-05",
                "time": "10:00",
                "location": "서울병원",
                "details": "의사 상담 예약",
                "description": "진료 예약 약속입니다."
            },
            {
                "category": "약속",
                "type": "식당",
                "date": "2025-03-12",
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

    data2 = {
        "쿠폰": [
            {
                "category": "쿠폰",
                "brand": "파리바게트",
                "type": "베이커리",
                "item": "빵 할인 쿠폰",
                "date": "2025-04-01",
                "time": "11:00",
                "code": "PB2025BREAD",
                "description": "파리바게트 빵 할인 쿠폰입니다."
            },
            {
                "category": "쿠폰",
                "brand": "이마트",
                "type": "식품",
                "item": "신선식품 할인 쿠폰",
                "date": "2025-04-05",
                "time": "10:30",
                "code": "EMARTFRESH25",
                "description": "이마트 신선식품 할인 쿠폰입니다."
            }
        ],
        "교통": [
            {
                "category": "교통",
                "type": "버스",
                "from_location": "부산역",
                "to_location": "울산역",
                "date": "2025-05-10",
                "time": "08:15",
                "description": "부산에서 울산까지의 버스 티켓입니다."
            },
            {
                "category": "교통",
                "type": "지하철",
                "from_location": "서울역",
                "to_location": "인천역",
                "date": "2025-05-12",
                "time": "09:00",
                "description": "서울에서 인천까지의 지하철 이용권입니다."
            },
            {
                "category": "교통",
                "type": "택시",
                "from_location": "강남",
                "to_location": "김포공항",
                "date": "2025-05-15",
                "time": "07:30",
                "description": "강남에서 김포공항까지 택시 예약 내역입니다."
            }
        ],
        "엔터테인먼트": [
            {
                "category": "엔터테인먼트",
                "type": "공연",
                "title": "뮤지컬 '라라랜드'",
                "date": "2025-06-20",
                "time": "18:00",
                "location": "대전 예술의전당",
                "description": "뮤지컬 라라랜드 할인 티켓입니다."
            },
            {
                "category": "엔터테인먼트",
                "type": "전시",
                "title": "현대미술 전시",
                "date": "2025-06-25",
                "time": "14:00",
                "location": "서울 현대미술관",
                "description": "현대미술 전시 할인 티켓입니다."
            },
            {
                "category": "엔터테인먼트",
                "type": "콘서트",
                "title": "K팝 콘서트",
                "date": "2025-07-05",
                "time": "20:00",
                "location": "부산 벡스코",
                "description": "K팝 콘서트 할인 티켓입니다."
            }
        ],
        "약속": [
            {
                "category": "약속",
                "type": "비즈니스",
                "date": "2025-03-20",
                "time": "15:30",
                "location": "강남 코엑스",
                "details": "파트너사 미팅",
                "description": "강남 코엑스에서의 비즈니스 미팅 예약입니다."
            },
            {
                "category": "약속",
                "type": "의료",
                "date": "2025-03-22",
                "time": "10:00",
                "location": "분당 서울병원",
                "details": "내과 상담",
                "description": "분당 서울병원 내과 상담 예약입니다."
            },
            {
                "category": "약속",
                "type": "기타",
                "date": "2025-03-25",
                "time": "19:00",
                "location": "홍대 카페",
                "details": "친구와 만남",
                "description": "홍대에서 친구와 만남 약속입니다."
            }
        ],
        "불명": [
            {
                "category": "불명",
                "type": "알 수 없음",
                "date": "2023-10-10",
                "time": "12:00",
                "description": "정보가 누락되었습니다."
            },
            {
                "category": "불명",
                "type": "미정",
                "date": "2023-11-11",
                "time": "13:00",
                "description": "추가 정보 필요."
            },
            {
                "category": "불명",
                "type": "회의",
                "date": "2023-12-01",
                "time": "14:45",
                "description": "회의록 검토 중입니다."
            }
        ],
        "기타": [
            {
                "category": "기타",
                "description": "기타 관련 내용입니다. 자세한 정보는 추후 업데이트 예정."
            },
            {
                "category": "기타",
                "description": "임시 데이터: 테스트 항목입니다."
            },
            {
                "category": "기타",
                "description": "예비 정보로만 사용됩니다."
            }
        ]
    }

    data3 = {
        "쿠폰": [
            {
                "category": "쿠폰",
                "brand": "롯데마트",
                "type": "식품",
                "item": "과자 할인권",
                "date": "2025-07-01",
                "time": "10:00",
                "code": "LOTTE2025",
                "description": "롯데마트 과자 할인쿠폰입니다."
            },
            {
                "category": "쿠폰",
                "brand": "GS슈퍼",
                "type": "전자제품",
                "item": "가전제품 할인권",
                "date": "2025-07-05",
                "time": "15:00",
                "code": "GS2025ELEC",
                "description": "GS슈퍼 가전 할인쿠폰입니다."
            }
        ],
        "교통": [
            {
                "category": "교통",
                "type": "지하철",
                "from_location": "서울역",
                "to_location": "수원역",
                "date": "2025-07-10",
                "time": "09:45",
                "description": "서울에서 수원까지 지하철 운행 정보입니다."
            },
            {
                "category": "교통",
                "type": "버스",
                "from_location": "대구",
                "to_location": "부산",
                "date": "2025-07-12",
                "time": "08:30",
                "description": "대구에서 부산으로 이동하는 버스 정보입니다."
            },
            {
                "category": "교통",
                "type": "기차",
                "from_location": "부산역",
                "to_location": "대전역",
                "date": "2025-07-15",
                "time": "13:20",
                "description": "부산에서 대전까지의 기차 티켓 정보입니다."
            }
        ],
        "엔터테인먼트": [
            {
                "category": "엔터테인먼트",
                "type": "영화",
                "title": "블루밍",
                "date": "2025-08-05",
                "time": "20:00",
                "location": "CGV 압구정",
                "description": "블루밍 영화 티켓 할인 이벤트."
            },
            {
                "category": "엔터테인먼트",
                "type": "콘서트",
                "title": "오로라 콘서트",
                "date": "2025-08-10",
                "time": "19:30",
                "location": "잠실 올림픽공원",
                "description": "오로라 콘서트 할인 티켓."
            },
            {
                "category": "엔터테인먼트",
                "type": "연극",
                "title": "햄릿",
                "date": "2025-08-12",
                "time": "18:00",
                "location": "서울 예술의전당",
                "description": "햄릿 연극 할인 티켓."
            }
        ],
        "약속": [
            {
                "category": "약속",
                "type": "비즈니스",
                "date": "2025-07-20",
                "time": "14:00",
                "location": "코엑스",
                "details": "미팅 예약",
                "description": "코엑스에서 파트너 미팅."
            },
            {
                "category": "약속",
                "type": "의료",
                "date": "2025-07-22",
                "time": "11:15",
                "location": "강남세브란스",
                "details": "내과 상담",
                "description": "강남세브란스 내과 예약."
            },
            {
                "category": "약속",
                "type": "식당",
                "date": "2025-07-25",
                "time": "19:45",
                "location": "홍대",
                "details": "저녁 식사",
                "description": "홍대에서 저녁 식사 약속."
            }
        ],
        "불명": [
            {
                "category": "불명",
                "type": "미정",
                "date": "2024-10-05",
                "time": "13:30",
                "description": "정보 제공 예정입니다."
            },
            {
                "category": "불명",
                "type": "알 수 없음",
                "date": "2024-11-11",
                "time": "12:00",
                "description": "세부 정보 없음."
            },
            {
                "category": "불명",
                "type": "미정",
                "date": "2024-12-01",
                "time": "14:20",
                "description": "상세 정보 추후 업데이트."
            }
        ],
        "기타": [
            {
                "category": "기타",
                "description": "추가 공지사항 및 업데이트 예정."
            },
            {
                "category": "기타",
                "description": "예비 데이터 입니다."
            },
            {
                "category": "기타",
                "description": "기타 관련 임시 정보."
            }
        ]
    }
    test_search_terms1 = [
        ["이전", "2025-02-13", "쿠폰"],
        ["이후", "2025-06-01", "티켓"],
        ["이전", "2025-02-06 22:00:00", "쿠폰"],
        ["이후", "2025-02-13 18:00:00", "약속"],
        ["부산", "티켓"],
        ["15일", "뉴욕", "티켓"],
        ["강남에서", "영화"],
        ["잠실에서", "지하철"],
        ["친구랑", "미팅"],
        ["정장", "할인"],
        ["대전", "기차", "광주"],
        ["스타벅스", "강남"],
        ["11월", "전시"],
        ["15일에", "부산으로", "버스표"],
        ['7시', '광주']
    ]

    test_search_terms2 = [
        ["이후", "2025-04-03", "쿠폰"],
        ["이후", "2025-05-11", "교통"],
        ["이전", "2025-05-12", "택시"],
        ["부산", "버스"],
        ["15일", "김포", "택시"],
        ["강남", "코엑스", "미팅"],
        ["서울", "전시"],
        ["파리바게트", "빵"],
        ["업데이트", "예정"],
        ["분당", "내과", "상담"],
        ["이전", "2025-06-01", "콘서트"]
    ]

    test_search_terms3 = [
        ["이후", "2025-07-02", "쿠폰"],
        ["이전", "2025-07-13", "기차"],
        ["서울", "수원", "지하철"],
        ["잠실", "콘서트"],
        ["강남세브란스", "내과", "상담"],
        ["예술의전당", "햄릿"],
        ["코엑스", "미팅"],
        ["대구", "부산", "버스"],
        ["홍대", "식당"]
    ]
    if dataset == 1:
        return data1, test_search_terms1
    elif dataset == 2:
        return data2, test_search_terms2
    elif dataset == 3:
        return data3, test_search_terms3

