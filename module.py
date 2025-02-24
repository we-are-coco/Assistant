# %%
import json
import base64
from openai import AzureOpenAI  
import re
import os
from dotenv import load_dotenv
from pathlib import Path


class AImodule:
    def __init__(self, subscription_key=None):
        # 현재 파일의 디렉토리에서 .env 파일 경로 설정
        env_path = Path(__file__).parent / '.env'
        load_dotenv(env_path)
        
        if subscription_key:
            self.subscription_key = subscription_key
        else:
            # .env 파일에서 API 키 가져오기
            self.subscription_key = os.getenv('AZURE_API_KEY')
            if not self.subscription_key:
                self.subscription_key = input("Azure OpenAI API 키를 입력하세요: ")
                
        self.endpoint_url = "https://team2openainorthcentralus.openai.azure.com/"
        self.deployment_name = "gpt-4o"
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint_url,
            api_key=self.subscription_key,
            api_version="2024-08-01-preview",
        )

    def extract_json_from_string(self, input_data):
        if isinstance(input_data, (list, dict)):
            return input_data
        elif isinstance(input_data, str):
            try:
                json_pattern = re.compile(r"```json\n([\s\S]*?)\n```")
                match = json_pattern.search(input_data)
                if match:
                    json_str = match.group(1)
                    return json.loads(json_str)
                else:
                    return json.loads(input_data)
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                return None
        else:
            print("지원되지 않는 데이터 형식입니다.")
            return None

    def call_azure_api(self, prompt: str, image: str) -> str:
        encoded_image = base64.b64encode(open(image, 'rb').read()).decode('ascii')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        },
                    },
                ],
            }
        ]

        completion = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )

        return completion.choices[0].message.content

    def analyze_image(self, image: str) -> str:
        unified_prompt = """
        You are an expert image analyzer and information extractor specialized in schedule-related images. Please analyze the provided image and perform the following tasks:

        1. Classify the image into ONE of the following categories:
           - "Coupon": Digital gift certificates, vouchers, mobile coupons, or any digital form of store credit
           - "Transportation": Any tickets related to transportation including train, bus, plane, or ferry tickets
           - "Entertainment": Tickets or bookings for leisure activities such as movies, concerts, theater performances, exhibitions, or sports events
           - "Appointment": Any form of scheduled meetings or reservations including text messages, chat conversations, reservation confirmations, calendar screenshots, or meeting invitations
           - "Others": Images that don't contain any schedule-related information

        2. Based on the classified category, extract relevant information using the appropriate format below:
    
           - For Coupon:
             {
                 "category": "쿠폰",
                 "brand": "가게 혹은 브랜드 이름",
                 "type": "물건, 음식 등 상품의 종류",
                 "item": "물건 혹은 서비스 이름",
                 "date": "YYYY-MM-DD",
                 "time": "HH:MM",
                 "code": "바코드 혹은 시리얼 번호",
                 "description" : "item의 유형 및 정보에 해당하는 태그 2~3개(예: 카페, 편의점, 커피, 디저트, 치킨, 피자 등)를 띄어쓰기를 구분으로 작성"
             }
 
           - For Transportation:
             {
                 "category": "교통",
                 "type": "기차/버스/비행기",
                 "from_location": "출발 장소",
                 "to_location": "도착 장소",
                 "date": "YYYY-MM-DD",
                 "time": "HH:MM",
                 "description": ""from" 출발 "to" 도착 "type" 추가 정보(좌석정보, 사용기한 등)"
                 - if "from" and "to" are not located in South Korea, translate the name of the cities in Korean and include the name of the country at description field.
             }
 
           - For Entertainment:
             {
                 "category": "엔터테인먼트",
                 "type": "영화/콘서트/전시",
                 "title": "이벤트 이름",
                 "date": "YYYY-MM-DD",
                 "time": "HH:MM",
                 "location": "장소 이름",
                 "description": "내용을 요약하여 작성"
             }
 
           - For Appointment:
             {
                 "category": "약속",
                 "type": "미팅/의료/식당/등등",
                 "date": "YYYY-MM-DD",
                 "time": "HH:MM",
                 "location": "장소 이름",
                 "details": "추가 정보",
                 "description": "내용을 요약하여 작성"
             }
 
            - For Unsure:
            {
                "category": "불명",
                "type": "정보 유형",
                "date": "YYYY-MM-DD",
                "time": "HH:MM",
                "description": "내용을 요약하여 작성"
            }        
 
           - For Others:
             {
                 "category": "기타",
                 "description": "재미있는 문구를 넣어서 이미지에 대한 정보를 요약"
             }
 
        IMPORTANT INSTRUCTIONS:
        - Choose EXACTLY ONE category that best describes the image
        - Use EXACTLY the format shown above for the identified category
        - Use "null" if information is not available
        - Return ONLY the JSON object, no additional text
        - If the image contains multiple items, return multiple JSON objects in exactly the same format as shown above
        - Keep all responses in Korean language including names of brands and cities
        - If unsure or if the image doesn't clearly fit into the first four categories, classify as "Others"
        """

        answer = self.call_azure_api(unified_prompt, image)
        answer_json = self.extract_json_from_string(answer)

        return answer_json
    
if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    image_path = "img/flight.png"
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    ai_module = AImodule()  # AImodule 인스턴스 생성
    result = ai_module.analyze_image(image_path)  # 인스턴스를 통해 메서드 호출
    print(result)


# %%
