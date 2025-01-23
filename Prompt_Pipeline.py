# %%
import requests
import json
import os  
import base64
from openai import AzureOpenAI  
import json
import re
import os

file_path = 'key.json'
with open(file_path, 'r') as file:
    api = json.load(file)

def extract_json_from_string(input_data):
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
 
def call_azure_api(prompt: str, image: str) -> str:
    ENDPOINT_URL = "https://team2northcentralus.openai.azure.com/"
    DEPLOYMENT_NAME = "gpt-4o"
    AZURE_OPENAI_API_KEY = api['azure']

    endpoint = os.getenv("ENDPOINT_URL", ENDPOINT_URL)  
    deployment = os.getenv("DEPLOYMENT_NAME", DEPLOYMENT_NAME)  
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY)  

    client = AzureOpenAI(  
        azure_endpoint=endpoint,  
        api_key=subscription_key,  
        api_version="2024-08-01-preview",
    )


    IMAGE_PATH = image
    encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

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

    completion = client.chat.completions.create(  
        model=deployment,
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

# 함수 호출
#prompt = "이 이미지에서 스케줄 관련 정보를 추출하고 JSON 형식으로 정리해줘."
#image_path = "img/image (1).jpg"

#try:
#    result = call_azure_api(prompt, image_path)
#    print(result)
#except Exception as e:
#    print(f"오류 발생: {e}")

# %%
def thinking(image: str) -> str:
    first_prompt = """
    You are an expert image analyzer specialized in categorizing schedule-related images. Please analyze the provided image and classify it into ONE of the following categories:

    1. "Gifticon": Digital gift certificates, vouchers, mobile coupons, or any digital form of store credit
    2. "Transportation": Any tickets related to transportation including:
       - Train tickets
       - Bus tickets
       - Plane tickets
       - Ferry tickets
    3. "Entertainment": Tickets or bookings for leisure activities such as:
       - Movie tickets
       - Concert tickets
       - Theater performances
       - Exhibition entries
       - Sports events
    4. "Appointment": Any form of scheduled meetings or reservations including:
       - Text messages about appointments
       - Chat conversations about scheduling
       - Reservation confirmations
       - Calendar screenshots
       - Meeting invitations
    5. "Others": Images that don't contain any schedule-related information

    IMPORTANT INSTRUCTIONS:
    - Choose EXACTLY ONE category that best describes the image
    - Return ONLY the following JSON format:
    {
        "category": "ONE_OF_THE_ABOVE_CATEGORIES"
    }
    - Do not include any additional text or explanations
    - If unsure or if the image doesn't clearly fit into the first four categories, classify as "Others"
    """

    answer1 = call_azure_api(first_prompt, image)

    answer1_json = extract_json_from_string(answer1)
    if answer1_json is None:
        return {"error": "Failed to extract JSON data"}

    category = answer1_json.get("category", "Others")
    
    second_prompt = """
    You are an expert information extractor. The image has been previously classified as: """ + category + """

    Based on the category, extract information using EXACTLY one of these formats:

    1. For Gifticon:
    {
        "category": "Gifticon",
        "brand": "Store or brand name",
        "item": "Product or service name",
        "valid_until": "YYYY-MM-DD",
        "code": "Barcode or serial number"
    }

    2. For Transportation:
    {
        "category": "Transportation",
        "type": "Train/Bus/Plane",
        "from": "Departure location",
        "to": "Arrival location",
        "date": "YYYY-MM-DD",
        "time": "HH:MM"
    }

    3. For Entertainment:
    {
        "category": "Entertainment",
        "type": "Movie/Concert/Exhibition",
        "title": "Event name",
        "date": "YYYY-MM-DD",
        "time": "HH:MM",
        "location": "Venue name"
    }

    4. For Appointment:
    {
        "category": "Appointment",
        "type": "Meeting/Medical/Restaurant/etc",
        "date": "YYYY-MM-DD",
        "time": "HH:MM",
        "location": "Place name",
        "details": "Additional info"
    }

    5. For Others:
    {
        "category": "Others",
        "description": "Brief description of content"
    }

    IMPORTANT:
    - Use EXACTLY the format shown above for the identified category
    - Use "null" if information is not available
    - Return ONLY the JSON object, no additional text
    - Keep all responses in Korean language
    """

    answer2 = call_azure_api(second_prompt, image)
    answer2_json = extract_json_from_string(answer2)
    print("answer2:", answer2)
    print(answer2_json, type(answer2_json))

    return answer2_json


# %%

def thinking2(image: str) -> str:
    unified_prompt = """
    You are an expert image analyzer and information extractor specialized in schedule-related images. Please analyze the provided image and perform the following tasks:

    1. Classify the image into ONE of the following categories:
       - "Gifticon": Digital gift certificates, vouchers, mobile coupons, or any digital form of store credit
       - "Transportation": Any tickets related to transportation including train, bus, plane, or ferry tickets
       - "Entertainment": Tickets or bookings for leisure activities such as movies, concerts, theater performances, exhibitions, or sports events
       - "Appointment": Any form of scheduled meetings or reservations including text messages, chat conversations, reservation confirmations, calendar screenshots, or meeting invitations
       - "Others": Images that don't contain any schedule-related information

    2. Based on the classified category, extract relevant information using the appropriate format below:

       - For Gifticon:
         {
             "category": "기프티콘",
             "brand": "가게 혹은 브랜드 이름",
             "item": "물건 혹은 서비스 이름",
             "valid_until": "YYYY-MM-DD",
             "code": "바코드 혹은 시리얼 번호",
             "description": "내용을 간략하게 번역하여 작성"
         }

       - For Transportation:
         {
             "category": "교통",
             "type": "기차/버스/비행기",
             "from": "출발 장소",
             "to": "도착 장소",
             "departure_date": "YYYY-MM-DD",
             "departure_time": "HH:MM",
             "description": "내용을 간략하게 번역하여 작성"
         }

       - For Entertainment:
         {
             "category": "엔터테인먼트",
             "type": "영화/콘서트/전시",
             "title": "이벤트 이름",
             "date": "YYYY-MM-DD or YYYY-MM-DD ~ YYYY-MM-DD",
             "time": "HH:MM",
             "location": "장소 이름",
             "description": "내용을 간략하게 번역하여 작성"
         }

       - For Appointment:
         {
             "category": "약속",
             "type": "미팅/의료/식당/등등",
             "date": "YYYY-MM-DD or YYYY-MM-DD ~ YYYY-MM-DD",
             "time": "HH:MM",
             "location": "장소 이름",
             "details": "추가 정보",
             "description": "내용을 간략하게 번역하여 작성"
         }

        - For Unsure:
        {
            "category": "불명",
            "type": "정보 유형",
            "date": "YYYY-MM-DD or YYYY-MM-DD ~ YYYY-MM-DD",
            "time": "HH:MM" or "HH:MM ~ HH:MM",
            "description": "내용 간략 설명을 재미있게 그리고 디테일하게 번역하여 작성"
        }        

       - For Others:
         {
             "category": "기타",
             "description": "내용 간략 설명을 재미있게 그리고 디테일하게 번역하여 작성"
         }

    IMPORTANT INSTRUCTIONS:
    - Choose EXACTLY ONE category that best describes the image
    - Use EXACTLY the format shown above for the identified category
    - Use "null" if information is not available
    - Return ONLY the JSON object, no additional text
    - Keep all responses in Korean language
    - If unsure or if the image doesn't clearly fit into the first four categories, classify as "Others"
    """

    answer = call_azure_api(unified_prompt, image)
    answer_json = extract_json_from_string(answer)
    print("answer:", answer)
    print(answer_json, type(answer_json))

    return answer_json

from PIL import Image
import matplotlib.pyplot as plt

image_path = "img/english_bc2.jpg"
image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')  # 축을 숨김
plt.show()
result = thinking2(image_path)

# %%
