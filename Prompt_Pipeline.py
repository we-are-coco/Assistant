# %%
import requests
import json
import os  
import base64
from openai import AzureOpenAI  
import json
import re

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
    AZURE_OPENAI_API_KEY = "your-api-key"

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

image_path = "img/image (3).jpg"
result = thinking(image_path)


# %%


# %%
