import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from pathlib import Path

# STT 모듈 (음성 → 텍스트 변환)
class STTModule:
    def __init__(self):
        # .env 파일에서 키 불러오기
        env_path = Path(__file__).parent / '.env'
        load_dotenv(env_path)
        
        subscription_key = os.getenv('STT_API_KEY')
        if not subscription_key:
            raise ValueError("STT_API_KEY not found in .env file")

        self.speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key, 
            region='eastus'
        )
        self.speech_config.speech_recognition_language = "ko-KR"
    
    def recognize_speech(self):
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, 
            audio_config=audio_config
        )
        
        print("검색할 내용을 말해주세요.")
        result = speech_recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"인식된 텍스트: {result.text}")
            return result.text
        else:
            print("음성을 인식하지 못했습니다.")
            return None


# 키워드 추출 모듈 (텍스트 → 키워드)
class KeyPhraseExtraction:
    def __init__(self):
        env_path = Path(__file__).parent / '.env'
        load_dotenv(env_path)
        self.endpoint = "https://team2language.cognitiveservices.azure.com/"
        self.key = os.getenv('KEYWORD_API_KEY')
        self.client = TextAnalyticsClient(
            endpoint=self.endpoint, 
            credential=AzureKeyCredential(self.key)
        )

    def extract_keywords(self, text):
        if not text:
            print("입력된 텍스트가 없습니다.")
            return []
        
        response = self.client.extract_key_phrases([text], language="ko")
        for document in response:
            print(f"추출된 키워드: {document.key_phrases}")
            return document.key_phrases

def main():
    stt_module = STTModule()
    keyphrase_extractor = KeyPhraseExtraction()

    # STT 실행 (음성 → 텍스트 변환)
    recognized_text = stt_module.recognize_speech()

    # 키워드 추출 실행
    if recognized_text:
        keywords = keyphrase_extractor.extract_keywords(recognized_text)
        print(f"최종 키워드 리스트: {keywords}")
    else:
        print("음성을 인식하지 못했습니다.")
    return keywords

# 전체 실행 흐름
if __name__ == "__main__":
    main()