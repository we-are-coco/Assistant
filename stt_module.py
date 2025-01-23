# %%
import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from pathlib import Path

class STTModule:
    def __init__(self):
        # .env 파일의 정확한 경로 설정
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
            print("인식된 텍스트: {}".format(result.text))
            return result.text
        return None

if __name__ == "__main__":
    stt_module = STTModule()
    recognized_text = stt_module.recognize_speech()
# %%
