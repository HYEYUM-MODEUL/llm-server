import json
import threading

from fastapi import FastAPI
from kafka import KafkaProducer

from app.config import Config

from service.llm_service import LlmService

from listner.modeullak_dialogue_listner import ModeullakDialogueListner
from listner.hyeyumteo_dialogue_listner import HyeyumteoDialogueListner
from listner.modeullak_listner import ModeullakListner
from listner.stt_text_dialogue_listner import SttTextDialogueListner

config = Config()
config.ready()

llm_service = LlmService()
llm_service.ready()

producer = KafkaProducer(
    bootstrap_servers=config.brokers,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

modeullak_dialogue_listner = ModeullakDialogueListner(config, producer, llm_service)
hyeyumteo_dialogue_listner = HyeyumteoDialogueListner(config, producer, llm_service)
stt_text_dialogue_listner = SttTextDialogueListner(config, producer, llm_service)
modeullak_listner = ModeullakListner(config, producer, llm_service)

# 실행시 Kafka 소비자가 백그라운드에서 계속 실행되도록 함
async def lifespan(app: FastAPI):
    print("Starting Kafka consumers...")
    threading.Thread(target=modeullak_dialogue_listner.consume, daemon=True).start()
    threading.Thread(target=hyeyumteo_dialogue_listner.consume, daemon=True).start()
    threading.Thread(target=modeullak_listner.consume, daemon=True).start()
    threading.Thread(target=stt_text_dialogue_listner.consume, daemon=True).start()
    yield
    print("Shutting down Kafka consumers...")
    modeullak_dialogue_listner.close()
    hyeyumteo_dialogue_listner.close()
    modeullak_listner.close()
    stt_text_dialogue_listner.close()

    llm_service.clear_data()

# FastAPI 애플리케이션 생성
app = FastAPI(lifespan=lifespan)