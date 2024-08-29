import json

from kafka import KafkaConsumer, KafkaProducer

from app.config import Config
from service.llm_service import LlmService

class SttTextDialogueListner:
    
    __topic: str = 'stt-text'
    
    def __init__(self, config: Config, producer: KafkaProducer, llm_service: LlmService):
        self.__producer = producer
        self.__llm_service = llm_service
        
        self.__consumer = KafkaConsumer(
            self.__topic,
            group_id='llm-server',
            enable_auto_commit=True,
            auto_offset_reset='earliest',
            bootstrap_servers=config.brokers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        

    def consume(self):
        for message in self.__consumer:
            try:
                # Refine Type In Message
                data = message.value
                message_type = data.get('type')
                
                # Process Message By Type
                if message_type == 'NOT_CORRECTION':
                    self.__process_stt_text(data)
                else:
                    continue
            except Exception as e:
                print(e)

    def __process_stt_text(self, data):
        print("Proofread STT Text Process Started...")
        
        dialogue_id = data.get('dialogue_id')
        question = data.get("question")
        stt_answer = data.get("stt_answer")

        if dialogue_id and question and stt_answer:
            response = self.__llm_service.proofread_stt_text(
                question=question,
                stt_answer=stt_answer
            )
            
            self.__producer.send(self.__topic, value={
                'type': 'CONRRECTION',
                'dialogue_id': dialogue_id,
                'proofreaded_answer': response
            })

        print("Proofread STT Text Processed Successfully!")

    def close(self):
        self.__consumer.close()