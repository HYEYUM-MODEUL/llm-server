import json

from kafka import KafkaConsumer, KafkaProducer

from app.config import Config
from service.llm_service import LlmService

class ModeullakListner:
    
    __topic: str = 'modeullak'
    
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
                if message_type == 'NOT_SUMMATION':
                    self.__process_summation(data)
                else:
                    continue
            except Exception as e:
                print(e)

    def __process_summation(self, data):
        print("Modeullak Summation Processing...")
        
        modeullak_id = data.get('modeullak_id')
        keywords = data.get('keywords')

        if modeullak_id and keywords:
            response = self.__llm_service.summarize_keyword_in_modeullak_using_dialogues(
                keywords=keywords,
            )
            
            self.__producer.send(self.__topic, value={
                'type': 'SUMMATION',
                'modeullak_id': modeullak_id,
                'keywords': response
            })
            
        print("Modeullak Summation Processed.")

    def close(self):
        self.__consumer.close()