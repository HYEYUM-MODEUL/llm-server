import json

from kafka import KafkaConsumer, KafkaProducer

from app.config import Config
from service.llm_service import LlmService

class ModeullakDialogueListner:
    
    __topic: str = 'modeullak-dialogue'
    
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
                if message_type == 'DIALOGUE':
                    self.__process_dialogue(data)
                elif message_type == 'QUESTION':
                    self.__process_question(data)
                else:
                    continue
            except Exception as e:
                print(e)

    def __process_dialogue(self, data):
        print("Processing Saving Modeullak Dialogue...")

        dialogue_id = data.get('dialogue_id')
        question = data.get("question")
        answer = data.get("answer")

        if dialogue_id and question and answer:
            self.__llm_service.save_modeullak_dialogue(dialogue_id, question, answer)

        print("Saving Modeullak Dialogue Processed")

    def __process_question(self, data):
        print("Processing Question In Dialogue Of Modeullak...")
        request_dialogue_id = data.get('dialogue_id')
        
        question_short_code = data.get('question_short_code')
        question_long_code = data.get('question_long_code')
        question_content = data.get('question_content')
        
        if request_dialogue_id and question_long_code and question_short_code and question_content:
            answer = None
            keyword = None
            similar_dialogue_id = None
            
            response = self.__llm_service.generate_answer_in_dialogue_of_modeullak(question_short_code, question_long_code, question_content)
            
            if response:
                similar_dialogue_id = response.get('dialogue_id')
                answer = response.get('answer')
            else:
                keyword = self.__llm_service.generate_keyword_by_question_in_dialogue_of_modeullak(question_short_code, question_long_code, question_content)

            self.__producer.send(self.__topic, value={
                'type': 'ANSWER',
                'request_dialogue_id': request_dialogue_id,
                'similar_dialogue_id': similar_dialogue_id,
                'answer': answer,
                'keyword': keyword
            })
        print("Question In Dialogue Of Modeullak Processed")

    def close(self):
        self.__consumer.close()