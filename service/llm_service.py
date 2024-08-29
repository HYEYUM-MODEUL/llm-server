import faiss
import numpy as np
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

class LlmService:
    def __init__(self):
        # Step 1: HuggingFace 모델 로드 & LLaMA 모델 설정
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},  # GPU 대신 CPU 사용
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # self.llm = ChatOllama(model="llama3:latest")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        
        self.dimension = 1024  # BAAI/bge-m3 모델에 맞춘 차원 수
        
        self.modeullak_question_index = faiss.IndexFlatL2(self.dimension)
        self.modeullak_answer_index = faiss.IndexFlatL2(self.dimension)
        self.modeullak_dialogue_ids = list()
        
        self.hyeyumteo_question_index = faiss.IndexFlatL2(self.dimension)
        self.hyeyumteo_answer_index = faiss.IndexFlatL2(self.dimension)
        self.hyeyumteo_dialogue_ids = list()
        
        self.save_directory = os.path.join(os.getcwd(), '.data')
        
    def ready(self):
        os.makedirs(self.save_directory, exist_ok=True)
        
        if os.path.exists(os.path.join(os.getcwd(), '.data', 'modeullak_question_faiss_index.bin')):
            self.modeullak_question_index = faiss.read_index(os.path.join(os.getcwd(), '.data', 'modeullak_question_faiss_index.bin'))
        if os.path.exists(os.path.join(os.getcwd(), '.data', 'modeullak_answer_faiss_index.bin')):
            self.modeullak_answer_index = faiss.read_index(os.path.join(os.getcwd(), '.data', 'modeullak_answer_faiss_index.bin'))

        if os.path.exists(os.path.join(os.getcwd(), '.data', 'hyeyumteo_question_faiss_index.bin')):
            self.hyeyumteo_question_index = faiss.read_index(os.path.join(os.getcwd(), '.data', 'hyeyumteo_question_faiss_index.bin'))
        if os.path.exists(os.path.join(os.getcwd(), '.data', 'hyeyumteo_answer_faiss_index.bin')):
            self.hyeyumteo_answer_index = faiss.read_index(os.path.join(os.getcwd(), '.data', 'hyeyumteo_answer_faiss_index.bin'))

        self.modeullak_answer_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """
                "You are a helpful, professional assistant named HyeyumModeul AI. Introduce yourself first, and answer the questions."
                """
            ),
            SystemMessagePromptTemplate.from_template(
                """
                "answer me in Korean no matter what. "
                """
            ),
            SystemMessagePromptTemplate.from_template(
                """
                you should respond similarly to the given example response.

                Example response to a similar user question: {context}

                Based on the above example, analyze the user's code and provide an answer to the question.

                Additionally, explain the code block in detail and provide examples to clarify.

                And just give the answer. And Korean only.
                """
            ),
            HumanMessagePromptTemplate.from_template(
                """
                Do not vectorize, and provide the answer in Korean form only.
                
                User's Full Code: {long_code}

                User's Partial Code: {short_code}

                User's Question: {question_content}
                """
            )
        ])
        
        self.hyeyumteo_answer_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """
                "You are a helpful, professional assistant named HyeyumModeul AI. Introduce yourself first, and answer the questions."
                """
            ),
            SystemMessagePromptTemplate.from_template(
                """
                "answer me in Korean no matter what. "
                """
            ),
            SystemMessagePromptTemplate.from_template(
                """
                you should respond similarly to the given example response.

                Example response to a similar user question: {context}

                And just give the answer. And Korean only.
                """
            ),
            HumanMessagePromptTemplate.from_template(
                """
                Do not vectorize, and provide the answer in Korean form only.
                
                User's Question: {question}
                """
            )
        ])
    
        self.keyword_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """
                You need to analyze the user's full code, partial code, and question content to extract the most important keyword, and use it to generate an answer.
                """
            ),
            SystemMessagePromptTemplate.from_template(
                """
                Specifically, you should only provide the keyword as the answer. Do not provide additional explanations, examples, etc.

                For example, if you extract the keyword 'assignment statement', then only provide the word 'assignment statement' as the answer.
                """
            ),
            SystemMessagePromptTemplate.from_template(
                """
                Finally, the answer should consist of just one word.
                """
            ),
            SystemMessagePromptTemplate.from_template(
                """
                The most important point is that you must respond in Korean.
                """
            ),
            HumanMessagePromptTemplate.from_template(
                """
                User's full code: {long_code}

                User's partial code: {short_code}

                User's question: {question_content}
                """
            )
        ])
        
        self.summarize_keyword_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """
                "You are an AI specialized in summarization."
                """
            ),
            SystemMessagePromptTemplate.from_template(
                """
                "You should respond only in Korean."
                """
            ),
            SystemMessagePromptTemplate.from_template(
                """
                "Analyze the user's previous questions and write a description for the relevant keyword. However, keep it under 200 characters."
                """
            ),
            HumanMessagePromptTemplate.from_template(
                """
                Do not vectorize, and provide the answer in Korean form only.

                Keyword name: {name}

                Questions related to the keyword: {questionsStr}
                """
            )
        ])

        self.proofread_stt_text_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """
                Refine the questions entered by the user and provide clear answers based on the refined questions.
                """
            ),
            SystemMessagePromptTemplate.from_template(
                """
                All you need to do is correct the response. Don't do any additional work.
                """
            ),
            HumanMessagePromptTemplate.from_template(
                """
                Do not vectorize, and provide the answer in Korean form only.
                
                Professor's original question: {question}

                Professor's original answer: {stt_answer}
                """
            )
        ])
    
    def save_modeullak_dialogue(self, dialogue_id: int, question: str, answer: str):
        
        self.modeullak_dialogue_ids.append(dialogue_id)
        
        answer_vector = self.embedding_model.embed_documents([answer])[0]
        answer_vector = np.array([answer_vector], dtype=np.float32)  # float32로 변환
        
        self.modeullak_answer_index.add(answer_vector)  # 답변 인덱스에 질문 벡터 추가

        # 질문의 인덱스를 검색하여 해당 위치에 답변 벡터 결합
        question_vector = self.embedding_model.embed_documents([question])[0]
        question_vector = np.array([question_vector], dtype=np.float32)
        
        self.modeullak_question_index.add(question_vector)  # 질문 인덱스에 답변 벡터 추가

        # 인덱스 저장
        question_filepath = os.path.join(self.save_directory, "modeullak_question_faiss_index.bin")
        answer_filepath = os.path.join(self.save_directory, "modeullak_answer_faiss_index.bin")
        
        faiss.write_index(self.modeullak_question_index, question_filepath)
        faiss.write_index(self.modeullak_answer_index, answer_filepath)

    def save_hyeyumteo_dialogue(self, dialogue_id: int, question: str, answer: str):
        
        self.hyeyumteo_dialogue_ids.append(dialogue_id)
        
        answer_vector = self.embedding_model.embed_documents([answer])[0]
        answer_vector = np.array([answer_vector], dtype=np.float32)  # float32로 변환
        
        self.hyeyumteo_answer_index.add(answer_vector)  # 답변 인덱스에 질문 벡터 추가

        # 질문의 인덱스를 검색하여 해당 위치에 답변 벡터 결합
        question_vector = self.embedding_model.embed_documents([question])[0]
        question_vector = np.array([question_vector], dtype=np.float32)
        
        self.hyeyumteo_question_index.add(question_vector)  # 질문 인덱스에 답변 벡터 추가

        # 인덱스 저장
        question_filepath = os.path.join(self.save_directory, "hyeyumteo_question_faiss_index.bin")
        answer_filepath = os.path.join(self.save_directory, "hyeyumteo_answer_faiss_index.bin")
        
        faiss.write_index(self.hyeyumteo_question_index, question_filepath)
        faiss.write_index(self.hyeyumteo_answer_index, answer_filepath)
    
    def generate_answer_in_dialogue_of_modeullak(self, short_code: str, long_code: str, question_content: str):
        # Question 벡터화
        question_vector = self.embedding_model.embed_documents([question_content])[0]
        question_vector = np.array([question_vector], dtype=np.float32)  # float32로 변환
        
        if self.modeullak_question_index.ntotal > 0:
            D_q, I_q = self.modeullak_question_index.search(question_vector, 1)  # 가장 유사한 질문 벡터 찾기

            if D_q[0][0] <= 0.2:
                closest_index = int(I_q[0][0])
                
                dialogue_id = self.modeullak_dialogue_ids[closest_index]
                closest_answer_vector = self.modeullak_answer_index.reconstruct(closest_index)  
                
                chain = self.modeullak_answer_prompt | self.llm | StrOutputParser()
                
                response = chain.invoke({
                    "context": closest_answer_vector,
                    "long_code": long_code,
                    "short_code": short_code,
                    "question_content": question_content
                })
                
                return {
                    "dialogue_id": dialogue_id,
                    "answer": response
                }
        else:
            return None

    def generate_answer_in_dialogue_of_hyeyumteo(self, question: str):
        # Question 벡터화
        question_vector = self.embedding_model.embed_documents([question])[0]
        question_vector = np.array([question_vector], dtype=np.float32)  # float32로 변환
        
        if self.hyeyumteo_question_index.ntotal > 0:
            D_q, I_q = self.hyeyumteo_question_index.search(question_vector, 1)  # 가장 유사한 질문 벡터 찾기

            if D_q[0][0] <= 0.2:
                closest_index = int(I_q[0][0])
                
                closest_answer_vector = self.hyeyumteo_answer_index.reconstruct(closest_index)  
                
                chain = self.hyeyumteo_answer_prompt | self.llm | StrOutputParser()
                
                response = chain.invoke({
                    "context": closest_answer_vector,
                    "question": question
                })
                
                return {
                    "answer": response
                }
        else:
            return None

    def generate_keyword_by_question_in_dialogue_of_modeullak(self, short_code: str, long_code: str, question_content: str) -> str:
        chain = self.keyword_prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "long_code": long_code,
            "short_code": short_code,
            "question_content": question_content
        })
        
        return response
    
    def summarize_keyword_in_modeullak_using_dialogues(self, keywords: list) -> list:
        result: list = list()
        
        chain = self.summarize_keyword_prompt | self.llm | StrOutputParser()
        
        for keyword in keywords:
            name = keyword.get('name')
            questions = keyword.get('questions')
            
            if len(questions) > 2:
                questions = np.random.choice(questions, 2, replace=False)
                
            questionsStr = "\n\n\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            
            response = chain.invoke({
                "name": name,
                "questionsStr": questionsStr
            })
            
            result.append({
                "name": name,
                "description": response
            })
        
        return result
    
    def proofread_stt_text(self, question: str, stt_answer: str) -> str:
        chain = self.proofread_stt_text_prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "question": question,
            "stt_answer": stt_answer
        })
        
        return "해당 답변은 HyeyumModeul AI에 의해 교정되었습니다.\n" + response
    
    def clear_data(self):
        os.system(f"rm -rf .data")
        os.makedirs('.data', exist_ok=True)