import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

'''
 ConversationBufferMemory 全量
 ConversationBufferWindowMemory（k=2） 最近K论对话
 ConversationTokenBufferMemory(llm=llm,max_token_limit=50) 最近的50token对话
 ConversationSummaryBufferMemory(llm=llm,max_token_limit=50) 存最近的50token对话，较早的部分交给语言模型生成摘要
'''
load_dotenv()

llm = ChatOpenAI(temperature=0.0)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)
# 存对话
memory.save_context({"input": "Hi"}, {"output": "What's up"})
# 读对话
memory.load_memory_variables({})
# 与gpt对话
print(conversation.predict(input="Hi,my name is Bowen"))
print(conversation.predict(input="waht's the answer about 1 plus 1"))
print(conversation.predict(input="what's my name"))
