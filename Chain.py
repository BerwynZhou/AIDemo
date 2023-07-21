import os
from dotenv import load_dotenv
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain,SequentialChain


load_dotenv()
df = pd.read_csv('data.csv')
openai_api_key = os.environ['OPENAI_API_KEY']

# print(df.head())

llm = ChatOpenAI(temperature=0.7)

# prompt = ChatPromptTemplate.from_template("wahts the avg price of {code}?")

# chain = LLMChain(llm=llm, prompt=prompt)

# code = "sh.600000"

# print(chain.run(code))

# >>>

# prompt_one = ChatPromptTemplate.from_template("translate to chinese the word of {word}")

# chain_one = LLMChain(llm=llm, prompt=prompt_one)

# prompt_two = ChatPromptTemplate.from_template("show me some words that's means something like {word}")

# chain_two = LLMChain(llm=llm, prompt=prompt_two)

# sequential_chain = SimpleSequentialChain(
#     chains=[chain_one, chain_two], verbose=False)

# word = "magnificent"
# print(sequential_chain.run(word))

# >>>
prompt_one = ChatPromptTemplate.from_template("translate to chinese the word of {word}")

chain_one = LLMChain(llm=llm,prompt=prompt_one,output_key="chinese_word")

prompt_two = ChatPromptTemplate.from_template("describe the word in chinese of {chinese_word}")

chain_two = LLMChain(llm=llm,prompt=prompt_two,output_key="chinese_describe")

prompt_three = ChatPromptTemplate.from_template("summary the sentence in english of {chinese_describe}")

chain_three = LLMChain(llm=llm,prompt=prompt_three,output_key="english_summary")

prompt_four = ChatPromptTemplate.from_template("write a follow up response to the following:"
                                               "word:{word}\n"
                                               "chinese:{chinese_word}"
                                               "describe:{chinese_describe}"
                                               "english summary:{english_summary}")

chain_four = LLMChain(llm=llm,prompt=prompt_four) 
word = "magnificent"
sequential_chain=SequentialChain(chains=[chain_one,chain_two,chain_three,chain_four],
                                       input_variables=["word"],
                                       output_variables=["chinese_word","chinese_describe","english_summary"],
                                       verbose=False)
print(sequential_chain(word))
# >>> 
# router chain demo
