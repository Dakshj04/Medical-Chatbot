from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Define the system and human message templates
system_prompt = """You are a Medical assistant for question-answering tasks.
Use the information provided in the retrieved context to answer. 
If you dont know the answer or it is not present in the context, respond with "I don't know."
Keep answers clear, concise, and no more than three sentences.
Cite the source from the context if relevant.

Context: {context}"""

human_template = "{input}"

# Create the chat prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template(human_template)
])