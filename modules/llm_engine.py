# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser
import config

class LLMEngine:
    def __init__(self):
        print(f"🤖 Initializing LLM: {config.LLM_MODEL}...")
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model_name=config.LLM_MODEL,
            temperature=0.3
        )
        self.output_parser = StrOutputParser()

    def get_response(self, query: str, context: str):
        """Generates a response based on the provided context."""
        template = """
        You are "CareerPath AI", a professional career and internship assistant for students.
        Your goal is to provide accurate, helpful, and encouraging advice based ONLY on the provided context.
        
        If the answer is not in the context, say: "I'm sorry, I don't have specific information about that in my knowledge base, but I recommend checking official company career pages or university resources."
        
        Context:
        {context}
        
        User Question:
        {query}
        
        Answer professionally in the same language as the question (French or English). Use bullet points for readability if appropriate.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | self.output_parser
        
        return chain.invoke({"context": context, "query": query})
