import os 
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#STEP 1 -> BUILDING RETRIVERS
def build_retrievers(pdf_path:str):
    loader=PyPDFLoader(pdf_path)
    docs=loader.load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=140)
    chunks=splitter.split_documents(docs)
    vector_store=FAISS.from_documents(chunks,embedding)
    return vector_store.as_retriever(search_type='mmr',
        search_kwargs={"k":4,"fetch_k":15,"lambda_mult":0.25},)

academic_retriever=build_retrievers(r"C:\Users\coone\OneDrive\Desktop\Langraph Workflows Sequential flow\Rag_project\academics_handbook.pdf")
fee_retriver=build_retrievers(r"C:\Users\coone\OneDrive\Desktop\Langraph Workflows Sequential flow\Rag_project\fee_structure.pdf")
llm=ChatGroq(model="openai/gpt-oss-120b",temperature=0.3)

#step2 State
class State(TypedDict):
    programme:str #type of programme inrolled in
    messages:Annotated[list,add_messages]#saving a history of the context like it will be a 
    classify:str #classifying the query whether it is academic personal or fees related
    retrieved_context:str # THE NODES RETRIEVED CONTEXT WETHER IT IS FROM FEE ,ACADEMICS AND ALSO PERSONAL CONTEXT A CONDITIONAL CONTEXT

#STEPS 3 NODES GENERATION
# classsification using llm for the node and router function 
def classifier_node(state:State)->dict:
    """Look at the following query and classify it to the context which path to take and which mode to select"""
    last_message=state["messages"][-1].content #messages will be like this ai message human message and then again responded ai message so before query is processed there will be a human message only
    prompt=(  "Classify the following student query into exactly one category: "
    "'academic', 'fee', or 'general'.\n\n"

    "IMPORTANT: Classify based on the TYPE OF INFORMATION the student is asking for, "
    "not simply because the topic is related to education.\n\n"

    "Use 'academic' ONLY for questions about the college's academic rules, "
    "policies, requirements, or programme-specific information, such as "
    "attendance requirements, exams, grading, credits, promotion, course structure, "
    "summer training requirements, degree requirements, semester rules, or "
    "college academic policies.\n\n"

    "Use 'fee' for questions about tuition, payments, refunds, late charges, "
    "scholarships, fines, fee deadlines, or any college-related money matter.\n\n"

    "Use 'general' for greetings, casual conversation, and general knowledge "
    "questions that can be answered without the college documents. This includes "
    "general educational or technical concepts such as DBMS, Operating Systems, "
    "Computer Networks, OOP, programming, AI, machine learning, mathematics, etc.\n\n"

    "IMPORTANT EXAMPLES:\n"
    "What is DBMS? -> general\n"
    "What is Computer Networks? -> general\n"
    "What is OOP? -> general\n"
    "What is an Operating System? -> general\n"
    "How many semesters are there in BTech? -> academic\n"
    "What is the attendance requirement? -> academic\n"
    "How many credits do I need to graduate? -> academic\n"
    "How much is the tuition fee? -> fee\n"
    "What is the late payment fee? -> fee\n\n"

    f"Query: {last_message}\n\n"

    "Return ONLY one word: academic, fee, or general.")
    response=llm.invoke(prompt)
    category=response.content.strip().lower()
    if "academic" in category:
        category="academic"
    elif "fee" in category:
        category="fee"
    else:
        category="general"
    return{"classify":category}

def academic_rag_node(state:State)->dict:
    """Retrieves relevant chunks from the academics handbook."""
    query=state["messages"][-1].content
    docs=academic_retriever.invoke(query)
    context="\n\n".join([doc.page_content for doc in docs])
    return{"retrieved_context":context}

def fees_rag_node(state:State)->dict:
    """Retrieves relevant chunks from the fees handbook."""
    query=state["messages"][-1].content
    docs=fee_retriver.invoke(query)
    context="\n\n".join([doc.page_content for doc in docs])
    return{"retrieved_context":context}


def genral_node(state:State)->dict:
    "Answers directly using the llm's own knowldge ,no retrieval needed"
    return{"retrieved_context":"No retrieval required"}


# response node the retrieved context is sent to llm
def response_node(state:State)->dict:
    "Generates the personalized answer , personalized using the student's programme"
    query=state["messages"][-1].content
    programme=state.get("programme","Unkown")# if there is no programme satisfied that the programmme will be unkown
    context=state["retrieved_context"]
    if context=="No retrieval required":
        prompt = f"""
You are a friendly college assistant.

The student is enrolled in: {programme}.

Answer the user's question accurately using your general knowledge.

Important rules:
- Answer the actual question asked by the student.
- Do NOT assume the student belongs to BCA or any other programme.
- Never mention another programme unless the user explicitly asks about it.
- Mention "{programme}" only when it is genuinely relevant.
- For general educational concepts like DBMS, Computer Networks, OS, OOP, AI, etc., explain them neutrally.
- Do not add irrelevant information about the student's programme.
- Do not ask the student to provide another question if the current question is clear."""
    else:
         prompt = (f"""
            You are a college assistant helping a {programme} student.

Use the following official college document context to answer the question.

Rules:
- The student's programme is exactly {programme}.
- Do not assume information belonging to another programme applies to this student.
- If the required information is not present in the context, say that it is not
  available in the provided college documents.
- Do not invent college-specific information .

Context:
{context}

Question:
{query}

Give a clear and precise answer.
     """   )
    response=llm.invoke(prompt)
    return{"messages":[("ai",response.content.strip())]}

# step4-router function it can use llm based classify or it can use pure if else classification as by user
# NAME OF NODES MUST ALSO BE SAME IN THE GRAPH
def route_query(state:State)->dict:
    if state["classify"]=="academic":
        return "academic_response"
    elif state["classify"]=="fee":
        return "fee_response" 
    else:
        return "general"

#Step 5 - building the graph

graph=StateGraph(State)

# adding nodes
graph.add_node("classifier",classifier_node)
graph.add_node("academic_response",academic_rag_node)
graph.add_node("fee_response",fees_rag_node)
graph.add_node("general",genral_node)
graph.add_node("response",response_node)


# Step5 edges adding

graph.add_edge(START,"classifier")

# ADDING CONDITIONAL EDGES USING ROUTER FUNCTIONS
graph.add_conditional_edges("classifier",route_query)

graph.add_edge("academic_response","response")
graph.add_edge("fee_response","response")
graph.add_edge("general","response")
graph.add_edge("response",END)
app=graph.compile()

#step 6 run the code
print("="*50+"Welcome to the College Assistant"+"="*50)
print("which programme are you im")
student_programme = input("Enter Your programme-> ").strip() or "not specified"

print(f"\nGreat! You're set as a {student_programme} student.")

while True:
    user_query=input("Your query:")

    if user_query.lower() in ["exit","quit"]:
        break
    result=app.invoke({
        "programme":student_programme,
        "messages":[("human",user_query)]
    })
    print(f"Assistant Reply:{result['messages'][-1].content}")



