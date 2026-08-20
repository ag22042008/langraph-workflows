import os
from typing import TypedDict,Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,Start,END
load_dotenv()
llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1)
def merge_score_dicts(existing: dict,new:dict)->dict:
    if existing is None:
        return new 
    else:
        return{**existing,**new}

#state creation
class Analyzer(TypedDict):
    raw_input : str
    # safety score will be edited by all three nodes that will give u a single score so that s why we use parallel reducers to prevent overriding many times
    safety_score :Annotated[dict[str ,int],merge_score_dicts]
    # safety_score: Annotated[dict[str, int], merge_score_dicts] → this field is a dictionary mapping strings to numbers (like {"toxicity": 2, "spam": 5}). The Annotated[..., merge_score_dicts] part is extra metadata attached to this field, saying: "whenever this field gets updated, use the merge_score_dicts function to combine old and new values instead of just overwriting."

def toxicity_detector(State:Analyzer)->dict:
    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{State['raw_input']}"
    )
    response=llm.invoke(prompt);

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
    return{"safety_score":{'toxic_score':score}}

def cultural_score(State:Analyzer)->dict:
    print("\n[Branch 2] Analyzing Regional & Cultural Sensitivity...")
    prompt=(
       " Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else strictly on the base of text.\n\n"
        f"text_provided:\n{'raw_input'}"
    )
    response=llm.invoke(prompt)
    try:
        score=int(response.content.strip())
    except ValueError:
            score = 0
    return{"safety_score":{'culture_score':score}}

def copywright_node(State:Analyzer)->dict:
    print("[Branch 3] Analyzing Copyright & Originality Risks...")
    prompt=("Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}")
    response=llm.invoke(prompt)
    try:
      score=int(response.content.strip())
    except ValueError:
      score = 0
    return{"safety_score":{'copywright_score':score}}


graph=StateGraph(Analyzer)
graph.add_node("cultural",cultural_score);
graph.add_node("toxcicity",toxicity_detector);
graph.add_node("copywright",copywright_node);

graph.add_edge(Start,"cultural")
graph.add_edge(Start,"toxicity")
graph.add_edge(Start,"copywright")

graph.add_edge("toxicity_node",END)
graph.add_edge("copyright_check",END)
graph.add_edge("culture_node",END)

app=graph.compile()


