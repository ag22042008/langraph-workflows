import os
from typing import TypedDict,Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
load_dotenv()
llm=ChatGroq(model="openai/gpt-oss-120b",temperature=0.1)
#reducers 
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
        f"text_provided:\n{State['raw_input']}"
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
         f"text_provided:\n{State['raw_input']}"
        )
    response=llm.invoke(prompt)
    try:
      score=int(response.content.strip())
    except ValueError:
      score = 0
    return{"safety_score":{'copywright_score':score}}


graph=StateGraph(Analyzer)
graph.add_node("cultural",cultural_score);
graph.add_node("toxicity",toxicity_detector);
graph.add_node("copywright",copywright_node);

graph.add_edge(START,"cultural")
graph.add_edge(START,"toxicity")
graph.add_edge(START,"copywright")

graph.add_edge("toxicity",END)
graph.add_edge("copywright",END)
graph.add_edge("cultural",END)

app=graph.compile()
sample_script="""
 Honestly, trying to do business in that part of the world is a complete joke. 
        Their entire culture is just backward and lazy compared to modern societies. 
        Their traditional foods are disgusting, and their local customs make absolutely 
        no sense for anyone trying to run a serious, civilized enterprise.


"""
initial_state={
    "raw_input":sample_script,
    "safety_score":{}#Intialized as an empty dictonary
}
# as we are invoking states dict it will also be giving out final state dict so ascess it using final state dict
final_state=app.invoke(initial_state)
print(final_state["safety_score"])



