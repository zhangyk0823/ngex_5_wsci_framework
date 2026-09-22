from pathlib import Path
from ollama import chat
import json


question = """
I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works.
"""

MODEL = "qwen3:8b"


## WRITE ##
service_status = {
    "wifi": "operational"
}

state = {
    "problem": question,
    "wi_fi status": "operational",
    "wi-fi_check": True
}

with open("state.json", "w") as file:
    json.dump(
        state,
        file,
        indent=2
    )

with open("state.json", "r") as file:
    state = json.load(file)

print("Initial state:", state)


## SELECT CONTEXT FILES BASED ON QUESTION
## Create the function that takes the student's question, takes some keywords and chooses the relevant files from the knowledge base. Return a list of the selected files.
## For example, if the question has the kyeword "print" or "printer", then the function should return the file "knowledge/printer_setup.txt" in a list.
def select_context(question):
    q = question.lower()
    keyword_file_map = {
        ("wifi", "wi-fi", "network", "eduroam", "connect"): "knowledge/wifi_setup.txt",
        ("password", "credentials"): "knowledge/password_changes.txt",
        ("status", "operational", "outage"): "knowledge/service_status.txt",
        ("email",): "knowledge/email_setup.txt",
        ("vpn",): "knowledge/vpn.txt",
        ("print", "printer"): "knowledge/printing.txt",
        ("projector", "display"): "knowledge/classroom_projectors.txt",
    }
    selected = []
    for keywords, path in keyword_file_map.items():
        if any(k in q for k in keywords):
            selected.append(path)
    return selected


selected_files = select_context(question)
print("Selected files:", selected_files)

## READ SELECTED FILES and add their contents to the context variable.
context = ""
for file in selected_files:
    context += Path(file).read_text()
    context += "\n\n"


## 
## COMPRESS CONTEXT
## Add logic to compress the context from above by calling Qwen with "context" and the "question" as the parameter
## The response from Qwen should be the compressed context. Store it in a variable called "compressed_context" 

def compress_context(context, question):
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                    Extract only information relevant to
                    the user's problem.

                    Do not solve the problem.
                    Do not add new information.
                """
            },
            {
                "role": "user",
                "content": f"""
                    USER PROBLEM:

                    {question}

                    DOCUMENT:

                    {context}
                """
            }
        ]
    )

    return response.message.content


compressed_context = compress_context(
    context,
    question
)

## Print the length of the compressed context
print("Compressed context characters:", len(compressed_context))

## Now, call Qwen again with the compressed context and the student's question. Store the response in a variable called "response" and print the response from Qwen.
## Ensure the model produces a structured output 
response = chat(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": """
                You are a university IT support assistant.
                Respond ONLY with valid JSON, no markdown fences, no extra text.
            """
        },
        {
            "role": "user",
            "content": f"""
QUESTION:

{question}

AVAILABLE INFORMATION:

{compressed_context}

Return a JSON object with these fields:
- "diagnosis": short explanation of the root cause
- "recommended_action": the single best fix
- "steps": list of step-by-step instructions
- "confidence": "high" | "medium" | "low"
"""
        }
    ]
)

print(response.message.content)


## WRITE the above output in an artifact called "state"
def extract_json(text):
    """Safely pull the first JSON object out of a model response."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found in model output")
    return json.loads(text[start:end + 1])


answer = extract_json(response.message.content)

## Update the rest of the code so that it uses the "state" artifact as part of the context. 
## It is important to ensure that the model uses only the relevant parts from the "state" artifact and not the entire artifact.
## For this, you may have to think of a good structure for the "state" artifact and how to use it in the context.
state = {
    "problem": question,
    "service_status": service_status,
    "selected_files": selected_files,
    "answer": answer,
}

with open("state.json", "w") as file:
    json.dump(
        state,
        file,
        indent=2
    )

with open("state.json", "r") as file:
    state = json.load(file)

## Only the relevant slice of the state is put into the context.
relevant_state = {
    "problem": state["problem"],
    "answer": state["answer"],
}

state_based_response = chat(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are a university IT support assistant. Answer briefly and directly."
        },
        {
            "role": "user",
            "content": f"""
QUESTION:

{question}

KNOWN STATE (relevant parts only):

{json.dumps(relevant_state, indent=2)}
"""
        }
    ]
)

print("--- Final answer using the state artifact ---")
print(state_based_response.message.content)


## ISOLATE ##
## Create different states/artifacts. Depending on the agent's task,
## the appropriate state is used. Qwen classifies the task type first.
diagnostic_context = {
    "problem": question,
    "device": "Windows laptop",
    "wifi_status": "operational"
}

report_context = {
    "total_wifi_cases": 37,
    "resolved_cases": 29,
    "unresolved_cases": 8
}


def isolate_state(state, question):
    classification = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                    Classify the user request into exactly one category:
                    "diagnostic" (needs troubleshooting / steps to fix a problem)
                    or "report" (needs statistics / case summary).
                    Reply with a single word only.
                """
            },
            {
                "role": "user",
                "content": question
            }
        ]
    ).message.content.strip().lower()

    if "report" in classification:
        return "report", report_context
    return "diagnostic", diagnostic_context


task_type, isolated = isolate_state(state, question)
print("--- ISOLATE ---")
print("Task type:", task_type)
print("Isolated state:", isolated)

isolated_response = chat(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are a university IT support assistant. Use only the given state to answer."
        },
        {
            "role": "user",
            "content": f"""
TASK TYPE: {task_type}

STATE:

{json.dumps(isolated, indent=2)}

Answer the user's request using only this state.
"""
        }
    ]
)

print(isolated_response.message.content)
