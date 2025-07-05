import praw
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import os
from datetime import datetime
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from datetime import datetime
from flask import make_response
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_core.tools import tool
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage 
from langchain_core.messages import ToolMessage 
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END,START
from langgraph.prebuilt import ToolNode
from pymongo import MongoClient
from langchain.schema import SystemMessage, HumanMessage,AIMessage
from transformers import pipeline
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
import torch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# Load once
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
from langchain.schema.runnable import RunnableBranch, RunnableLambda

# Initialize the Ollama model (you can use llama3, mistral, etc.)
load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0.2
)
#model = Ollama(model="llama3.2")
summarizer = pipeline("summarization", model="t5-small")
app = Flask(__name__)
CORS(app, supports_credentials=True)
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["social_db"]
user_sessions = db["user_sessions"]
users_collection = db["users"]
"""model = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # Replace with the model you want
    task="text-generation",
    temperature=0.2,
    max_new_tokens=512
)"""
"""model = ChatGoogleGenerativeAI(
    model="gemini-1.5",  # or "gemini-1.5-pro", depending on what you want
    temperature=0.9
)"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
#print(os.environ.get("OPENAI_API_KEY"))
#model= OpenAI()

reddit = praw.Reddit(
    client_id='RAXrbO7u_KzmyvumzVEgyg',
    client_secret='3leOhBq9_-S5gJNfaGHFDwG7AmZJHQ',
    username='EfficiencyNervous911',
    password='#pqop3699',
    user_agent='script:Random_app (by /u/EfficiencyNervous911'
)

def save_session_history(session_id: str, new_messages: list):
    """
    Appends new messages to the session history in the user_sessions collection.
    If the session doesn't exist, creates it.

    Args:
        session_id (str): Unique identifier for the user's session.
        new_messages (list): List of message dicts or BaseMessage objects.
    """
    # Convert messages to dicts if they are Langchain BaseMessage objects
    serialized_msgs = [
        msg.dict() if hasattr(msg, "dict") else msg for msg in new_messages
    ]

    user_sessions.update_one(
        {"session_id": session_id},
        {
            "$push": {"messages": {"$each": serialized_msgs}},
            "$setOnInsert": {"created_at": datetime.utcnow()}
        },
        upsert=True
    )

def get_session_history(session_id: str):
    session = user_sessions.find_one({"session_id": session_id})
    history = ChatMessageHistory()
    if session and "messages" in session:
        for msg in session["messages"]:
            msg_obj = BaseMessage.parse_obj(msg)
            history.add_message(msg_obj)
    return history


def generate_full_summary(messages):
    full_text = " ".join(msg.content for msg in messages if hasattr(msg, 'content'))

    if len(full_text) > 4000:
        full_text = full_text[:4000]  # To avoid too large input

    num_words = len(full_text.split())

    if num_words < 20:
        # Very small conversation, don't even summarize, return as it is
        return full_text

    max_length = min(100, int(num_words * 0.7))  # 70% of words

    # Never let it be too tiny
    max_length = max(max_length, 10)

    summary = summarizer(
        full_text,
        max_length=max_length,
        min_length=max(5, int(max_length * 0.5)),
        do_sample=False
    )
    return summary[0]['summary_text']

def update_session_summary(session_id, new_summary):
    user_sessions.update_one(
        {"session_id": session_id},
        {"$set": {"summary": new_summary}}
    )

def maybe_update_summary(session_id, all_messages):
    if len(all_messages)%10==0:  # 10 user+bot pairs
        new_summary = generate_full_summary(all_messages)
        update_session_summary(session_id, new_summary)

def get_session_summary(session_id):
    """
    Fetch the saved conversation summary from MongoDB for the given session ID.
    """
    session_data = user_sessions.find_one({"session_id": session_id})
    if session_data and "summary" in session_data:
        return session_data["summary"]
    return None


@tool
def make_reddit_post(msg: str, content: str) -> str:
    """make a post in reddit, you have to generate the message msg, and content yourself"""
    try:
        subreddit = reddit.subreddit("Random_community_3")
        submission = subreddit.submit(title=msg, selftext=content)
        return f"✅ Post created: {submission.url}"
    except Exception as e:
        return f"❌ Failed to post to Reddit: {str(e)}"
    
@tool
def make_post_in_vibenet(msg: str, content: str) -> str:
    """
    make a  post in vibenet.
    you have to generate the message msg, and content yourself
    """
    try:
        post = {
            "title": msg,
            "message": content,
            "timestamp": datetime.utcnow(),
            "status": "pending"
        }
        db["social_media"].insert_one(post)
        return "✅ Posted successfully, check on http://localhost:3000/posts "
    except Exception as e:
        return f"❌ Failed to save to database: {str(e)}"

@tool
def set_reminder(user: str, title: str, time: str, note: str = "") -> str:
    """
    Save a reminder for a user 
    - time format: ISO string (e.g., '2025-06-22T22:00:00')
    if any of the details not provided you only assume any random
    """
    try:
        reminder = {
            "user": user,
            "title": title,
            "note": note,
            "time": datetime.fromisoformat(time),
            "created_at": datetime.utcnow(),
            "status": "active"
        }
        db["reminders"].insert_one(reminder)
        return f"✅ Reminder set for {title} at {time}!"
    except Exception as e:
        return f"❌ Failed to set reminder: {str(e)}"

@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello {name}, I'm Social media bot! 🌸 I'm here to help you regarding social media posts."
    
tools=[make_reddit_post,greet,make_post_in_vibenet,set_reminder]   
tool_map = {tool.name: tool for tool in tools}
def create_system_prompt(user_details):
    if user_details:
        # Dynamically customize system prompt with user info
        user_name = user_details.get("name", "User")
        
        # Create a string-based system prompt template
        system_prompt_str = f"""
        You are a helpful and creative AI content coach specialized in generating social media content. 
        Your tasks include writing captions, adapting tone, understanding inputs like topic/intent/audience, 
        suggesting post formats, and keeping content concise, trendy, and platform-specific.
        User details:
        Name: {user_name}
        """
        
        # Now, create a SystemMessagePromptTemplate with the string
        system_prompt = SystemMessagePromptTemplate.from_template(system_prompt_str)
        return system_prompt
    return None   
system_template = SystemMessagePromptTemplate.from_template(
    "You are a helpful and creative AI content coach specialized in generating social media content.\n" 
    "Your tasks include writing captions, adapting tone, understanding inputs like topic/intent/audience,suggesting post formats, and keeping content concise, trendy, and platform-specific.\n"
    "User details:\n"
    "Name: {user_name}\n"
)

# 2️⃣ Define your human prompt template (summary + recent messages + input)
human_template = HumanMessagePromptTemplate.from_template(
    "Summary of previous conversation:\n"
    "{summary}\n\n"
    "Recent conversation:\n"
    "{recent_messages}\n\n"
    "User’s latest question:\n"
    "{input}"
)
#################### Graph_creation #####################################################################
def prompt_with_tool_call(response):
    for call in response.tool_calls:
        tool_name = call["name"]
        args = call["args"]
        tool_result = tool_map[tool_name].invoke(args)
    return str(tool_result)

branches = RunnableBranch(
    (
        lambda x: hasattr(x, "tool_calls") and x.tool_calls,
        RunnableLambda(lambda x: prompt_with_tool_call(x))
    ),
    RunnableLambda(lambda x: x)
)
llm =model.bind_tools(tools)
chain = llm
chain2 = chain | branches
########################################################


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if users_collection.find_one({"email": email}):
        return jsonify({"message": "User already exists!"}), 409  # Conflict

    hashed_pw = generate_password_hash(password)

    user_data = {
        "name": data.get("name"),
        "email": email,
        "password": hashed_pw
    }

    users_collection.insert_one(user_data)

    # Create a response with cookie
    response = make_response(jsonify({"message": "User registered successfully!"}))
    response.set_cookie("user_email", email,httponly=False,samesite="Lax",secure=False)
    response.set_cookie("user_name", user_data["name"],httponly=False,samesite="Lax",secure=False)

    return response, 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = users_collection.find_one({"email": email})
    if check_password_hash(user["password"], password):
        response = make_response(jsonify({"message": "Login successful!"}))
        response.set_cookie("user_email", email, httponly=False, samesite="Lax", secure=False)
        response.set_cookie("user_name", user.get("name", ""), httponly=False, samesite="Lax", secure=False)
        return response, 200

    return jsonify({"message": "Invalid credentials"}), 401



@app.route("/logout")
def logout():
    resp = make_response(jsonify({"message": "Logged out"}))
    # Clear all user-related cookies
    resp.set_cookie("user_id", "", expires=0)
    resp.set_cookie("user_email", "", expires=0)
    resp.set_cookie("user_name", "", expires=0)
    return resp, 200
chat_prompt = ChatPromptTemplate.from_messages([
    system_template,
    human_template
])
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    session_id = request.cookies.get("user_email")
    if not session_id:
        return jsonify({"error": "No session ID in cookies"}), 401

    user = users_collection.find_one({"email": session_id})
    if not user:
        return jsonify({"error": "User not found"}), 404

    name = user.get("name", "User")
    history = get_session_history(session_id).messages
    maybe_update_summary(session_id, history)
    saved_summary = get_session_summary(session_id) or "No prior summary."
    recent_msgs = history[-10:]
    recent_text = "\n".join(
        f"{'User' if m.type == 'human' else 'AI'}: {m.content}"
        for m in recent_msgs
    )

    # Format prompt
    pv = chat_prompt.format_prompt(
        user_name=name,
        summary=saved_summary,
        recent_messages=recent_text,
        input=user_input
    )
    messages = pv.to_messages()

    # Get LLM response
    llm_response = chain.invoke(messages)

    if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
        reply_text = prompt_with_tool_call(llm_response)
        # Store as simple string
        reply_content = "tool called"
        res=reply_text
    else:
        reply_text = llm_response
        # Extract just the content if it's an AIMessage
        reply_content = reply_text.content if hasattr(reply_text, 'content') else str(reply_text)
        res=reply_content

    # Save conversation (simplified format)
    save_session_history(session_id, [
    HumanMessage(content=user_input),
    AIMessage(content=reply_content)
])

    return jsonify({"response": res})
    """user_input = request.json.get("message", "").strip()
    session_id = request.cookies.get("user_email")
    
    if not session_id:
        return jsonify({"error": "No session ID in cookies"}), 401

    user = users_collection.find_one({"email": session_id})
    if not user:
        return jsonify({"error": "User not found"}), 404

    name = user.get("name", "User")

    # 1. Load session history
    history = get_session_history(session_id).messages
    maybe_update_summary(session_id, history)
    saved_summary = get_session_summary(session_id) or "No prior summary."

    # 2. Add current user input to the messages
    history.append(HumanMessage(content=user_input))

    # 3. Build agent state
    state = {"messages": history}

    # 4. Run the graph agent
    new_state = gapp.invoke(state)

    # 5. Extract the assistant message
    new_messages = new_state["messages"]
    latest_response = new_messages[-1].content if new_messages else "Sorry, I couldn't respond."

    # 6. Save new history
    save_session_history(session_id, new_messages)

    # 7. Return response
    return jsonify({"response": latest_response})"""

@app.route("/get_messages", methods=["GET"])
def get_messages():
    try:
        messages = list(db["social_media"].find({}, {"_id": 0}))  # Exclude MongoDB ObjectID for clean output
        return jsonify({"messages": messages}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_reminders", methods=["GET"])
def get_reminders():
    try:
        reminders = list(db["reminders"].find({}, {"_id": 0}))
        return jsonify({"reminders": reminders}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
