# import json
# import os
# import glob
# from collections import Counter
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# import pandas as pd
# import requests
# import textwrap
# from groq import Groq

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv(dotenv_path=".env.local")


# # Function to load chat history files
# def load_chat_sessions(directory="database/mock_chat_sessions"):
#     all_messages = []
#     files = glob.glob(f"{directory}/*.json")
    
#     for file_path in files:
#         with open(file_path, 'r') as file:
#             session_data = json.load(file)
#             all_messages.extend(session_data["items"])
    
#     return all_messages

# # Function to filter messages for specific date range
# def filter_by_date_range(messages, start_date, end_date):
#     filtered_messages = []
    
#     for message in messages:
#         message_date = datetime.fromisoformat(message["timestamp"])
#         if start_date <= message_date <= end_date:
#             filtered_messages.append(message)
    
#     return filtered_messages

# # Function to analyze interest distribution
# def analyze_interest_distribution(messages):
#     # Filter only user messages with category info
#     user_interests = [msg for msg in messages if msg["role"] == "user" and "category" in msg]
    
#     # Count occurrences of each category
#     category_counts = Counter([msg["category"] for msg in user_interests])
    
#     # Convert to percentage
#     total = sum(category_counts.values())
#     category_percentages = {category: (count / total) * 100 for category, count in category_counts.items()}
    
#     return category_percentages

# # Function to analyze topics explored
# def analyze_topics(messages):
#     # Filter only user messages with topic info
#     user_topics = [msg for msg in messages if msg["role"] == "user" and "topic" in msg]
    
#     # Count occurrences of each topic
#     topic_counts = Counter([msg["topic"] for msg in user_topics])
    
#     # Get the most common topics
#     most_common_topics = topic_counts.most_common()
    
#     return most_common_topics

# # Function to create visual interest distribution
# def create_interest_distribution_chart(categories):
#     plt.figure(figsize=(8, 8))
#     labels = list(categories.keys())
#     sizes = list(categories.values())
    
#     # Define colors for each category
#     colors = {
#         'How Things Work': '#FF6B6B',
#         'Big Questions': '#4ECDC4',
#         'Creativity & Expression': '#FFD166',
#         'World & Cultures': '#6A0572',
#         'Nature & Life': '#1A936F',
#         'Self & Emotions': '#C06C84'
#     }
    
#     plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=[colors.get(key, '#999999') for key in labels])
#     plt.axis('equal')
#     plt.title('Interest Distribution')
#     plt.savefig('interest_distribution.png')
    
#     return 'interest_distribution.png'

# # LLM API client (replace with your preferred LLM API)
# class LLMClient:
#     def __init__(self, api_key=None):
        
#         self.api_key = os.environ.get("GROQ_API_KEY", "")

#         # Initialize the Groq client
#         self.client = Groq(api_key=self.api_key)
#         self.model = "llama-3.3-70b-versatile"
    
#     def generate_insights(self, prompt):
#         """
#         Send a prompt to the Groq LLM API and get a response.
#         """
#         try:
#             # Use the Groq client to generate a completion
#             chat_completion = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.7,
#                 max_tokens=500
#             )
            
#             # Extract and return the generated content
#             return chat_completion.choices[0].message.content
#         except Exception as e:
#             print(f"Error calling Groq LLM API: {e}")
#             return "Error generating insights. Please try again later."

# # Function to generate weekly summary using LLM
# def generate_weekly_summary_llm(messages, llm_client):
#     # Extract key information for the LLM
#     user_topics = [msg for msg in messages if msg["role"] == "user" and "topic" in msg]
#     topics = Counter([msg["topic"] for msg in user_topics]).most_common()
#     topics_str = ", ".join([f"{topic} ({count})" for topic, count in topics])
    
#     # Extract categories
#     categories = Counter([msg["category"] for msg in user_topics if "category" in msg]).most_common()
#     categories_str = ", ".join([f"{category} ({count})" for category, count in categories])
    
#     # Create a prompt for the LLM
#     prompt = f"""
#     Generate a personalized weekly summary for a parent about their child's learning interests.
    
#     The child explored these topics this week (topic and count):
#     {topics_str}
    
#     The topics fall into these categories (category and count):
#     {categories_str}
    
#     Please create a 2-3 sentence summary highlighting the main interests, 
#     what specifically caught the child's attention, and any patterns in their curiosity.
#     Write in a professional but warm tone, addressing the parent directly.
#     """
    
#     # Get response from LLM
#     response = llm_client.generate_insights(prompt)
#     return response

# # Function to generate conversation starters using LLM
# def generate_conversation_starters_llm(messages, llm_client):
#     # Extract key information for the LLM
#     user_topics = [msg for msg in messages if msg["role"] == "user" and "topic" in msg]
#     topics = Counter([msg["topic"] for msg in user_topics]).most_common(5)
#     topics_str = ", ".join([f"{topic} ({count})" for topic, count in topics])
    
#     # Create a prompt for the LLM
#     prompt = f"""
#     Generate 3 conversation starters for a parent to engage their child based on the child's recent interests.
    
#     The child's top interests this week were (topic and count):
#     {topics_str}
    
#     Create 3 open-ended questions that:
#     1. Are engaging and thought-provoking for a child
#     2. Relate directly to the topics the child has shown interest in
#     3. Encourage critical thinking and imagination
#     4. Are phrased in a way that would make a child excited to respond
    
#     Format the response as a JSON array of strings.
#     """
    
#     # Get response from LLM
#     response = llm_client.generate_insights(prompt)
    
#     try:
#         # Clean up the response to ensure it's valid JSON
#         # Remove any leading/trailing whitespace and markdown code block formatting if present
#         cleaned_response = response.strip()
#         if cleaned_response.startswith("```json"):
#             cleaned_response = cleaned_response[7:]  # Remove ```json
#         if cleaned_response.startswith("```"):
#             cleaned_response = cleaned_response[3:]  # Remove ```
#         if cleaned_response.endswith("```"):
#             cleaned_response = cleaned_response[:-3]  # Remove trailing ```
        
#         cleaned_response = cleaned_response.strip()
        
#         # Parse the JSON
#         starters = json.loads(cleaned_response)
        
#         # Ensure we have a list
#         if isinstance(starters, list):
#             return starters[:3]
#         else:
#             # If it's not a list (e.g., it's a dict with a 'starters' key)
#             if isinstance(starters, dict) and 'starters' in starters:
#                 return starters['starters'][:3]
#             else:
#                 # If we can't find a list, create fallback
#                 raise ValueError("Response is not in expected format")
#     except Exception as e:
#         print(f"Error parsing LLM response as JSON: {e}")
#         print(f"Raw response: {response}")

# # Function to generate suggested activities using LLM
# def generate_suggested_activities_llm(messages, llm_client):
#     # Extract key information for the LLM
#     user_topics = [msg for msg in messages if msg["role"] == "user" and "topic" in msg]
#     topics = Counter([msg["topic"] for msg in user_topics]).most_common(5)
#     topics_str = ", ".join([f"{topic} ({count})" for topic, count in topics])
    
#     # Extract categories
#     categories = Counter([msg["category"] for msg in user_topics if "category" in msg]).most_common()
#     categories_str = ", ".join([f"{category} ({count})" for category, count in categories])
    
#     # Create a prompt for the LLM
#     prompt = f"""
#     Generate 3 educational activities for a parent to do with their child based on the child's recent interests.
    
#     The child's top interests this week were (topic and count):
#     {topics_str}
    
#     The interests fall into these categories (category and count):
#     {categories_str}
    
#     Create 3 activities that:
#     1. Are engaging and educational for a child
#     2. Relate directly to the topics the child has shown interest in
#     3. Can be done at home or nearby
#     4. Encourage hands-on learning
    
#     Format the response as a JSON array of strings.
#     """
    
#     # Get response from LLM
#     response = llm_client.generate_insights(prompt)
    
#     try:
#         # Clean up the response to ensure it's valid JSON
#         # Remove any leading/trailing whitespace and markdown code block formatting if present
#         cleaned_response = response.strip()
#         if cleaned_response.startswith("```json"):
#             cleaned_response = cleaned_response[7:]  # Remove ```json
#         if cleaned_response.startswith("```"):
#             cleaned_response = cleaned_response[3:]  # Remove ```
#         if cleaned_response.endswith("```"):
#             cleaned_response = cleaned_response[:-3]  # Remove trailing ```
        
#         cleaned_response = cleaned_response.strip()
        
#         # Parse the JSON
#         starters = json.loads(cleaned_response)
        
#         # Ensure we have a list
#         if isinstance(starters, list):
#             return starters[:3]
#         else:
#             # If it's not a list (e.g., it's a dict with a 'starters' key)
#             if isinstance(starters, dict) and 'starters' in starters:
#                 return starters['starters'][:3]
#             else:
#                 # If we can't find a list, create fallback
#                 raise ValueError("Response is not in expected format")
#     except Exception as e:
#         print(f"Error parsing LLM response as JSON: {e}")   
#         print(f"Raw response: {response}")

# # Main function to analyze the past week
# def analyze_past_week():
#     # Initialize LLM client
#     llm_client = LLMClient()
    
#     # Load all chat sessions
#     all_messages = load_chat_sessions()
    
#     # Define date range for the past week
#     end_date = datetime.now()
#     start_date = end_date - timedelta(days=7)
    
#     # Filter messages for the past week
#     weekly_messages = filter_by_date_range(all_messages, start_date, end_date)
    
#     # Analyze interest distribution
#     category_percentages = analyze_interest_distribution(weekly_messages)
    
#     # Create chart
#     chart_path = create_interest_distribution_chart(category_percentages)
    
#     # Analyze topics
#     # topics = analyze_topics(weekly_messages)
    
#     # Generate weekly summary using LLM
#     summary = generate_weekly_summary_llm(weekly_messages, llm_client)
    
#     # Generate conversation starters using LLM
#     starters = generate_conversation_starters_llm(weekly_messages, llm_client)
    
#     # Generate suggested activities using LLM
#     activities = generate_suggested_activities_llm(weekly_messages, llm_client)
    
#     # Create report
#     report = {
#         "interest_distribution": dict(category_percentages),
#         "weekly_summary": summary,
#         "conversation_starters": starters,
#         "suggested_activities": activities,
#         "chart_path": chart_path
#     }

#     return report
    
# #     # Save report
# #     with open('weekly_report.json', 'w') as f:
# #         json.dump(report, f, indent=2)
    
# #     return report

# # if __name__ == "__main__":
# #     # Analyze the past week's data
# #     report = analyze_past_week()
# #     print(json.dumps(report, indent=2))

import json
import os
import glob
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt
from groq import Groq

# Function to load chat sessions from a directory
def load_chat_sessions(directory="database/chat_sessions"):
    """Load chat history from all JSON files in a directory"""
    all_messages = []
    
    try:
        # Check if the input is a directory or file
        if os.path.isdir(directory):
            files = glob.glob(os.path.join(directory, "*.json"))
            
            if not files:
                print(f"No JSON files found in directory: {directory}")
                return []
                
            for file_path in files:
                try:
                    with open(file_path, 'r') as file:
                        session_data = json.load(file)
                        # Handle both single session files and combined files
                        if "items" in session_data:
                            all_messages.extend(session_data["items"])
                        else:
                            all_messages.append(session_data)
                except Exception as e:
                    print(f"Error loading chat file {file_path}: {e}")
        else:
            # Handle single file input
            with open(directory, 'r') as f:
                data = json.load(f)
                if "items" in data:
                    all_messages.extend(data["items"])
                else:
                    all_messages.append(data)
    except Exception as e:
        print(f"Error loading chat history: {e}")
    
    return all_messages

# LLM API client (using Groq)
class LLMClient:
    def __init__(self, api_key=None):
        self.api_key = os.environ.get("GROQ_API_KEY", api_key or "")
        # Initialize the Groq client
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
    
    def generate_insights(self, prompt):
        """Send a prompt to the Groq LLM API and get a response."""
        try:
            # Use the Groq client to generate a completion
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract and return the generated content
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling Groq LLM API: {e}")
            return "Error generating insights. Please try again later."

    def enrich_message_metadata(self, message_content):
        """
        Use LLM to analyze a message and generate category, subcategory, and topic.
        """
        if not message_content or not isinstance(message_content, str):
            return {
                "category": "Other",
                "subcategory": "General",
                "topic": "general"
            }
        
        prompt = f"""
        Analyze this child's question or statement and classify it:

        "{message_content}"

        Provide JSON output with:
        1. category - Must be one of: "How Things Work", "Big Questions", "Creativity & Expression", "World & Cultures", "Nature & Life", "Self & Emotions"
        2. subcategory - A more specific classification within the category
        3. topic - The main subject of interest (1-2 words)

        Return only valid JSON like this:
        {{
          "category": "category name",
          "subcategory": "subcategory name",
          "topic": "specific topic"
        }}
        """

        try:
            response = self.generate_insights(prompt)
            
            # Clean up response to ensure valid JSON
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            metadata = json.loads(cleaned_response)
            
            return {
                "category": metadata.get("category", "Other"),
                "subcategory": metadata.get("subcategory", "General"),
                "topic": metadata.get("topic", "general").lower()
            }
        except Exception as e:
            print(f"Error parsing metadata from LLM: {e}")
            print(f"Raw response: {response}")
            return {
                "category": "Other",
                "subcategory": "General",
                "topic": "general"
            }

# Function to enrich chat history with metadata
def enrich_chat_history(messages, llm_client):
    """Add category, subcategory, and topic metadata to user messages"""
    enriched_messages = []
    
    for message in messages:
        enriched_message = message.copy()
        
        # Only enrich user messages and only if they don't already have metadata
        if message["role"] == "user" and "category" not in message:
            metadata = llm_client.enrich_message_metadata(message["content"])
            enriched_message.update(metadata)
        
        enriched_messages.append(enriched_message)
    
    return enriched_messages

# Function to filter messages for specific date range
def filter_by_date_range(messages, start_date, end_date):
    filtered_messages = []
    
    for message in messages:
        try:
            message_date = datetime.fromisoformat(message["timestamp"])
            if start_date <= message_date <= end_date:
                filtered_messages.append(message)
        except (KeyError, ValueError) as e:
            print(f"Error processing message timestamp: {e}")
            continue
    
    return filtered_messages

# Function to analyze interest distribution
def analyze_interest_distribution(messages):
    # Filter only user messages with category info
    user_interests = [msg for msg in messages if msg["role"] == "user" and "category" in msg]
    
    if not user_interests:
        return {}
        
    # Count occurrences of each category
    category_counts = Counter([msg["category"] for msg in user_interests])
    
    # Convert to percentage
    total = sum(category_counts.values())
    category_percentages = {category: (count / total) * 100 for category, count in category_counts.items()}
    
    return category_percentages

# Function to analyze topics explored
def analyze_topics(messages):
    # Filter only user messages with topic info
    user_topics = [msg for msg in messages if msg["role"] == "user" and "topic" in msg]
    
    # Count occurrences of each topic
    topic_counts = Counter([msg["topic"] for msg in user_topics])
    
    # Get the most common topics
    most_common_topics = topic_counts.most_common()
    
    return most_common_topics

# Function to create visual interest distribution
def create_interest_distribution_chart(categories):
    if not categories:
        return None
        
    plt.figure(figsize=(8, 8))
    labels = list(categories.keys())
    sizes = list(categories.values())
    
    # Define colors for each category
    colors = {
        'How Things Work': '#FF6B6B',
        'Big Questions': '#4ECDC4',
        'Creativity & Expression': '#FFD166',
        'World & Cultures': '#6A0572',
        'Nature & Life': '#1A936F',
        'Self & Emotions': '#C06C84',
        'Other': '#999999'
    }
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=[colors.get(key, '#999999') for key in labels])
    plt.axis('equal')
    plt.title('Interest Distribution')
    
    # Only save the chart, don't create any other files
    chart_path = 'interest_distribution.png'
    plt.savefig(chart_path)
    
    return chart_path

# Function to generate weekly summary using LLM
def generate_weekly_summary_llm(messages, llm_client):
    # Extract key information for the LLM
    user_topics = [msg for msg in messages if msg["role"] == "user" and "topic" in msg]
    
    if not user_topics:
        return "No topics were discussed this week."
        
    topics = Counter([msg["topic"] for msg in user_topics]).most_common()
    topics_str = ", ".join([f"{topic} ({count})" for topic, count in topics])
    
    # Extract categories
    categories = Counter([msg["category"] for msg in user_topics if "category" in msg]).most_common()
    categories_str = ", ".join([f"{category} ({count})" for category, count in categories])
    
    # Create a prompt for the LLM
    prompt = f"""
    Generate a personalized weekly summary for a parent about their child's learning interests.
    
    The child explored these topics this week (topic and count):
    {topics_str}
    
    The topics fall into these categories (category and count):
    {categories_str}
    
    Please create a 2-3 sentence summary highlighting the main interests, 
    what specifically caught the child's attention, and any patterns in their curiosity.
    Write in a professional but warm tone, addressing the parent directly.
    """
    
    # Get response from LLM
    response = llm_client.generate_insights(prompt)
    return response

# Function to generate conversation starters using LLM
def generate_conversation_starters_llm(messages, llm_client):
    # Extract key information for the LLM
    user_topics = [msg for msg in messages if msg["role"] == "user" and "topic" in msg]
    
    if not user_topics:
        return ["What new things are you curious about today?", 
                "What would you like to learn about?", 
                "Is there anything interesting you've been wondering about?"]
                
    topics = Counter([msg["topic"] for msg in user_topics]).most_common(5)
    topics_str = ", ".join([f"{topic} ({count})" for topic, count in topics])
    
    # Create a prompt for the LLM
    prompt = f"""
    Generate 3 conversation starters for a parent to engage their child based on the child's recent interests.
    
    The child's top interests this week were (topic and count):
    {topics_str}
    
    Create 3 open-ended questions that:
    1. Are engaging and thought-provoking for a child
    2. Relate directly to the topics the child has shown interest in
    3. Encourage critical thinking and imagination
    4. Are phrased in a way that would make a child excited to respond
    
    Format the response as a JSON array of strings.
    """
    
    # Get response from LLM
    response = llm_client.generate_insights(prompt)
    
    try:
        # Clean up the response to ensure it's valid JSON
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        
        cleaned_response = cleaned_response.strip()
        
        # Parse the JSON
        starters = json.loads(cleaned_response)
        
        # Ensure we have a list
        if isinstance(starters, list):
            return starters[:3]
        elif isinstance(starters, dict) and 'starters' in starters:
            return starters['starters'][:3]
        else:
            raise ValueError("Response is not in expected format")
    except Exception as e:
        print(f"Error parsing LLM response as JSON: {e}")
        print(f"Raw response: {response}")
        return ["What was your favorite thing you learned about this week?", 
                "What are you most curious about right now?", 
                "What would you like to explore together?"]

# Function to generate suggested activities using LLM
def generate_suggested_activities_llm(messages, llm_client):
    # Extract key information for the LLM
    user_topics = [msg for msg in messages if msg["role"] == "user" and "topic" in msg]
    
    if not user_topics:
        return ["Visit a local museum and explore different exhibits", 
                "Set up a simple science experiment at home", 
                "Read a book on a new topic and discuss it together"]
                
    topics = Counter([msg["topic"] for msg in user_topics]).most_common(5)
    topics_str = ", ".join([f"{topic} ({count})" for topic, count in topics])
    
    # Extract categories
    categories = Counter([msg["category"] for msg in user_topics if "category" in msg]).most_common()
    categories_str = ", ".join([f"{category} ({count})" for category, count in categories])
    
    # Create a prompt for the LLM
    prompt = f"""
    Generate 3 educational activities for a parent to do with their child based on the child's recent interests.
    
    The child's top interests this week were (topic and count):
    {topics_str}
    
    The interests fall into these categories (category and count):
    {categories_str}
    
    Create 3 activities that:
    1. Are engaging and educational for a child
    2. Relate directly to the topics the child has shown interest in
    3. Can be done at home or nearby
    4. Encourage hands-on learning
    
    Format the response as a JSON array of strings.
    """
    
    # Get response from LLM
    response = llm_client.generate_insights(prompt)
    
    try:
        # Clean up the response to ensure it's valid JSON
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        
        cleaned_response = cleaned_response.strip()
        
        # Parse the JSON
        activities = json.loads(cleaned_response)
        
        # Ensure we have a list
        if isinstance(activities, list):
            return activities[:3]
        elif isinstance(activities, dict) and 'activities' in activities:
            return activities['activities'][:3]
        else:
            raise ValueError("Response is not in expected format")
    except Exception as e:
        print(f"Error parsing LLM response as JSON: {e}")   
        print(f"Raw response: {response}")
        return ["Build a model related to your child's interests", 
                "Plan a field trip to explore topics in real life", 
                "Start a small project that builds on their questions"]

# Main function to analyze the past week
def analyze_past_week(chat_directory="database/chat_sessions", api_key=None):
    # Initialize LLM client
    llm_client = LLMClient(api_key=api_key)
    
    # Load all chat sessions
    all_messages = load_chat_sessions(chat_directory)
    
    if not all_messages:
        return {
            "error": "No chat history found or files could not be loaded."
        }
    
    # Define date range for the past week
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Filter messages for the past week
    weekly_messages = filter_by_date_range(all_messages, start_date, end_date)
    
    if not weekly_messages:
        return {
            "error": "No chat messages found for the past week."
        }
    
    # Enrich chat history with metadata
    enriched_messages = enrich_chat_history(weekly_messages, llm_client)
    
    # Analyze interest distribution
    category_percentages = analyze_interest_distribution(enriched_messages)
    
    # Create chart
    chart_path = create_interest_distribution_chart(category_percentages)
    
    # Analyze topics
    topics = analyze_topics(enriched_messages)
    
    # Generate weekly summary using LLM
    summary = generate_weekly_summary_llm(enriched_messages, llm_client)
    
    # Generate conversation starters using LLM
    starters = generate_conversation_starters_llm(enriched_messages, llm_client)
    
    # Generate suggested activities using LLM
    activities = generate_suggested_activities_llm(enriched_messages, llm_client)
    
    # Create report
    report = {
        "interest_distribution": dict(category_percentages),
        "topics": [(topic, count) for topic, count in topics],
        "weekly_summary": summary,
        "conversation_starters": starters,
        "suggested_activities": activities,
        "chart_path": chart_path
    }

    return report

# Example usage
if __name__ == "__main__":
    # You can set your API key here or as an environment variable
    api_key = ""  # Replace with your actual API key
    
    # Directory containing chat history files
    chat_directory = "database/chat_sessions"
    
    # Analyze the past week's chat history
    report = analyze_past_week(chat_directory, api_key)
    
    # Print the report
    print("\n=== WEEKLY LEARNING REPORT ===\n")
    
    if "error" in report:
        print(f"Error: {report['error']}")
    else:
        print("INTEREST DISTRIBUTION:")
        for category, percentage in report["interest_distribution"].items():
            print(f"- {category}: {percentage:.1f}%")
        
        print("\nTOP TOPICS:")
        for topic, count in report["topics"][:5]:
            print(f"- {topic}: {count} times")
        
        print("\nWEEKLY SUMMARY:")
        print(report["weekly_summary"])
        
        print("\nSUGGESTED CONVERSATION STARTERS:")
        for i, starter in enumerate(report["conversation_starters"], 1):
            print(f"{i}. {starter}")
        
        print("\nSUGGESTED ACTIVITIES:")
        for i, activity in enumerate(report["suggested_activities"], 1):
            print(f"{i}. {activity}")
        
        if report["chart_path"]:
            print(f"\nInterest distribution chart saved as: {report['chart_path']}")