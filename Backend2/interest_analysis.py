import json
import os
import glob
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt

# Import Google Cloud Vertex AI for Gemini
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession

# Function to load chat sessions from a directory
def load_chat_sessions(directory="./Backend2"): # Adjusted default directory for consistency with previous changes
    """Load chat history from all JSON files in a directory"""
    all_messages = []
    
    try:
        # Construct path to chat_history.json within the specified directory
        file_path = os.path.join(directory, "chat_history.json")
            
        if not os.path.exists(file_path):
            print(f"Chat history file not found: {file_path}")
            return []
            
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
    except Exception as e:
        print(f"Error loading chat history: {e}")
    
    return all_messages

# LLM API client (using Gemini via Vertex AI)
class LLMClient:
    def __init__(self, project_id=None, location=None):
        try:
            # Initialize Vertex AI for Gemini access
            # project_id and location should come from the environment or function args
            vertexai.init(project=project_id, location=location)
            self.model = GenerativeModel("gemini-1.5-flash") # Using a fast Gemini model
            print(f"LLMClient: Vertex AI and Gemini model 'gemini-1.5-flash' initialized successfully for analysis.")
        except Exception as e:
            print(f"LLMClient Error: Failed to initialize Vertex AI / Gemini model: {e}")
            print("Hint: Ensure 'GOOGLE_APPLICATION_CREDENTIALS' is set, billing is enabled, and 'Vertex AI' API is enabled for your project.")
            self.model = None # Set to None if initialization fails

    def generate_insights(self, prompt):
        """Send a prompt to the Gemini LLM API and get a response."""
        if self.model is None:
            return "Error: LLM model not initialized."
        try:
            # Use the Gemini model to generate content
            # For analysis, we can often use generate_content directly without a chat session
            # if the prompt is self-contained.
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.7, "max_output_tokens": 500}
            )
            
            # Extract and return the generated content
            return response.text
        except Exception as e:
            print(f"Error calling Gemini LLM API for insights: {e}")
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
        1. category - Must be one of: "How Things Work", "Big Questions", "Creativity & Expression", "World & Cultures", "Nature & Life", "Self & Emotions", "Other"
        2. subcategory - A more specific classification within the category (1-3 words)
        3. topic - The main subject of interest (1-2 words, lowercase)

        Return only valid JSON. Ensure the JSON is properly formatted with double quotes for keys and string values.
        Example:
        {{
          "category": "category name",
          "subcategory": "subcategory name",
          "topic": "specific topic"
        }}
        """

        try:
            response_text = self.generate_insights(prompt)
            
            # Clean up response to ensure valid JSON (LLMs sometimes wrap JSON in markdown)
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            # Attempt to parse JSON. Add a fallback if parsing fails.
            try:
                metadata = json.loads(cleaned_response)
            except json.JSONDecodeError as jde:
                print(f"JSON Decode Error for metadata: {jde}")
                print(f"Raw response attempting to parse: {cleaned_response}")
                # Fallback to a default structure if parsing fails
                metadata = {}
            
            return {
                "category": metadata.get("category", "Other"),
                "subcategory": metadata.get("subcategory", "General"),
                "topic": metadata.get("topic", "general").lower()
            }
        except Exception as e:
            print(f"Error parsing metadata from LLM or generating insights: {e}")
            print(f"Raw response that caused error: {response_text}")
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
        # Only enrich user messages and only if they don't already have metadata
        if message["role"] == "user" and "category" not in message:
            print(f"Enriching message: {message['content']}")
            metadata = llm_client.enrich_message_metadata(message["content"])
            enriched_message = message.copy() # Create a copy before modifying
            enriched_message.update(metadata)
        else:
            enriched_message = message.copy() # Always copy to avoid modifying original list items
        
        enriched_messages.append(enriched_message)
    
    return enriched_messages

# Function to filter messages for specific date range
def filter_by_date_range(messages, start_date, end_date):
    filtered_messages = []
    
    for message in messages:
        try:
            # Handle potential timezone info by making it timezone-naive or aware consistently
            message_date = datetime.fromisoformat(message["timestamp"])
            # If message_date is timezone-aware, ensure start_date and end_date are also aware, or convert message_date to naive
            # For simplicity, if timestamps are always local, comparing directly works.
            # Otherwise, use .replace(tzinfo=None) or convert to UTC.
            if message_date.tzinfo is not None:
                message_date = message_date.replace(tzinfo=None)

            if start_date <= message_date <= end_date:
                filtered_messages.append(message)
        except (KeyError, ValueError) as e:
            print(f"Error processing message timestamp '{message.get('timestamp')}': {e}. Skipping message.")
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

# Function to create visual interest distribution (remains the same)
def create_interest_distribution_chart(categories):
    if not categories:
        print("No categories to chart.")
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
    
    # Ensure all labels have a color, default to grey if not in map
    pie_colors = [colors.get(label, '#999999') for label in labels]

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=pie_colors, startangle=90)
    plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Child Interest Distribution')
    
    # Only save the chart, don't create any other files.
    # The path should be relative to where the API runs and potentially served.
    chart_path = 'interest_distribution.png'
    plt.savefig(chart_path)
    plt.close() # Close the plot to free memory
    
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
    Generate a personalized weekly summary for a parent about their child's learning interests based on conversation topics.
    
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
            cleaned_response = cleaned_codes[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        if cleaned_response.endswith("```json"): # Catch cases where it might end with ```json unexpectedly
            cleaned_response = cleaned_response[:-7]
        
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
        print(f"Error parsing LLM response as JSON for conversation starters: {e}")
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
        if cleaned_response.endswith("```json"): # Catch cases where it might end with ```json unexpectedly
            cleaned_response = cleaned_response[:-7]
        
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
        print(f"Error parsing LLM response as JSON for suggested activities: {e}")   
        print(f"Raw response: {response}")
        return ["Build a model related to your child's interests", 
                "Plan a field trip to explore topics in real life", 
                "Start a small project that builds on their questions"]

# Main function to analyze the past week
def analyze_past_week(chat_directory="./Backend2", project_id=None, location=None):
    # Initialize LLM client with GCP project and location for Vertex AI
    llm_client = LLMClient(project_id=project_id, location=location)
    
    if llm_client.model is None:
        return {
            "error": "LLM client could not be initialized. Check GCP credentials and Vertex AI setup."
        }

    # Load all chat sessions from the chat_history.json file
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
            "error": "No chat messages found for the past week in the last week."
        }
    
    # Enrich chat history with metadata
    # This step involves LLM calls
    enriched_messages = enrich_chat_history(weekly_messages, llm_client)
    
    # Analyze interest distribution
    category_percentages = analyze_interest_distribution(enriched_messages)
    
    # Create chart
    chart_path = create_interest_distribution_chart(category_percentages)
    
    # Analyze topics (this is internal and can be displayed raw or summarized)
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
        "top_topics": [(topic, count) for topic, count in topics[:5]], # Return top 5 topics
        "weekly_summary": summary,
        "conversation_starters": starters,
        "suggested_activities": activities,
        "chart_path": chart_path # Path to the generated chart image
    }

    return report

def run_analysis(project_id=None, location=None):
    """
    Wrapper function to run the full chat analysis pipeline.
    project_id and location are passed for Vertex AI initialization.
    """
    chat_directory = "Backend2/" # This should align with where chat_history.json is
    report = analyze_past_week(chat_directory, project_id, location)
    
    return report
