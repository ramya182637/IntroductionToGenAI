import os
from huggingface_hub import InferenceClient

# Initialize the AI model client
ai_client = InferenceClient(
    model="mistralai/Mistral-Nemo-Instruct-2407",
    token="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"  # API token (removed for security purposes)
)

print("Type your message to start chatting with the AI. Type 'exit' or 'quit' to end the conversation. Enjoy!")
print()

# Create an empty list to store conversation history
dialogue_history = []

# Start an infinite loop for continuous conversation
while True:
    user_input_text = input("User: ")
    
    # Check if the user wants to exit the conversation
    if user_input_text.lower() in ['exit', 'quit']:
        print("Session ended. Farewell!")
        break
    
    # Handle empty or invalid input
    if not user_input_text.strip():
        print("Please enter a valid message.")
        continue
    
    # Append user input to the dialogue history
    dialogue_history.append({'role': 'user', 'content': user_input_text})
    
    try:
        # Generate a response from the AI based on the conversation so far
        ai_response = ai_client.chat_completion(dialogue_history, max_tokens=150, seed=42)
        
        # Display the AI's response
        print(f"AI: {ai_response.choices[0].message.content}")
        
        # Append the AI's response to the conversation history
        dialogue_history.append({'role': 'assistant', 'content': ai_response.choices[0].message.content})
    
    except Exception as e:
        # Handle any errors that may occur during the conversation
        print(f"Error encountered: {e}")
        break
