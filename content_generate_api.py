from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import logging
import json
import re
from typing import List, Dict, Any
from http import HTTPStatus

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0.5)

# Prompt Template
SYSTEM_PROMPT = SYSTEM_PROMPT = """
Task: Provide a detailed syllabus for the given subject in strict JSON format, adhering to these guidelines:
    -> Generate a concise, focused syllabus for the given subject.
    -> The syllabus should be in JSON format and consist only of the subject name and a list of topics.
    -> Avoid repeating topics or adding redundant information. Limit each topic to a single line.
    -> Ensure no additional text, explanations, or information outside the JSON structure.
    
Example:
{{
    "subject": "Software Engineering",
    "syllabus": ["Introduction to software engineering", "Software crises", "Software Life Cycle Model", "Waterfall Model", "Prototype Model", "Spiral Model", "Agile Model", "Software Requirement Analysis and Specification"]
}}
Output: Provide only the JSON object as per the format above.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "Subject: {subject}")  
])

# Create chain
model = prompt_template | llm

def clean_response(response: str) -> Dict[str, Any]:
    """Clean the LLM response and extract pure JSON object."""
    try:
        # Try to find content between <startJson> and </endJson>
        if "<startJson>" in response:
            json_str = response.split("<startJson>")[1].split("</endJson>")[0].strip()
            # Wrap in curly braces if not present
            if not json_str.startswith("{"):
                json_str = "{" + json_str + "}"
        else:
            # Try to find JSON object in the regular text
            # Remove newlines and escape characters for better regex matching
            cleaned_text = response.replace('\n', ' ').replace('\\', '')
            json_match = re.search(r'({[^}]*})', cleaned_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON found, create a basic structure
                return {
                    "error": "Could not extract JSON from response",
                    "raw_response": response
                }
        
        # Parse the JSON string
        json_obj = json.loads(json_str)
        return json_obj
        
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Error cleaning response: {e}\nResponse: {response}")
        return {
            "error": "Failed to parse JSON response",
            "details": str(e),
            "raw_response": response
        }

def validate_request(data: List[Dict[str, Any]]) -> List[str]:
    """Validate the incoming request data."""
    errors = []
    if not isinstance(data, list):
        errors.append("Request body must be a list of objects")
        return errors
    
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Item at index {idx} must be an object")
        elif "subject" not in item:
            errors.append(f"Missing 'subject' field in item at index {idx}")
        elif not isinstance(item.get("subject"), str):
            errors.append(f"'subject' must be a string in item at index {idx}")
        elif not item.get("subject").strip():
            errors.append(f"'subject' cannot be empty in item at index {idx}")
    
    return errors

def process_item(subject: str) -> Dict[str, Any]:
    """Process a single subject and generate its syllabus."""
    try:
        logger.info(f"Processing subject: {subject}")
        response = model.invoke({"subject": subject}) 
        output = response.content
        
        # Log the raw output for debugging
        logger.debug(f"Raw output from LLM for subject '{subject}': {output}")
        
        # Clean the response
        # cleaned_response = clean_response(output)
        
        return output
        
    except Exception as e:
        logger.error(f"Error processing subject '{subject}': {str(e)}")
        return {
            "error": "An unexpected error occurred during processing",
            "details": str(e)
        }

@app.route('/SwaySyllabusGenerator', methods=['POST'])
def generate_syllabus():
    """Generate syllabi for multiple subjects."""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), HTTPStatus.BAD_REQUEST
        
        # Validate request
        validation_errors = validate_request(data)
        if validation_errors:
            return jsonify({"errors": validation_errors}), HTTPStatus.BAD_REQUEST
        
        # Process each subject
        responses = []
        for item in data:
            response = process_item(subject=item["subject"])
            responses.append(response)
        
        return jsonify(responses), HTTPStatus.OK

    except Exception as e:
        logger.error(f"Error in generate_syllabus endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), HTTPStatus.INTERNAL_SERVER_ERROR

@app.route('/', methods=['GET'])
def home():
    """Home endpoint."""
    return jsonify({
        "message": "Welcome to Sway Syllabus Generator API",
        "version": "1.0",
        "endpoints": {
            "/SwaySyllabusGenerator": {
                "method": "POST",
                "description": "Generate syllabi for multiple subjects",
                "request_format": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {
                                "type": "string",
                                "description": "The subject name"
                            }
                        }
                    }
                }
            }
        }
    })

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler."""
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({
        "error": "Internal server error",
        "details": str(error)
    }), HTTPStatus.INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, threaded=True)