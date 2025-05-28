import pandas as pd
import re

# Configuration
data_path = 'routerbench_no_mcq_with_groundtruth.csv'  # Update with your dataset path

# Define response columns and ground truth
response_columns = [
    'WizardLM/WizardLM-13B-V1.2|model_response',
    'claude-instant-v1|model_response',
    'claude-v1|model_response',
    'claude-v2|model_response',
    'gpt-3.5-turbo-1106|model_response',
    'gpt-4-1106-preview|model_response',
    'meta/llama-2-70b-chat|model_response',
    'mistralai/mixtral-8x7b-chat|model_response',
    'zero-one-ai/Yi-34B-Chat|model_response',
    'meta/code-llama-instruct-34b-chat|model_response',
    'mistralai/mistral-7b-chat|model_response'
]
ground_truth_column = 'GroundTruth'

# Load dataset
df = pd.read_csv(data_path)

# Function to detect response type
def detect_response_type(response):
    if pd.isna(response):
        return 'none'
    response = str(response)
    # Code: ```language, ```, [LANGUAGE], or code keywords
    code_markers = [
        r'```(python|cpp|java|javascript)?',  # Matches ```python, ```cpp, or just ```
        r'\[PYTHON\]', r'\[CPP\]', r'\[JAVA\]', r'\[JAVASCRIPT\]',
        'def ', 'assert ', 'import ', 'class '
    ]
    if any(re.search(marker, response, re.IGNORECASE) for marker in code_markers):
        return 'code'
    # Math: number-operator patterns, LaTeX markers, or math keywords
    math_markers = [
        r'\d+\s*[\+\-\*/\^]\s*\d+', r'\$', r'\\times', r'\\frac', 'equation'
    ]
    if any(pattern in response for pattern in math_markers):
        return 'math'
    return 'text'

# Function to preprocess a single response
def preprocess_response(response):
    if pd.isna(response):
        return ""
    
    # Convert to string
    response = str(response)
    
    # Remove serialization artifacts (e.g., [' or [" at start, '] or "] at end)
    # Updated regex to be more robust
    response = re.sub(r'^\s*\[\'|^ \s*\[\"', '', response)  # Remove [' or [" at start
    response = re.sub(r'\'\]\s*$|\"\]\s*$', '', response)  # Remove '] or "] at end
    
    # Normalize escaped apostrophes (e.g., i\'m to i'm)
    response = response.replace(r'\'', "'")
    
    response_type = detect_response_type(response)
    
    if response_type == 'code':
        # Handle code responses (preserve \n in code blocks)
        parts = []
        in_code_block = False
        current_part = []
        # Match ```language, ```, or [LANGUAGE]
        code_start_markers = r'(```(python|cpp|java|javascript)?|\[(PYTHON|CPP|JAVA|JAVASCRIPT)\])'
        code_end_markers = r'(```|\[/(PYTHON|CPP|JAVA|JAVASCRIPT)\])'
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # Check for start of code block
            if re.search(code_start_markers, line, re.IGNORECASE):
                # If we were collecting non-code lines, join them with spaces
                if current_part and not in_code_block:
                    parts.append(' '.join(current_part))
                    current_part = []
                in_code_block = True
                current_part.append(line)
            # Check for end of code block
            elif re.search(code_end_markers, line, re.IGNORECASE):
                in_code_block = False
                current_part.append(line)
                # Join code block with preserved \n
                parts.append('\n'.join(current_part))
                current_part = []
            # Inside code block: preserve line
            elif in_code_block:
                current_part.append(line)
            # Outside code block: collect lines to join with spaces
            else:
                current_part.append(line)
        # Handle any remaining part
        if current_part:
            if in_code_block:
                parts.append('\n'.join(current_part))
            else:
                parts.append(' '.join(current_part))
        # Join parts, ensuring no empty parts
        cleaned_response = ' '.join(part for part in parts if part.strip())
    
    else:
        # Non-code responses: replace all \n with space
        # Math: normalize operators
        if response_type == 'math':
            response = re.sub(r'\\{2,}', r'\\', response)
            response = response.replace(r'\\ *', '*')
            response = response.replace(r'\\+', '+')
            response = response.replace(r'\\-', '-')
            response = response.replace(r'\\^', '^')
        # Replace \n with space for all non-code responses
        cleaned_response = response.replace('\n', ' ')
        cleaned_response = ' '.join(cleaned_response.split())
    
    return cleaned_response

# Apply preprocessing to response columns and ground truth
for col in response_columns + [ground_truth_column]:
    df[col] = df[col].apply(preprocess_response)

# Save preprocessed dataset
df.to_csv('preprocessed_dataset.csv', index=False)

print("Preprocessing completed. Cleaned dataset saved as 'preprocessed_dataset.csv'.")


# Apply preprocessing to response columns and ground truth
for col in response_columns + [ground_truth_column]:
    df[col] = df[col].apply(preprocess_response)

# --- START: Additional processing for specific requirements ---

# Define columns for the additional specific processing
# These are the columns mentioned by the user for this specific rule
columns_for_additional_processing = response_columns + [ground_truth_column]
if 'prompt' in df.columns:
    columns_for_additional_processing.append('prompt')
else:
    # If 'prompt' column might not exist, you can add a check or handle it
    print("Warning: 'prompt' column not found in DataFrame. Skipping for additional processing.")
    # Or, if 'prompt' is essential and might be missing, you might want to ensure it's in response_columns or handle error
columns_for_additional_processing = sorted(list(set(columns_for_additional_processing))) # Unique and sorted

# 1. Remove [" or [' from the beginning of the specified columns for all rows
for col in columns_for_additional_processing:
    if col in df.columns:
        # Ensure column is string type before attempting string operations
        df[col] = df[col].astype(str).str.replace(r'^\s*(?:\[\"|\[\')', '', regex=True)
        # Also remove trailing "] or '] if they exist, as per original preprocess_response logic
        df[col] = df[col].str.replace(r'(?:\"\]|\'\])\s*$', '', regex=True)
        # Normalize escaped apostrophes again, in case new ones were exposed or missed
        df[col] = df[col].str.replace("\\'", "'", regex=False)


# 2. For the top 8227 rows of these specific columns, replace \n with " " and normalize spaces
num_rows_to_modify = 8227
actual_rows_to_modify = min(num_rows_to_modify, len(df)) # Ensure we don't exceed DataFrame length
df_top_rows_indices = df.index[:actual_rows_to_modify]

for col in columns_for_additional_processing:
    if col in df.columns:
        # Ensure we are working with string data for the slice
        # Apply string operations to a copy of the slice to avoid potential SettingWithCopyWarning
        df.loc[df_top_rows_indices, col] = (
            df.loc[df_top_rows_indices, col]
            .astype(str)  # Convert to string
            .str.replace('\n+', ' ', regex=False)  # Replace \n with space
            .str.replace(r'\s+', ' ', regex=True)  # Normalize multiple spaces
            .str.strip()  # Strip leading/trailing whitespace
        )

# --- END: Additional processing for specific requirements ---

# Save preprocessed dataset
df.to_csv('preprocessed_dataset.csv', index=False)

print("Preprocessing completed. Cleaned dataset saved as 'preprocessed_dataset.csv'.")