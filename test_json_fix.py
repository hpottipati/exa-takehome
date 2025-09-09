#!/usr/bin/env python3
"""Test the JSON parsing fix for control characters"""

import json
import re

def clean_json_string(text):
    """Clean a string that should contain JSON by escaping control characters"""
    # First, try to find the JSON object boundaries
    start_idx = text.find('{')
    end_idx = -1
    
    if start_idx != -1:
        # Count braces to find the matching closing brace
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            # Track if we're inside a string
            if char == '"' and not escape_next:
                in_string = not in_string
            elif char == '\\':
                escape_next = not escape_next
                continue
            
            escape_next = False
            
            # Count braces only outside of strings
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
        
        if end_idx > start_idx:
            # Extract just the JSON part
            json_str = text[start_idx:end_idx+1]
            
            # Replace unescaped control characters within string values
            # This regex finds string values and processes them
            def escape_string_value(match):
                value = match.group(1)
                # Replace actual newlines, tabs, etc. with escaped versions
                value = value.replace('\n', '\\n')
                value = value.replace('\r', '\\r')
                value = value.replace('\t', '\\t')
                # Remove other control characters
                value = ''.join(char if ord(char) >= 32 or char in '\n\r\t' else ' ' for char in value)
                return f'"{value}"'
            
            # Process string values in the JSON
            json_str = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', escape_string_value, json_str)
            return json_str
    
    return text

# Test cases
test_cases = [
    # Case 1: JSON with newlines in reasoning
    '''{
  "winner": "A",
  "confidence": 2,
  "reasoning": "Answer A provides a comprehensive analysis of the compliance requirements for using AI in hiring across NYC, California, and Illinois.
The answer breaks down the specific requirements for each location.
This is much better than Answer B."
}''',
    
    # Case 2: JSON with tabs and newlines
    '''{
  "winner": "B",
  "confidence": 3,
  "reasoning": "Answer B is superior because:
	- It provides specific citations
	- It covers all jurisdictions
	- The analysis is more thorough"
}''',
    
    # Case 3: JSON with control characters
    '''Some text before {
  "winner": "A",
  "confidence": 2,
  "reasoning": "This has a control char \x0c here and a newline
here"
} and some text after'''
]

print("Testing JSON cleaning function...")
print("=" * 60)

for i, test_json in enumerate(test_cases, 1):
    print(f"\nTest Case {i}:")
    print("Original (has control chars):", repr(test_json[:100]) + "...")
    
    try:
        # Try parsing without cleaning
        result = json.loads(test_json)
        print("✓ Parsed without cleaning")
    except json.JSONDecodeError as e:
        print(f"✗ Failed without cleaning: {e}")
        
        # Try with cleaning
        cleaned = clean_json_string(test_json)
        try:
            result = json.loads(cleaned)
            print(f"✓ Parsed after cleaning")
            print(f"  Winner: {result['winner']}, Confidence: {result['confidence']}")
        except json.JSONDecodeError as e2:
            print(f"✗ Still failed after cleaning: {e2}")

print("\n" + "=" * 60)
print("Test complete!")