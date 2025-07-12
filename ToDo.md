
# Data flow
## Input

1. Structural symptom questions
-yes or no
2. lab parameters
- user enter value
- user upload lab test report file
  - OPENAI read in data
  - Agreed rule: overwrite user entered value for the same lab parameter key
- Read in values from user's lab test; overwrite the interface lab parameters values

## Output
- contain user's input and analysis result
- text display 
- pdf file of analysis report - <b>to do</b>

