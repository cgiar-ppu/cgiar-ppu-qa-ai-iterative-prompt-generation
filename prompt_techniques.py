# prompt_techniques.py

PROMPT_TECHNIQUES = [
    {
        'name': 'Simplify Language',
        'function': lambda prompt: prompt.replace('Please analyze', 'Analyze')
    },
    {
        'name': 'Emphasize Instructions',
        'function': lambda prompt: prompt.replace('Provide a score', '***Provide a score***')
    },
    # Add more techniques as needed
]