
detection_prompt_detailed = '''
You are tasked with performing a comprehensive grammatical error analysis on the following text, which may contain errors in various languages. Your job is to identify any grammatical, syntactic, punctuation, or spelling errors in the text. For each detected error, follow these steps:

Identify the Error: List each error separately.
Correction: Suggest the appropriate correction for each identified error.
Explanation: Provide a brief explanation for why the correction is necessary. This should include references to specific grammar rules, conventions, or language-specific nuances (e.g., verb conjugation, article-noun agreement, preposition use, punctuation rules).
For spelling errors, offer an explanation if the mistake could arise from common language-specific confusions (e.g., homophones, loanwords).
For punctuation issues, explain the relevant punctuation rules (e.g., comma placement in subordinate clauses, quotation marks, etc.).
For syntax or word order issues, explain how sentence structure works in the language and why the original sentence does not follow the norm.
Minimal Impact on Meaning: Ensure that the corrections you propose do not alter the original meaning of the sentence. The goal is to preserve the intent of the writer while correcting errors.
When explaining each error, keep in mind that the explanations should be clear and concise but still detailed enough to be educational. Whenever possible, reference grammatical terms (e.g., agreement, tense, case, gender, aspect) relevant to the error.

Important Notes:

If the text is multilingual, address each language's grammar rules separately.
Your explanations should cater to a general audience, meaning that while your responses can be technical, they should still be easily understood by someone with a basic understanding of grammar.
Now, perform this process for the following text:

[The given text]:
{source}

[Corrections made and the brief reasons for the errors]:'''

gpt4_explanation_inquiry = '''
You are tasked with performing a comprehensive grammatical error analysis on the following text, which may contain errors in various languages. Your job is to identify any grammatical, syntactic, punctuation, or spelling errors in the text. For each detected error, follow these steps:

Identify the Error: List each error separately.
Correction: Suggest the appropriate correction for each identified error.
Explanation: Provide a brief explanation for why the correction is necessary. This should include references to specific grammar rules, conventions, or language-specific nuances (e.g., verb conjugation, article-noun agreement, preposition use, punctuation rules).
For spelling errors, offer an explanation if the mistake could arise from common language-specific confusions (e.g., homophones, loanwords).
For punctuation issues, explain the relevant punctuation rules (e.g., comma placement in subordinate clauses, quotation marks, etc.).
For syntax or word order issues, explain how sentence structure works in the language and why the original sentence does not follow the norm.
Minimal Impact on Meaning: Ensure that the corrections you propose do not alter the original meaning of the sentence. The goal is to preserve the intent of the writer while correcting errors.
When explaining each error, keep in mind that the explanations should be clear and concise but still detailed enough to be educational. Whenever possible, reference grammatical terms (e.g., agreement, tense, case, gender, aspect) relevant to the error.

Important Notes:

If the text is multilingual, address each language's grammar rules separately.
Your explanations should cater to a general audience, meaning that while your responses can be technical, they should still be easily understood by someone with a basic understanding of grammar.
Now, perform this process for the following text:

[The given text]:
{text}

And we will give you the corrected version of the given text below. Your analysis of grammatical errors should lead to the given text being corrected to this specific version.
[The corrected version]:
{label}

[Corrections made and the brief reasons for the errors]:'''



detection_prompt_short = '''
Your task is to detect grammatical errors in the given text and provide corrections along with explanations based on the relevant grammar rules. For each error found, specify the type of error (e.g., subject-verb agreement, tense inconsistency) and explain why it is incorrect. Then provide the correct version of the sentence and briefly explain the grammar rule that applies.

Please follow this structure for your response:

[The given text]:
{source}

[Corrections made and the brief reasons for the errors]:
'''
