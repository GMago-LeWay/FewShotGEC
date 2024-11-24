correct_inquiry_default = '''For the given text, you should detect a few short segments that are most important for grammatical correctness. And briefly explain the grammatical knowledge involved. 
Here are some examples:

[The given text]:
Also , you 'll meet friendly people who usually ask you to be friends and exchange telephone numbers .
[Important segments]:
<Explanation Start>
<Segment>`ask` at `usually ask you`</Segment>:<Grammar>The subject-verb agreement is correct, as "people" is a plural noun and "ask" is a plural verb in the present tense.</Grammar>
<Segment>`telephone numbers` at `exchange telephone numbers .`</Segment>:<Grammar>Based on the context, "exchange telephone numbers" refers to exchanging phone numbers with friendly people, so using the plural "numbers" is correct.</Grammar>
<Explanation End>

Now it is your turn to do the same detection process for the following text. Follow the above format and do not include additional polite or summarizing remarks. And your response should not exceed 200 words. 
You need to note that only one pair of <Explanation Start> and <Explanation End> can appear in your answer to identify the formatted answer boundary.

[The given text]:
{text}
[Important segments]:
'''

error_inquiry_default = '''Please provide an explanation for each grammatical error correction needed in the sentence. 

Follow the explanation format and do not include additional polite or summarizing remarks. And your response should not exceed 200 words. We have extract edits from the given text, so your explanation should be formatted as:
[The explanation]:
{error_format}

The content between [] is the place where you should replace with your grammar analysis. You need to note that only one pair of <Explanation Start> and <Explanation End> can appear in your answer to identify the formatted answer boundary.

[The given text]:
{text}
[Corrected text]:
{label}
[The explanation]:
'''

correct_inquiry_unlimited = '''For the given text, you should detect a few short segments that are most important for grammatical correctness. And briefly explain the grammatical knowledge involved. 
Here are some examples:

[The given text]:
Also , you 'll meet friendly people who usually ask you to be friends and exchange telephone numbers .
[Important segments]:
<Explanation Start>
<Segment>`ask` at `usually ask you`</Segment>:<Grammar>The subject-verb agreement is correct, as "people" is a plural noun and "ask" is a plural verb in the present tense.</Grammar>
<Segment>`telephone numbers` at `exchange telephone numbers .`</Segment>:<Grammar>Based on the context, "exchange telephone numbers" refers to exchanging phone numbers with friendly people, so using the plural "numbers" is correct.</Grammar>
<Explanation End>

Now it is your turn to do the same detection process for the following text. Follow the above format and do not include additional polite or summarizing remarks.
You need to note that only one pair of <Explanation Start> and <Explanation End> can appear in your answer to identify the formatted answer boundary.

[The given text]:
{text}
[Important segments]:
'''

error_inquiry_unlimited = '''Please provide an explanation for each grammatical error correction needed in the sentence. 

Follow the explanation format and do not include additional polite or summarizing remarks. We have extract edits from the given text, so your explanation should be formatted as:
[The explanation]:
{error_format}

The content between [] is the place where you should replace with your grammar analysis. You need to note that only one pair of <Explanation Start> and <Explanation End> can appear in your answer to identify the formatted answer boundary.

[The given text]:
{text}
[Corrected text]:
{label}
[The explanation]:
'''


correct_inquiry_example_explanation = '''For the given text, you should detect a few short segments that are most important for grammatical correctness. And briefly explain the grammatical knowledge involved. 
Here are some examples:

[The given text]:
Also , you 'll meet friendly people who usually ask you to be friends and exchange telephone numbers .
[Important segments]:
<Explanation Start>
`ask`: The subject-verb agreement is correct, as "people" is a plural noun and "ask" is a plural verb in the present tense.
`telephone numbers`: Based on the context, "exchange telephone numbers" refers to exchanging phone numbers with friendly people, so using the plural "numbers" is correct.
<Explanation End>

[The given text]:
Aber ich habe oft gesehen , wie viele Leute , vor allem in Südafrika , nach so einem Studium keinen Job finden können .
[Important segments]:
<Explanation Start>
`einem`: The preposition "nach" indicates "after" in terms of time, and when it is followed by a noun phrase, the dative case should be used. Since "Studium" is a neuter noun, the correct form of the indefinite article in the dative case is "einem".
<Explanation End>

Now it is your turn to do the same detection process for the following text. Follow the above format and do not include additional polite or summarizing remarks. And your response should not exceed 200 words. 
You need to note that only one pair of <Explanation Start> and <Explanation End> can appear in your answer to identify the formatted answer boundary.

[The given text]:
{text}
[Important segments]:
'''

error_inquiry_example_explanation = '''Please provide an explanation for each grammatical error correction needed in the sentence. 
Here are some examples:

[The given text]:
I am applying to this position because I would like to work as organizer .
[Corrected text]:
I am applying for this position because I would like to work as an organizer .
[The explanation]:
<Explanation Start>
Replace `to` with `for`: The word "to" is a preposition used to indicate movement or direction, whereas "for" is a preposition used to indicate purpose or intention. In this context, the sentence is describing the reason for applying, so "for" is the correct choice to indicate the purpose of applying for the position.
Insert `an` between `as` and `organizer`: The word "organizer" starts with a vowel sound, so the indefinite article "an" is used instead of "a". This is a rule in English grammar that applies to words beginning with a vowel sound, such as "an apple" or "an organizer".
<Explanation End>

[The given text]:
Es gibt diesen Paradox , dass man sich befreit indem man Grenzen stellt .
[Corrected text]:
Es gibt dieses Paradox , dass man sich befreit , indem man Grenzen stellt .
[The explanation]:
<Explanation Start>
Replace `diesen` with `dieses`: The word `diesen` is a masculine accusative form of the demonstrative pronoun `dies`, but in this context, it should be `dieses` because `Paradox` is a neuter noun. In German, the article agrees with the noun in gender, number, and case. Therefore, the correct form is `dieses`.
Insert `,` between `befreit` and `indem`: In German, when using the subordinating conjunction `indem` to introduce a subordinate clause, a comma is usually placed before it to separate the main clause from the subordinate clause. This is a common punctuation rule in German to improve sentence clarity and readability.
<Explanation End>

Now it is your turn to do the same explanation process for the following text. Follow the above format and do not include additional polite or summarizing remarks. And your response should not exceed 200 words. We have extract edits from the given text, so your explanation should be formatted as:
{error_format}
The content between [] is the place where you should replace with your analysis. You need to note that only one pair of <Explanation Start> and <Explanation End> can appear in your answer to identify the formatted answer boundary.

[The given text]:
{text}
[Corrected text]:
{label}
[The explanation]:
'''


correct_inquiry_edit_example_explanation = '''For the given text, you should detect a few short segments that are most important for grammatical correctness. And briefly explain the grammatical knowledge involved. 
Here are some examples:

[The given text]:
Also , you 'll meet friendly people who usually ask you to be friends and exchange telephone numbers .
[Important segments]:
<Explanation Start>
<Segment>`ask` at `usually ask you`</Segment>:<Grammar>The subject-verb agreement is correct, as "people" is a plural noun and "ask" is a plural verb in the present tense.</Grammar>
<Segment>`telephone numbers` at `exchange telephone numbers .`</Segment>:<Grammar>Based on the context, "exchange telephone numbers" refers to exchanging phone numbers with friendly people, so using the plural "numbers" is correct.</Grammar>
<Explanation End>

[The given text]:
Aber ich habe oft gesehen , wie viele Leute , vor allem in Südafrika , nach so einem Studium keinen Job finden können .
[Important segments]:
<Explanation Start>
<Segment>`einem` at `so einem Studium`</Segment>:<Grammar>The preposition "nach" indicates "after" in terms of time, and when it is followed by a noun phrase, the dative case should be used. Since "Studium" is a neuter noun, the correct form of the indefinite article in the dative case is "einem".</Grammar>
<Explanation End>

Now it is your turn to do the same detection process for the following text. Follow the above format and do not include additional polite or summarizing remarks. And your response should not exceed 200 words. 
You need to note that only one pair of <Explanation Start> and <Explanation End> can appear in your answer to identify the formatted answer boundary.

[The given text]:
{text}
[Important segments]:
'''

error_inquiry_edit_example_explanation = '''Please provide an explanation for each grammatical error correction needed in the sentence. 
Here are some examples:

[The given text]:
I am applying to this position because I would like to work as organizer .
[Corrected text]:
I am applying for this position because I would like to work as an organizer .
[The explanation]:
<Explanation Start>
<Segment>Replace `to` with `for` at `applying to this`</Segment>:<Grammar>The word "to" is a preposition used to indicate movement or direction, whereas "for" is a preposition used to indicate purpose or intention. In this context, the sentence is describing the reason for applying, so "for" is the correct choice to indicate the purpose of applying for the position.</Grammar>
<Segment>Insert `an` between `as` and `organizer`</Segment>:<Grammar>The word "organizer" starts with a vowel sound, so the indefinite article "an" is used instead of "a". This is a rule in English grammar that applies to words beginning with a vowel sound, such as "an apple" or "an organizer".</Grammar>
<Explanation End>

[The given text]:
Es gibt diesen Paradox , dass man sich befreit indem man Grenzen stellt .
[Corrected text]:
Es gibt dieses Paradox , dass man sich befreit , indem man Grenzen stellt .
[The explanation]:
<Explanation Start>
<Segment>Replace `diesen` with `dieses` at `gibt diesen Paradox`</Segment>:<Grammar>The word `diesen` is a masculine accusative form of the demonstrative pronoun `dies`, but in this context, it should be `dieses` because `Paradox` is a neuter noun. In German, the article agrees with the noun in gender, number, and case. Therefore, the correct form is `dieses`.</Grammar>
<Segment>Insert `,` between `befreit` and `indem`</Segment>:<Grammar>In German, when using the subordinating conjunction `indem` to introduce a subordinate clause, a comma is usually placed before it to separate the main clause from the subordinate clause. This is a common punctuation rule in German to improve sentence clarity and readability.</Grammar>
<Explanation End>

Now it is your turn to do the same explanation process for the following text. Follow the above format and do not include additional polite or summarizing remarks. And your response should not exceed 200 words. We have extract edits from the given text, so your explanation should be formatted as:
{error_format}
The content between [] is the place where you should replace with your analysis. You need to note that only one pair of <Explanation Start> and <Explanation End> can appear in your answer to identify the formatted answer boundary.

[The given text]:
{text}
[Corrected text]:
{label}
[The explanation]:
'''
