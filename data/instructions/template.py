import os
from ..editor import Editor
from .explanation_template import *
from .detection_template import *

TEMPLATES = {
}

EXPLANATION_TEMPLATES = {
    'detailed_explanation_prompt': {
        "system": "You are tasked with performing a comprehensive grammatical error analysis on the following text, which may contain errors in various languages.",
        "correct_inquiry": gpt4_explanation_inquiry,
        "correct_format": "No error found.",
        "error_inquiry": gpt4_explanation_inquiry,
        "error_format": "{error_units}.",
        "error_unit": "{edit}",
        "error_unit_separator": "; ",
        "answer_start": "",
        "answer_end": "",
        "edit_mode": "default",
    },
    'standard_explanation': {
        "system": "You, a language expert, can briefly explain how to judge a sentence is grammatically correct and why some corrections are essential.",
        "correct_inquiry": "For the given text:\n{text}\nCorrected text:\n{label}\nList a few short segments that are most important for grammatical correctness, no more than 5. And briefly explain the grammatical knowledge involved. You should directly response following the format as:\n{correct_format}\nThe content between [] is the place where you should replace with your analysis. You need to note that only one pair of <Explanation Start> and <Explanation End> can appear in your answer to identify the formatted answer boundary.",
        "correct_format": "<Explanation Start>\nShort segment here: [a brief explanation of its correctness and the brief relevant grammar knowledege here.]\n...\n<Explanation End>",
        "error_inquiry": "For the given text:\n{text}\nCorrected text:\n{label}\nPlease briefly tell me why you did these corrections one by one, including the relevant grammar knowledge if possible. You should directly response following the format as:\n{error_format}\nThe content between [] is the place where you should replace with your analysis. You need to note that only one pair of <Explanation Start> and <Explanation End> can appear in your answer to identify the formatted answer boundary.",
        "error_format": "<Explanation Start>\n{error_units}\n<Explanation End>",
        "error_unit": "{edit}: [The simple reason and the brief relevant grammar knowledege here.]",
        "error_unit_separator": "\n",
        "answer_start": "<Explanation Start>",
        "answer_end": "<Explanation End>",
        "edit_mode": "default",
    },
    'example_explanation': {
        "system": "You, a language expert, can briefly explain how to judge a sentence is grammatically correct and why some corrections are essential.",
        "correct_inquiry": correct_inquiry_example_explanation,
        "correct_format": "<Explanation Start>\nShort segment here: [a brief explanation of its correctness and the brief relevant grammar knowledege here.]\n...\n<Explanation End>",
        "error_inquiry": error_inquiry_example_explanation,
        "error_format": "<Explanation Start>\n{error_units}\n<Explanation End>",
        "error_unit": "{edit}: [The simple reason and the brief relevant grammar knowledege here.]",
        "error_unit_separator": "\n",
        "answer_start": "<Explanation Start>",
        "answer_end": "<Explanation End>",
        "edit_mode": "default",
    },
    'edit_example_explanation': {
        "system": "You are a multilingual grammar expert proficient in correcting grammatical errors in sentences and explaining the reasons for these corrections from a grammatical perspective.",
        "correct_inquiry": correct_inquiry_edit_example_explanation,
        "correct_format": "<Explanation Start>\nShort segment here: [a brief explanation of its correctness and the brief relevant grammar knowledege here.]\n...\n<Explanation End>",
        "error_inquiry": error_inquiry_edit_example_explanation,
        "error_format": "<Explanation Start>\n{error_units}\n<Explanation End>",
        "error_unit": "<Segment>{edit}</Segment>:<Grammar>[The simple reason and the brief relevant grammar knowledege here.]</Grammar>",
        "error_unit_separator": "\n",
        "answer_start": "<Explanation Start>",
        "answer_end": "<Explanation End>",
        "edit_start": "<Segment>",
        "edit_end": "</Segment>",
        "sep": ":",
        "grammar_start": "<Grammar>",
        "grammar_end": "</Grammar>",
        "edit_mode": "complete",
    },
    'edit_simple_standard_explanation': {
        "system": "You are a multilingual grammar expert proficient in correcting grammatical errors in sentences and explaining the reasons for these corrections from a grammatical perspective.",
        "correct_inquiry": correct_inquiry_default,
        "correct_format": "<Explanation Start>\nShort segment here: [a brief explanation of its correctness and the brief relevant grammar knowledege here.]\n...\n<Explanation End>",
        "error_inquiry": error_inquiry_default,
        "error_format": "<Explanation Start>\n{error_units}\n<Explanation End>",
        "error_unit": "<Segment>{edit}</Segment>:<Grammar>[The simple reason and the brief relevant grammar knowledege here.]</Grammar>",
        "error_unit_separator": "\n",
        "answer_start": "<Explanation Start>",
        "answer_end": "<Explanation End>",
        "edit_start": "<Segment>",
        "edit_end": "</Segment>",
        "sep": ":",
        "grammar_start": "<Grammar>",
        "grammar_end": "</Grammar>",
        "edit_mode": "default",
    },
    'edit_standard_explanation': {
        "system": "You are a multilingual grammar expert proficient in correcting grammatical errors in sentences and explaining the reasons for these corrections from a grammatical perspective.",
        "correct_inquiry": correct_inquiry_default,
        "correct_format": "<Explanation Start>\nShort segment here: [a brief explanation of its correctness and the brief relevant grammar knowledege here.]\n...\n<Explanation End>",
        "error_inquiry": error_inquiry_default,
        "error_format": "<Explanation Start>\n{error_units}\n<Explanation End>",
        "error_unit": "<Segment>{edit}</Segment>:<Grammar>[The simple reason and the brief relevant grammar knowledege here.]</Grammar>",
        "error_unit_separator": "\n",
        "answer_start": "<Explanation Start>",
        "answer_end": "<Explanation End>",
        "edit_start": "<Segment>",
        "edit_end": "</Segment>",
        "sep": ":",
        "grammar_start": "<Grammar>",
        "grammar_end": "</Grammar>",
        "edit_mode": "complete",
    },
    'edit_standard_explanation_unlimited': {
        "system": "You are a multilingual grammar expert proficient in correcting grammatical errors in sentences and explaining the reasons for these corrections from a grammatical perspective.",
        "correct_inquiry": correct_inquiry_unlimited,
        "correct_format": "<Explanation Start>\nShort segment here: [a brief explanation of its correctness and the brief relevant grammar knowledege here.]\n...\n<Explanation End>",
        "error_inquiry": error_inquiry_unlimited,
        "error_format": "<Explanation Start>\n{error_units}\n<Explanation End>",
        "error_unit": "<Segment>{edit}</Segment>:<Grammar>[The simple reason and the brief relevant grammar knowledege here.]</Grammar>",
        "error_unit_separator": "\n",
        "answer_start": "<Explanation Start>",
        "answer_end": "<Explanation End>",
        "edit_start": "<Segment>",
        "edit_end": "</Segment>",
        "sep": ":",
        "grammar_start": "<Grammar>",
        "grammar_end": "</Grammar>",
        "edit_mode": "complete",
    },
}

ICL_TEMPLATES = {
    'reproduce': {
        'system': "",
        'prompt': "There is an erroneous sentence between `<erroneous sentence>` and `</erroneous sentence>`. Then grammatical errors in the erroneous sentence will be corrected. The corrected version will be between `<corrected sentence>` and `</corrected sentence>`.\n{icl_examples}<erroneous sentence>{source}</erroneous sentence>\n<corrected sentence>",
        'icl_example': "<erroneous sentence>{source}</erroneous sentence>\n<corrected sentence>{target}</corrected sentence>\n",
        'answer_start': "<corrected sentence>",
        'answer_end': "</corrected sentence>",
    },
    'min_edit_zeroshot': {
        'system': "You are an language expert who is responsible for grammatical, lexical and orthographic error corrections given an input sentence. Your job is to fix grammatical mistakes, awkward phrases, spelling errors, etc. following standard written usage conventions, but your corrections must be conservative. Please keep the original sentence (words, phrases, and structure) as much as possible. The ultimate goal of this task is to make the given sentence sound natural to native speakers without making unnecessary changes. Corrections are not required when the sentence is already grammatical and sounds natural.",
        'prompt': "Here is the input sentence containing errors that needs to be corrected.\n```{source}```\nPlease give the corrected sentence directly, which should be surrounded by ``` and ```:",
        'answer_start': "",
        'answer_end': "",
    },
    'min_edit_fewshot': {
        'system': "You are an language expert who is responsible for grammatical, lexical and orthographic error corrections given an input sentence. Your job is to fix grammatical mistakes, awkward phrases, spelling errors, etc. following standard written usage conventions, but your corrections must be conservative. Please keep the original sentence (words, phrases, and structure) as much as possible. The ultimate goal of this task is to make the given sentence sound natural to native speakers without making unnecessary changes. Corrections are not required when the sentence is already grammatical and sounds natural.",
        'prompt': "There is an erroneous sentence between `<erroneous sentence>` and `</erroneous sentence>`. Then grammatical errors in the erroneous sentence will be corrected. The corrected version will be between `<corrected sentence>` and `</corrected sentence>`.\n{icl_examples}<erroneous sentence>{source}</erroneous sentence>\n<corrected sentence>",
        'icl_example': "<erroneous sentence>{source}</erroneous sentence>\n<corrected sentence>{target}</corrected sentence>\n",
        'answer_start': "<corrected sentence>",
        'answer_end': "</corrected sentence>",
    },
    'min_edit_fewshot_cot': {
        'system': "You are an language expert who is responsible for grammatical, lexical and orthographic error corrections given an input sentence. Your job is to fix grammatical mistakes, awkward phrases, spelling errors, etc. following standard written usage conventions, but your corrections must be conservative. Please keep the original sentence (words, phrases, and structure) as much as possible. The ultimate goal of this task is to make the given sentence sound natural to native speakers without making unnecessary changes. Corrections are not required when the sentence is already grammatical and sounds natural.",
        'prompt': "There is an erroneous sentence between `<erroneous sentence>` and `</erroneous sentence>`. Then grammatical errors in the erroneous sentence will be corrected. The corrected version will be between `<corrected sentence>` and `</corrected sentence>`.\n{icl_examples}<erroneous sentence>{source}</erroneous sentence>\n<analysis>\n{description}\n</analysis>\n<corrected sentence>",
        'icl_example': "<erroneous sentence>{source}</erroneous sentence>\n<analysis>\n{description}\n</analysis>\n<corrected sentence>{target}</corrected sentence>\n\n",
        'answer_start': "<corrected sentence>",
        'answer_end': "</corrected sentence>",
    },
    'min_edit_fewshot_augmented': {
        'system': "You are an language expert who is responsible for grammatical, lexical and orthographic error corrections given an input sentence. Your job is to fix grammatical mistakes, awkward phrases, spelling errors, etc. following standard written usage conventions, but your corrections must be conservative. Please keep the original sentence (words, phrases, and structure) as much as possible. The ultimate goal of this task is to make the given sentence sound natural to native speakers without making unnecessary changes. Corrections are not required when the sentence is already grammatical and sounds natural.",
        'prompt': "There is an erroneous sentence between `<erroneous sentence>` and `</erroneous sentence>`. Then grammatical errors in the erroneous sentence will be corrected. The corrected version will be between `<corrected sentence>` and `</corrected sentence>`.\nPlease try to modify the original sentence as little as possible.\n{icl_examples}<erroneous sentence>{source}</erroneous sentence>\n<corrected sentence>",
        'icl_example': "<erroneous sentence>{source}</erroneous sentence>\n<corrected sentence>{target}</corrected sentence>\n",
        'answer_start': "<corrected sentence>",
        'answer_end': "</corrected sentence>",
    },
    'min_edit_fewshot_description': {
        'system': "You are an language expert who is responsible for grammatical, lexical and orthographic error corrections given an input sentence. Your job is to fix grammatical mistakes, awkward phrases, spelling errors, etc. following standard written usage conventions, but your corrections must be conservative. Please keep the original sentence (words, phrases, and structure) as much as possible. The ultimate goal of this task is to make the given sentence sound natural to native speakers without making unnecessary changes. Corrections are not required when the sentence is already grammatical and sounds natural.",
        'prompt': "There is an erroneous sentence between `<erroneous sentence>` and `</erroneous sentence>`. Then grammatical errors in the erroneous sentence will be corrected. The corrected version will be between `<corrected sentence>` and `</corrected sentence>`.\n{icl_examples}<erroneous sentence>{source}</erroneous sentence>\n<corrected sentence>",
        'icl_example': "<erroneous sentence>{source}</erroneous sentence>\n<corrected sentence>{target}</corrected sentence>\n<description>\n{description}\n</description>\n\n",
        'answer_start': "<corrected sentence>",
        'answer_end': "</corrected sentence>",
    },
    'min_edit_fewshot2': {
        'system': "You are an language expert who is responsible for grammatical, lexical and orthographic error corrections given an input sentence. Your job is to fix grammatical mistakes, awkward phrases, spelling errors, etc. following standard written usage conventions, but your corrections must be conservative. Please keep the original sentence (words, phrases, and structure) as much as possible. The ultimate goal of this task is to make the given sentence sound natural to native speakers without making unnecessary changes. Corrections are not required when the sentence is already grammatical and sounds natural.",
        'prompt': "There is an erroneous sentence between `<error>` and `</error>`. Then grammatical errors in the erroneous sentence will be corrected. The corrected version will be between `<correct>` and `</correct>`.\n{icl_examples}<error>{source}</error>\n<correct>",
        'icl_example': "<error>{source}</error>\n<correct>{target}</correct>\n",
        'answer_start': "<correct>",
        'answer_end': "</correct>",
    },
    'raw_detection': {
        'system': "You, a language expert, can briefly judge if a sentence is grammatically correct and find the specific segments in the sentence which may violate the grammar.",
        'prompt': "For the given text:\n```{source}```\nList the text segments that should be focused on and explain if they are grammatically correct:",
        'answer_start': "",
        'answer_end': "",
    },
    'detailed_detection_prompt': {
        'system': "You are tasked with performing a comprehensive grammatical error analysis on the following text, which may contain errors in various languages.",
        'prompt': detection_prompt_detailed,
        'answer_start': "",
        'answer_end': "",
    },
    'short_detection_prompt': {
        'system': "You are tasked with performing a comprehensive grammatical error analysis on the following text, which may contain errors in various languages.",
        'prompt': detection_prompt_short,
        'answer_start': "",
        'answer_end': "",
    },
}


class PromptTemplate:
    def __init__(self, template) -> None:
        assert template in TEMPLATES, f"{template} not supported."
        self.template = TEMPLATES[template]
    
    def get_answer_start(self):
        return self.template['answer_start']
    
    def get_answer_end(self):
        return self.template['answer_end']

    def format(self, **kwargs):
        final_instruction = self.template['prompt'].format(**kwargs)
        return self.template['system'], final_instruction

    def postprocess(self, output_str):
        if self.template["answer_start"]:
            if output_str.find(self.template["answer_start"]) != -1:
                output_str = output_str.split(self.template["answer_start"], 1)[1]
        if self.template["answer_end"]:
            if output_str.find(self.template["answer_end"]) != -1:
                output_str = output_str.split(self.template["answer_end"], 1)[0]
        return output_str


class ICLPromptTemplate(PromptTemplate):
    def __init__(self, template) -> None:
        assert template in ICL_TEMPLATES, f"{template} not supported."
        self.template = ICL_TEMPLATES[template]
    
    def format(self, examples_list, **kwargs):
        icl_examples = ''
        for example in examples_list:
            example_str = self.template['icl_example'].format(**example)
            icl_examples += example_str
        kwargs['icl_examples'] = icl_examples
        final_instruction = self.template['prompt'].format(**kwargs)
        return self.template['system'], final_instruction


class ExplainationTemplate(PromptTemplate):
    def __init__(self, template) -> None:
        assert template in EXPLANATION_TEMPLATES, f"{template} not supported for explaination template."
        self.template = EXPLANATION_TEMPLATES[template]
    
    def format(self, data_item, error_list):
        src = data_item['text']
        hypo = data_item['hypothesis']
        if len(error_list) == 0:
            instruction = self.template['correct_inquiry'].format(
                text=src,
                label=hypo,
                correct_format=self.template['correct_format'],
            )
            return self.template['system'], instruction
        else:
            error_unit_list = [self.template['error_unit'].format(edit=edit) for edit in error_list]
            error_units = self.template['error_unit_separator'].join(error_unit_list)
            error_format = self.template['error_format'].format(error_units=error_units)
            instruction = self.template['error_inquiry'].format(
                text=src,
                label=hypo,
                error_format=error_format,
            )
            return self.template['system'], instruction

    def get_edit_extract_mode(self):
        return self.template['edit_mode']

    def explanation_structure(self):
        '''
        return : str   
        "structured" will means edit and explanations are structured and ca be parsed by ExplanationParser
        "default" means there are not special structure for the edits
        '''
        if "edit_start" in self.template and "edit_end" in self.template:
            return "structured"
            
        return "default"


class SuperPrompt(PromptTemplate):
    def __init__(self, template_name) -> None:
        self.template_name = template_name
        if template_name in ICL_TEMPLATES:
            self.templator = ICLPromptTemplate(template_name)
        elif template_name in EXPLANATION_TEMPLATES:
            self.templator = ExplainationTemplate(template_name)
        else:
            self.templator = PromptTemplate(template_name)
        self.template = self.templator.template
        self.editor: Editor = None
    
    def set_editor(self, dataset_name):
        self.editor: Editor = Editor(dataset_name=dataset_name, edit_mode=self.get_edit_extract_mode())

    def format(self, data_item, **kwargs):
        if self.template_name in ICL_TEMPLATES:
            assert 'examples' in kwargs
            if 'description' in data_item:
                return self.templator.format(examples_list=kwargs['examples'], source=data_item['text'], description=data_item['description'])
            else:
                return self.templator.format(examples_list=kwargs['examples'], source=data_item['text'])
        elif self.template_name in EXPLANATION_TEMPLATES:
            # explanation template requires error (edit) list, if there is no error list, we need to extract it from hypothesis and source.
            assert 'error_list' in kwargs or 'hypothesis' in data_item
            if 'error_list' in kwargs:
                return self.templator.format(data_item=data_item, error_list=kwargs['error_list'])
            else:
                assert self.editor is not None, "Please set editor first because we need to extract edit."
                error_list = self.editor.compare_text_by_edit(data_item['text'], data_item['hypothesis'])
                return self.templator.format(data_item=data_item, error_list=error_list)
        else:
            # normal template
            return self.templator.format(**data_item)
        
    def get_edit_extract_mode(self):
        assert 'edit_mode' in self.templator.template, f"Current template {self.template_name} is not an ExplanationTemplate which has an edit mode required."
        return self.templator.template['edit_mode']
    
    def explanation_structure(self):
        '''
        return : str   
        "structured" will means edit and explanations are structured and ca be parsed by ExplanationParser
        "default" means there are not special structure for the edits
        '''
        if self.template_name in EXPLANATION_TEMPLATES:
            if "edit_start" in self.template and "edit_end" in self.template:
                return "structured"
            
        return "default"
