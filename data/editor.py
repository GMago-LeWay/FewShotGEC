import spacy
from difflib import SequenceMatcher
import re

SPACY_MODEL_MAP = {
    'en': "en_core_web_sm",
    'de': "de_core_news_sm",
    'ru': "ru_core_news_sm",
    'et': "et_dep_ud_sm",
    'kr': "ko_core_news_sm",
    'zh': "zh_core_web_sm",
    'ro': "ro_core_news_sm",
    'ar': None,
}

LANGUAGE = {
    'wilocness': 'en',
    'conll14': 'en',
    'fce': 'en',
    'nucle': 'en',
    'mucgec': 'zh',
    'falko_merlin': 'de',
    'estgec': 'et',
    'rogec': 'ro',
    'rulec': 'ru',
    'qalb2014': 'ar',
    'kor_union': 'kr',
    'hsk': 'zh',
    'nlpcc18': 'zh'
}

ERROR1 = 0
ERROR2 = 0
ERROR3 = 0

class Editor:
    def __init__(self, dataset_name, edit_mode="complete") -> None:
        self.language = LANGUAGE[dataset_name]
        self.edit_mode = edit_mode
        self.split_char = '' if self.language in ['zh'] else ' '
        self.spacy_model = spacy.load(SPACY_MODEL_MAP[self.language]) if SPACY_MODEL_MAP[self.language] else None

    def _split_sentence(self, text):
        if self.spacy_model:
            return [token.text for token in self.spacy_model(text)]
        else:
            return text.split()

    def get_tokenizer(self):
        return self._split_sentence
    
    def set_edit_mode(self, edit_mode):
        self.edit_mode = edit_mode
        
    def compare_text_by_edit(self, src, tgt):
        if self.edit_mode == "complete":
            return self.compare_text_by_edit_complete(src, tgt)
        elif self.edit_mode == "default":
            return self.compare_text_by_edit_simple(src, tgt)
        else:
            raise NotImplementedError(f"Mode '{self.edit_mode}' for Editor is not implemented")

    def compare_text_by_edit_simple(self, src, tgt):
        src_tokens = self._split_sentence(src)
        tgt_tokens = self._split_sentence(tgt)
        r = SequenceMatcher(None, src_tokens, tgt_tokens)
        diffs = r.get_opcodes()
        edit_diffs = [diff for diff in diffs if diff[0] in ['replace', 'insert', 'delete']]

        edit_descriptions = []
        
        if edit_diffs:
            # compare_result = 'Correction Process: '
            for edit in edit_diffs:
                _, i1, i2, j1, j2 = edit
                if i1==i2:
                    # insert
                    insert_content = self.split_char.join(tgt_tokens[j1: j2])
                    if i1==0:
                        description = f"Insert `{insert_content}` at start"
                        edit_descriptions.append(description)
                    elif i2==len(src_tokens):
                        description = f"Insert `{insert_content}` at the end"
                        edit_descriptions.append(description)
                    else:
                        description = f"Insert `{insert_content}` between `{src_tokens[i1-1]}` and `{src_tokens[i2]}`"
                        edit_descriptions.append(description)
                elif j1==j2:
                    # delete
                    delete_content = self.split_char.join(src_tokens[i1: i2])
                    description = f"Delete `{delete_content}`"
                    edit_descriptions.append(description)
                else:
                    # replace
                    replace_content_src, replace_content_tgt = self.split_char.join(src_tokens[i1: i2]), self.split_char.join(tgt_tokens[j1: j2])
                    description = f"Replace `{replace_content_src}` with `{replace_content_tgt}`"
                    edit_descriptions.append(description)
        else:
            description = 'No error found.'

        return edit_descriptions


    def compare_text_by_edit_complete(self, src, tgt):
        src_tokens = self._split_sentence(src)
        tgt_tokens = self._split_sentence(tgt)
        r = SequenceMatcher(None, src_tokens, tgt_tokens)
        diffs = r.get_opcodes()
        edit_diffs = [diff for diff in diffs if diff[0] in ['replace', 'insert', 'delete']]

        edit_descriptions = []
        
        if edit_diffs:
            # compare_result = 'Correction Process: '
            for edit in edit_diffs:
                _, i1, i2, j1, j2 = edit
                if i1==i2:
                    # insert
                    insert_content = self.split_char.join(tgt_tokens[j1: j2])
                    if i1==0:
                        description = f"Insert `{insert_content}` before `{src_tokens[i1]}`"
                        edit_descriptions.append(description)
                    elif i2==len(src_tokens):
                        description = f"Insert `{insert_content}` after `{src_tokens[i2-1]}`"
                        edit_descriptions.append(description)
                    else:
                        description = f"Insert `{insert_content}` between `{src_tokens[i1-1]}` and `{src_tokens[i2]}`"
                        edit_descriptions.append(description)
                elif j1==j2:
                    # delete
                    delete_content = self.split_char.join(src_tokens[i1: i2])
                    delete_content_at = self.split_char.join(src_tokens[max(i1-1, 0): i2+1])
                    description = f"Delete `{delete_content}` at `{delete_content_at}`"
                    edit_descriptions.append(description)
                else:
                    # replace
                    replace_content_src, replace_content_tgt = self.split_char.join(src_tokens[i1: i2]), self.split_char.join(tgt_tokens[j1: j2])
                    replace_content_at = self.split_char.join(src_tokens[max(i1-1, 0): i2+1])
                    description = f"Replace `{replace_content_src}` with `{replace_content_tgt}` at `{replace_content_at}`"
                    edit_descriptions.append(description)
        else:
            description = 'No error found.'

        return edit_descriptions
    

class ExplanationsParser:
    def __init__(self, templator, editor) -> None:
        self.templator = templator
        self.editor = editor
        self.editor.set_edit_mode(self.templator.get_edit_extract_mode())

        # format markers
        self.edit_mode = self.templator.template["edit_mode"]
        self.edit_start = self.templator.template["edit_start"]
        self.edit_end = self.templator.template["edit_end"]
        self.sep = self.templator.template["sep"]
        self.grammar_start = self.templator.template["grammar_start"]
        self.grammar_end = self.templator.template["grammar_end"]

    def parse_response(self, text):
        # 构建正则表达式模式
        pattern = rf'{self.edit_start}(.*?){self.edit_end}.*?{self.sep}.*?{self.grammar_start}(.*?){self.grammar_end}'
        
        # mode1: normal
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            result = [{"edit": match[0].strip(), "grammar": match[1].strip()} for match in matches]
            return result
        

        # mode 2 : without edit_start and edit_end
        pattern = rf'(.*?){self.sep}.*?{self.grammar_start}(.*?){self.grammar_end}'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            result = [{"edit": match[0].strip(), "grammar": match[1].strip()} for match in matches]
            return result
        
        # mode 3: without grammar_start and grammar_end
        pattern = rf'{self.edit_start}(.*?){self.edit_end}.*?{self.sep}(.*?)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            result = [{"edit": match[0].strip(), "grammar": match[1].strip()} for match in matches]
            return result

        # mode 4 : without edit_start and edit_end and sep
        pattern = rf'(.*?){self.grammar_start}(.*?){self.grammar_end}'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            result = [{"edit": match[0].strip(), "grammar": match[1].strip()} for match in matches]
            return result
        
        # mode 5: without grammar_start and grammar_end and sep
        pattern = rf'{self.edit_start}(.*?){self.edit_end}(.*?)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            result = [{"edit": match[0].strip(), "grammar": match[1].strip()} for match in matches]
            return result
        
        return []


    @staticmethod
    def detect_repetitive_substrings(s, threshold=5):
        n = len(s)
        # 滑动窗口的最大长度为字符串长度的一半
        max_length = n // 2
        
        for length in range(1, max_length + 1):  # 子串长度
            repeat_count = 1  # 初始化重复计数器
            for i in range(0, n - length, length):  # 步长为子串长度
                if s[i:i+length] == s[i+length:i+2*length]:
                    repeat_count += 1
                    if repeat_count > threshold:
                        return True
                else:
                    repeat_count = 1  # 重置计数器
                    
        return False
    

    def filter_explanation(self, error_list, explanations):
        global ERROR1, ERROR2, ERROR3
        # for explanation in explanations:
        #     if ExplanationsParser.detect_repetitive_substrings(explanation["grammar"]):
        #         print(explanation["grammar"])
        if len(error_list) != len(explanations):
            # ERROR1 += 1
            return []
        filtered_explanations = []
        for error, explanation in zip(error_list, explanations):
            if error.replace(' ', '') == explanation["edit"].replace(' ', ''):
                if ExplanationsParser.detect_repetitive_substrings(explanation["grammar"]):
                    # ERROR3 += 1
                    return []
                else:
                    filtered_explanations.append(explanation)
            else:
                # ERROR2 += 1
                return []
        return filtered_explanations


    def initial_parse(self, data_item, response):
        error_list = self.editor.compare_text_by_edit(data_item['text'], data_item['label'])
        explanations = self.parse_response(response)
        clean_explanations = self.filter_explanation(error_list=error_list, explanations=explanations)
        # print(ERROR1, ERROR2, ERROR3)
        return clean_explanations
