import json
import os
import logging
from tqdm import tqdm
import zipfile
import codecs
# from nltk import word_tokenize
import re
import codecs
from .ChERRANT.main import compute_cherrant_with_ref_file, compute_cherrant
from configs.config import DATA_ROOT_DIR
import spacy
import nltk

en = spacy.load("en_core_web_sm")

logger = logging.getLogger(__name__)

# conda create --name m2 python==2.7.18
PYTHON2_PATH = '/data/xxx/anaconda3/envs/m2/bin/python'
# PYTHON2_PATH = '/Users/yyy12/anaconda3/envs/m2/bin/python'

NLPCC18_M2_FILE = "evaluators/ChERRANT/samples/nlpcc2018_official.ref.m2.char"

BEA_DEV_M2_FILE = os.path.join(DATA_ROOT_DIR, 'WILocness/wi+locness/m2/ABCN.dev.gold.bea19.m2')
MUCGEC_DEV_M2_FILE = os.path.join(DATA_ROOT_DIR, 'MuCGEC/MuCGEC_dev/valid.gold.m2.char')
FCGEC_DEV_FILE = os.path.join(DATA_ROOT_DIR, 'FCGEC/FCGEC_dev/test.json')

def m2score(model_output, m2_file, result_output):
    os.system(f"{PYTHON2_PATH} evaluators/m2scorer/scripts/m2scorer.py {model_output} {m2_file} >> {result_output}")

CN_MARKER_MAP = {
    ',': '，',
    ';': '；',
    ':': '：',
    '(': '（',
    ')': '）',
    '?': '？',
    '!': '！',
}

RETOKENIZATION_RULES = [
    # Remove extra space around single quotes, hyphens, and slashes.
    (" ' (.*?) ' ", " '\\1' "),
    (" - ", "-"),
    (" / ", "/"),
    # Ensure there are spaces around parentheses and brackets.
    (r"([\]\[\(\){}<>])", " \\1 "),
    (r"\s+", " "),
]


class PostProcessManipulator:
    cn_marker = 'cn_marker'
    mucgec_eval = 'mucgec_dev_eval'
    fcgec_eval = 'fcgec_dev_eval'
    bea_eval = 'bea_eval'
    merge_sample = 'merge_sample'
    en_test = 'en_test'
    en_test_py3 = 'en_test_py3'
    conll14_bea = 'conll14_bea'
    de_tokenize = 'de_tokenize'
    ru_tokenize = 'ru_tokenize'
    et_tokenize = 'et_tokenize'
    ro_tokenize = 'ro_tokenize'
    replace_linebreaker = 'replace_\n'
    conll14 = 'conll14'
    nlpcc18 = 'nlpcc18'
    bea19 = 'bea19'


class PostProcess:
    def __init__(self, args) -> None:
        '''
        Post process after model generated json-like result list. Must be run in infer mode.
        json_results: List[Dict], like [{'id': str or num, 'src': str, 'predict': str, ('tgt': str)}]
        save_dir: save directory name inside the args.save_dir
        '''
        self.args = args
        self.save_dir = args.dir
        self.results_file = os.path.join(self.save_dir, 'predictions.jsonl')
        self.results = [json.loads(item.strip()) for item in open(self.results_file).readlines()]
        # assert 'infer' in args.task_mode

        # set save directory
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self._temp_key_transform()

        # recognize post process
        self._post_process_identification()

        self.post_process_func_map = {
            PostProcessManipulator.cn_marker: self._chinese_marker_substitute,
            PostProcessManipulator.merge_sample: self._merge_split_test_sample,
            PostProcessManipulator.en_test_py3: self._en_conll_bea_postprocess_py3,
            PostProcessManipulator.mucgec_eval: self._mucgec_dev_evaluation,
            PostProcessManipulator.bea_eval: self._bea_dev_evaluation,
            PostProcessManipulator.fcgec_eval: self._fcgec_dev_evaluation,
            PostProcessManipulator.replace_linebreaker: self._replace_linebreaker,
            PostProcessManipulator.de_tokenize: self._de_retokenize,
            PostProcessManipulator.ru_tokenize: self._ru_retokenize,
            PostProcessManipulator.et_tokenize: self._et_retokenize,
            PostProcessManipulator.ro_tokenize: self._ro_retokenize,
            PostProcessManipulator.conll14_bea: self._conll14_bea, 
            PostProcessManipulator.conll14: self._conll14, 
            PostProcessManipulator.bea19: self._en_bea_postprocess_py3,
            PostProcessManipulator.nlpcc18: self._nlpcc18_evaluate,
            # 'spacy_retokenize': self._retokenize,
        }

        self.allowed_dataset = {
            PostProcessManipulator.cn_marker: ['mucgec', 'fcgec', 'pretrain', 'fangzhenggrammar', 'fangzhengspell', 'mucgec_dev', 'fcgec_dev'],
            PostProcessManipulator.merge_sample: ['mucgec', 'fcgec', 'pretrain', 'fangzhenggrammar', 'fangzhengspell', 'c4', 'lang8', 'clang8', 'nucle', 'wilocness', 'hybrid'],
            PostProcessManipulator.en_test_py3: ['c4', 'lang8', 'clang8', 'nucle', 'wilocness', 'hybrid'],
            PostProcessManipulator.conll14_bea: ['c4', 'lang8', 'clang8', 'nucle', 'wilocness', 'hybrid'],
            PostProcessManipulator.mucgec_eval: ['mucgec_dev'],
            PostProcessManipulator.bea_eval: ['bea_dev'],
            PostProcessManipulator.fcgec_eval: ['fcgec_dev'],
            PostProcessManipulator.de_tokenize: ['falko_merlin'],
            PostProcessManipulator.ru_tokenize: ['rulec'],
            PostProcessManipulator.et_tokenize: ['estgec'],
            PostProcessManipulator.ro_tokenize: ['rogec'],
            PostProcessManipulator.conll14: ['conll14'],
            PostProcessManipulator.bea19: ['bea19'],
            PostProcessManipulator.nlpcc18: ['nlpcc18'],
            PostProcessManipulator.replace_linebreaker: [],
        }

    def _temp_key_transform(self):
        old_standard_results = []
        for item in self.results:
            old_standard_results_item = {"id": item["id"], "src": item["text"], "predict": item["prediction"]}
            if "label" in item:
                old_standard_results_item["tgt"] = item["label"]
            old_standard_results.append(old_standard_results_item)
        self.results = old_standard_results

    def _post_process_identification(self):
        if self.args.dataset == "mucgec":
            self.post_process = [PostProcessManipulator.replace_linebreaker]
        elif self.args.dataset == "wilocness":
            self.post_process = [PostProcessManipulator.replace_linebreaker, PostProcessManipulator.en_test_py3]
        elif self.args.dataset == "falko_merlin":
            self.post_process = [PostProcessManipulator.replace_linebreaker, PostProcessManipulator.de_tokenize]
        elif self.args.dataset == "rulec":
            self.post_process = [PostProcessManipulator.replace_linebreaker, PostProcessManipulator.ru_tokenize]
        elif self.args.dataset == "estgec":
            self.post_process = [PostProcessManipulator.replace_linebreaker, PostProcessManipulator.et_tokenize]
        elif self.args.dataset == "rogec":
            self.post_process = [PostProcessManipulator.replace_linebreaker, PostProcessManipulator.ro_tokenize]
        elif self.args.dataset == "kor_union":
            self.post_process = [PostProcessManipulator.replace_linebreaker]
        elif self.args.dataset == "qalb2014":
            self.post_process = [PostProcessManipulator.replace_linebreaker]
        elif self.args.dataset == "conll14":
            self.post_process = [PostProcessManipulator.replace_linebreaker, PostProcessManipulator.conll14]
        elif self.args.dataset == "bea19":
            self.post_process = [PostProcessManipulator.replace_linebreaker, PostProcessManipulator.bea19]
        elif self.args.dataset == "nlpcc18":
            self.post_process = [PostProcessManipulator.replace_linebreaker, PostProcessManipulator.nlpcc18]
        elif self.args.dataset == "fcgec":
            self.post_process = [PostProcessManipulator.replace_linebreaker]
        else:
            raise NotImplementedError()
    
    @staticmethod
    def gec_reform_answer(string):
        contractions = ["n't", "'d", "'ll", "'m", "'re", "'s", "'ve"]
        tokens = string.split()
        for i in range(len(tokens)):
            if tokens[i] in contractions:
                tokens[i-1] = tokens[i-1] + tokens[i]
                tokens[i] = ''
        tokens = [token for token in tokens if token != '']
        answer = ' '.join(tokens)
        return answer


    def _chinese_marker_substitute(self):
        for i in range(len(self.results)):
            for key in CN_MARKER_MAP:
                self.results[i]["predict"] = self.results[i]["predict"].replace(key, CN_MARKER_MAP[key])

    def _merge_split_test_sample(self):
        merged_results = []
        assert self.results, "Result Null"
        discourse_index = self.results[0]["id"].split('#')[0]
        source_discourse_buff = ""
        target_discourse_buff = ""
        last_item = None
        cur_item = None
        for item in self.results:
            cur_item = item
            line = item["predict"]
            line = line.strip()
            cur_index, _, end = item["id"].split('#')
            end = end.strip()
            if cur_index == discourse_index:
                source_discourse_buff += item["src"]
                target_discourse_buff += line
                last_item = item
            else:
                if "tgt" in item:
                    merged_results.append({"id": discourse_index, "src": source_discourse_buff, "tgt": last_item["tgt"], "predict": target_discourse_buff})
                else:
                    merged_results.append({"id": discourse_index, "src": source_discourse_buff, "predict": target_discourse_buff})
                discourse_index = cur_index
                source_discourse_buff = item["src"]
                target_discourse_buff = line
            if end != 'P':
                target_discourse_buff = target_discourse_buff[:-1] + end
        else:
            if "tgt" in item:
                merged_results.append({"id": discourse_index, "src": source_discourse_buff, "tgt": last_item["tgt"], "predict": target_discourse_buff})
            else:
                merged_results.append({"id": discourse_index, "src": source_discourse_buff, "predict": target_discourse_buff})
        
        logger.info(f"Results length before merged: {len(self.results)}; After merged: {len(merged_results)}")
        self.results = merged_results

    
    @staticmethod
    def conll_postprocess(item):
        global RETOKENIZATION_RULES, en

        item["oripred"] = str(item["predict"])
        line = str(item["predict"])
        line = re.sub(" '\s?((?:m )|(?:ve )|(?:ll )|(?:s )|(?:d ))",
                        "'\\1", line)
        line = " ".join([t.text for t in en.tokenizer(line)])
        # fix tokenization issues for CoNLL
        for rule in RETOKENIZATION_RULES:
            line = re.sub(rule[0], rule[1], line)
        item["predict"] = line
        return item
    
    @staticmethod
    def bea_postprocess(item):
        global en
        item["oripred"] = str(item["predict"])
        line = str(item["predict"])
        line = re.sub(" '\s?((?:m )|(?:ve )|(?:ll )|(?:s )|(?:d ))",
                        "'\\1", line)
        line = " ".join([t.text for t in en.tokenizer(line)])
        # in spaCy v1.9.0 and the en_core_web_sm-1.2.0 model
        # 80% -> 80%, but in newest ver. 2.3.9', 80% -> 80 %
        # haven't -> haven't, but in newest ver. 2.3.9', haven't -> have n't
        line = re.sub("(?<=\d)\s+%", "%", line)
        line = re.sub("((?:have)|(?:has)) n't", "\\1n't", line)
        line = re.sub("^-", "- ", line)
        line = re.sub(r"\s+", " ", line)
        item["predict"] = line
        return item
    
    def _conll14(self):
        for i in range(len(self.results)):
            self.results[i] = PostProcess.conll_postprocess(self.results[i])
        # conll14 result save
        conll14_file_name = 'conll14.txt'
        conll14_file_path = os.path.join(self.save_dir, conll14_file_name)
        with open(conll14_file_path, 'w') as f:
            for item in self.results:
                f.write(item["predict"] + '\n')
        


    def _en_conll_bea_postprocess_py3(self):
        last_number = -1
        bea19_start_index = None
        test_data = {'conll14': [], 'bea19': []}
        current_dataset = 'conll14'
        logger.info("Postprocessing CoNLL14 test data...")
        for i in range(len(self.results)):
            assert 'conll14' in self.results[i]["id"] or 'bea19' in self.results[i]["id"], "Current test set is not the concatenation of CoNLL14 and BEA19."
            # check current result item belongs to which test set
            number, data_source = self.results[i]["id"].split('_')
            if data_source != current_dataset:
                last_number = -1
                bea19_start_index = i
                current_dataset = 'bea19'
                logger.info("Postprocessing BEA19 test data...")
            assert eval(number) == last_number + 1
            last_number = eval(number)
            assert data_source in test_data
            if current_dataset == 'conll14':
                self.results[i] = PostProcess.conll_postprocess(self.results[i])
            else:
                self.results[i] = PostProcess.bea_postprocess(self.results[i])
            test_data[data_source].append(self.results[i])

        CONLL14_NUM, BEA19_NUM = 1312, 4477
        assert len(test_data["bea19"]) == BEA19_NUM and len(test_data['conll14']) == CONLL14_NUM

        # save file for further evaluation
        # conll14 evaluation
        conll14_file_name = 'conll14.txt'
        conll14_file_path = os.path.join(self.save_dir, conll14_file_name)
        with open(conll14_file_path, 'w') as f:
            for item in test_data["conll14"]:
                f.write(item["predict"] + '\n')
        
        # pack bea19 output 
        bea19_file_name = 'bea19.txt'
        bea19_file_path = os.path.join(self.save_dir, bea19_file_name)
        with open(bea19_file_path, 'w') as f:
            for item in test_data["bea19"]:
                f.write(item["predict"] + '\n')
        with zipfile.ZipFile(os.path.join(self.save_dir, 'bea19.zip'), mode='w') as zipf:
            zipf.write(bea19_file_path, bea19_file_name)

    def _en_bea_postprocess_py3(self):
        for i in range(len(self.results)):
            self.results[i] = PostProcess.bea_postprocess(self.results[i])

        BEA19_NUM = 4477
        assert len(self.results) == BEA19_NUM

        # save file for further evaluation
        
        # pack bea19 output 
        bea19_file_name = 'bea19.txt'
        bea19_file_path = os.path.join(self.save_dir, bea19_file_name)
        with open(bea19_file_path, 'w') as f:
            for item in self.results:
                f.write(item["predict"] + '\n')
        with zipfile.ZipFile(os.path.join(self.save_dir, 'bea19.zip'), mode='w') as zipf:
            zipf.write(bea19_file_path, bea19_file_name)

    @staticmethod
    def conll_postprocess_new(item):
        item["oripred"] = str(item["predict"])
        line = str(item["predict"])
        answer = PostProcess.gec_reform_answer(line)

        tokens = nltk.word_tokenize(answer)
        tokenized_text = ' '.join(tokens).strip()
        item["predict"] = tokenized_text
        return item
    
    @staticmethod
    def bea_postprocess_new(item):
        global en
        item["oripred"] = str(item["predict"])
        line = str(item["predict"])
        answer = PostProcess.gec_reform_answer(line)

        doc = en(answer)
        tokens = []
        for token in doc:
            tokens.append(token.text)
        tokenized_text = ' '.join(tokens).strip()
        item["predict"] = tokenized_text
        return item

    def _conll14_bea(self):
        last_number = -1
        bea19_start_index = None
        test_data = {'conll14': [], 'bea19': []}
        current_dataset = 'conll14'
        logger.info("Postprocessing CoNLL14 test data...")
        for i in range(len(self.results)):
            assert 'conll14' in self.results[i]["id"] or 'bea19' in self.results[i]["id"], "Current test set is not the concatenation of CoNLL14 and BEA19."
            # check current result item belongs to which test set
            number, data_source = self.results[i]["id"].split('_')
            if data_source != current_dataset:
                last_number = -1
                bea19_start_index = i
                current_dataset = 'bea19'
                logger.info("Postprocessing BEA19 test data...")
            assert eval(number) == last_number + 1
            last_number = eval(number)
            assert data_source in test_data
            if current_dataset == 'conll14':
                self.results[i] = PostProcess.conll_postprocess_new(self.results[i])
            else:
                self.results[i] = PostProcess.bea_postprocess_new(self.results[i])
            test_data[data_source].append(self.results[i])

        CONLL14_NUM, BEA19_NUM = 1312, 4477
        assert len(test_data["bea19"]) == BEA19_NUM and len(test_data['conll14']) == CONLL14_NUM

        # save file for further evaluation
        # conll14 evaluation
        conll14_file_name = 'conll14.txt'
        conll14_file_path = os.path.join(self.save_dir, conll14_file_name)
        with open(conll14_file_path, 'w') as f:
            for item in test_data["conll14"]:
                f.write(item["predict"] + '\n')
        
        # pack bea19 output 
        bea19_file_name = 'bea19.txt'
        bea19_file_path = os.path.join(self.save_dir, bea19_file_name)
        with open(bea19_file_path, 'w') as f:
            for item in test_data["bea19"]:
                f.write(item["predict"] + '\n')
        with zipfile.ZipFile(os.path.join(self.save_dir, 'bea19.zip'), mode='w') as zipf:
            zipf.write(bea19_file_path, bea19_file_name)

    def _bea_dev_evaluation(self):
        for i in range(len(self.results)):
            self.results[i] = PostProcess.bea_postprocess(self.results[i])
        # conll14 evaluation
        global BEA_DEV_M2_FILE
        bea_dev_file_name = 'bea19_dev.txt'
        bea_dev_file_path = os.path.join(self.save_dir, bea_dev_file_name)
        evaluation_result_file = os.path.join(self.save_dir, 'bea19_dev_metrics.txt')
        with open(bea_dev_file_path, 'w') as f:
            for item in self.results:
                f.write(item["predict"] + '\n')

    def _mucgec_dev_evaluation(self):
        ids = [item["id"] for item in self.results]
        src_texts = [item["src"] for item in self.results]
        predict_texts = [item["predict"] for item in self.results]
        eval_information, eval_metrics = compute_cherrant_with_ref_file(
            ids=ids,
            src_texts=src_texts,
            predict_texts=predict_texts,
            ref_file=MUCGEC_DEV_M2_FILE,
            device=self.args.device
        )
        evaluation_result_file = os.path.join(self.save_dir, 'mucgec_dev_eval.txt')
        open(evaluation_result_file, 'w').write(eval_information)
        evaluation_metric_file = os.path.join(self.save_dir, 'mucgec_dev_metrics.json')
        json.dump(eval_metrics, open(evaluation_metric_file, 'w'), indent=4, ensure_ascii=False)

    def _fcgec_dev_evaluation(self):
        ids = [item["id"] for item in self.results]
        src_texts = [item["src"] for item in self.results]
        predict_texts = [item["predict"] for item in self.results]
        dev_data = json.load(open(FCGEC_DEV_FILE))
        tgt_texts = [[item['label']] + item['other_labels'] for item in dev_data]

        # check id
        assert len(ids) == len(dev_data)
        for id1, item in zip(ids, dev_data):
            assert id1 == item["id"]

        eval_information, eval_metrics = compute_cherrant(
            ids=ids,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            predict_texts=predict_texts,
            device=self.args.device
        )
        evaluation_result_file = os.path.join(self.save_dir, 'fcgec_dev_eval.txt')
        open(evaluation_result_file, 'w').write(eval_information)
        evaluation_metric_file = os.path.join(self.save_dir, 'fcgec_dev_metrics.json')
        json.dump(eval_metrics, open(evaluation_metric_file, 'w'), indent=4, ensure_ascii=False)

    def _nlpcc18_evaluate(self):
        ids = [item["id"] for item in self.results]
        src_texts = [item["src"] for item in self.results]
        predict_texts = [item["predict"] for item in self.results]
        eval_information, eval_metrics = compute_cherrant_with_ref_file(
            ids=ids,
            src_texts=src_texts,
            predict_texts=predict_texts,
            ref_file=NLPCC18_M2_FILE,
            device=self.args.device
        )
        evaluation_result_file = os.path.join(self.save_dir, 'nlpcc18_eval.txt')
        open(evaluation_result_file, 'w').write(eval_information)
        evaluation_metric_file = os.path.join(self.save_dir, 'nlpcc18.score')
        json.dump(eval_metrics, open(evaluation_metric_file, 'w'), indent=4, ensure_ascii=False)

    def _replace_linebreaker(self):
        for i, item in enumerate(self.results):
            # self.results[i]["predict"] = self.results[i]["predict"].replace('\n', ' ').strip()
            item["predict"] = item["predict"].replace('\n', ' ').strip()
            item["predict"] = item["predict"].replace('\r', ' ').strip()

    def _get_retokenized_predictions(self, tokenizer_model):
        retokenized_output = []
        for item in self.results:
            line = item["predict"]
            line = " ".join([t.text for t in tokenizer_model.tokenizer(line)])
            retokenized_output.append(line)
        
        save_path = os.path.join(self.save_dir, f'{self.args.dataset}-output-retokenized.txt')
        with codecs.open(save_path, "w", "utf-8") as f:
            for item in retokenized_output:
                save_str = item.strip()
                f.write(f"{save_str}\n")


    def _de_retokenize(self):
        de_model = spacy.load("de_core_news_sm")
        self._get_retokenized_predictions(de_model)

    def _ru_retokenize(self):
        ru_model = spacy.load("ru_core_news_sm")
        self._get_retokenized_predictions(ru_model)

    def _et_retokenize(self):
        et_model = spacy.load("et_dep_ud_sm")
        self._get_retokenized_predictions(et_model)
    
    def _ro_retokenize(self):
        ro_model = spacy.load("ro_core_news_sm")
        self._get_retokenized_predictions(ro_model)


    def basic_saving(self):
        save_path = os.path.join(self.save_dir, f'{self.args.dataset}-processed.json')
        with codecs.open(save_path, "w", "utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)

        save_path = os.path.join(self.save_dir, f'{self.args.dataset}-output-processed.txt')
        with codecs.open(save_path, "w", "utf-8") as f:
            for item in self.results:
                save_str = item['predict'].strip()
                f.write(f"{save_str}\n")

        save_txt = os.path.join(self.save_dir, f'{self.args.dataset}-processed.txt')
        with codecs.open(save_txt, "w", "utf-8") as f:
            for item in self.results:
                if "tgt" in item:
                    f.write("%s\t%s\t%s\n" % (item["src"], item["tgt"], item["predict"]))
                else:
                    f.write("%s\t%s\n" % (item["src"], item["predict"]))
        logger.info(f"Results have been stored in {save_path}.")


    def prediction_saving(self):
        """
        In infer task, some dataset requires a specific version of results to evaluate, this function will do the formatting.
        """
        self.basic_saving()
        ## MuCGEC output
        if self.args.dataset.lower() == 'mucgec':
            save_txt = os.path.join(self.save_dir, f'MuCGEC_test.txt')
            with codecs.open(save_txt, "w", "utf-8") as f:
                for item in self.results:
                    f.write("%s\t%s\t%s\n" % (item["id"], item["src"], item["predict"]))
            with zipfile.ZipFile(os.path.join(self.save_dir, 'submit.zip'), mode='w') as zipf:
                zipf.write(save_txt, 'MuCGEC_test.txt')
        
        ## FCGEC output
        if self.args.dataset.lower() == 'fcgec':
            fcgec_json = {}
            for item in self.results:
                error_flag = 1 if item["src"] != item["predict"] else 0
                fcgec_json[item['id']] = {"error_flag": error_flag, "error_type": "IWO", "correction": item["predict"]}
            fcgec_path = os.path.join(self.save_dir, 'predict.json')
            with codecs.open(fcgec_path, "w", "utf-8") as f:
                json.dump(fcgec_json, f, ensure_ascii=False, indent=4)      
            with zipfile.ZipFile(os.path.join(self.save_dir, 'predict.zip'), mode='w') as zipf:
                zipf.write(fcgec_path, 'predict.json')

    def post_process_func(self):
        if 'pre_split_length_for_infer' in self.args and self.args.pre_split_length_for_infer and PostProcessManipulator.merge_sample not in self.post_process:
            logger.info(f"Auto Set: You enable split_sentence for the test set but you did not include {PostProcessManipulator.merge_sample} as a postprocess. Auto added it in the front.")
            self.post_process.insert(0, PostProcessManipulator.merge_sample)
        if self.args.dataset == 'mucgec_dev' and PostProcessManipulator.mucgec_eval not in self.post_process:
            logger.info(f"Auto Set: You are using mucgec dev set for the test set but you did not include {PostProcessManipulator.mucgec_eval} as a postprocess. Auto added it in the rear.")
            self.post_process.append(PostProcessManipulator.mucgec_eval)      
        if self.args.dataset == 'fcgec_dev' and PostProcessManipulator.fcgec_eval not in self.post_process:
            logger.info(f"Auto Set: You are using fcgec dev set for the test set but you did not include {PostProcessManipulator.fcgec_eval} as a postprocess. Auto added it in the rear.")
            self.post_process.append(PostProcessManipulator.fcgec_eval) 
        # if self.args.dataset in ['wilocness']:
        #     if PostProcessManipulator.en_test not in self.post_process and PostProcessManipulator.en_test_py3 not in self.post_process:
        #         logger.info(f'Auto Set: You are using conll14 and bea19 united test set but you did not include postprocess function. Auto add {PostProcessManipulator.en_test_py3}.')
        #         self.post_process.append(PostProcessManipulator.en_test_py3) 
        print(self.post_process)
        for name in self.post_process:
            # check if it is an allowed processing
            allowed = False
            if name in self.allowed_dataset:
                if self.allowed_dataset[name] == []:
                    allowed = True
                elif self.args.dataset.lower() in self.allowed_dataset[name]:
                    allowed = True
            
            if allowed:
                self.post_process_func_map[name]()
            else:
                raise NotImplementedError(f"Error: Unsupported post process of {name} for {self.args.dataset}. Skipped.")
    
    def post_process_and_save(self):
        self.post_process_func()
        self.prediction_saving()

    def get_results(self):
        return self.results
