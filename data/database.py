import os
import json
from tqdm import tqdm
import torch
import spacy
import transformers
import hashlib
import random
import logging
from transformers import AutoTokenizer, AutoModel
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from llama_index.core import Document, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.core import load_index_from_storage, Settings
from llama_index.core.llms import MockLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import NodeWithScore
from transformers import pipeline
from llm.pipeline import TextGeneration

from .editor import Editor, LANGUAGE, ExplanationsParser
from .instructions.template import ExplainationTemplate
from .check_jsonl import extract_matching_valid_lines, rewrite_jsonl_with_valid_lines
from utils.log import setup_log

Settings.llm = MockLLM()  # we do not use llm in llama-index infra

logger = logging.getLogger(__name__)


STEM_LANG = {
    'en': 'english',
    'zh': None,
    'de': 'german',
    'et': 'finnish',
    'ru': 'russian',
    'ro': 'romanian',
    'ar': 'arabic',
    'kr': None
}

class DATABASE_TYPE:
    rule_relation = "rule_relation"
    edit = "edit"
    relation = "relation"
    text = "text"

DEFAULT_DATABASE_TYPE = 'text'


def generate_key_from_whole_explanation(process_mode, explanation, data_item):
    if process_mode == "prefix":
        medium_result_prefix = "For sentence:\n{text}\nGrammatical errors should be corrected as follows: \n"
        return medium_result_prefix.format(text=data_item['text']) + explanation.strip()
    elif process_mode == "":
        return explanation.strip()
    else:
        raise NotImplementedError(f"{process_mode} key generation method is not implemented by Edit-Based Database.")


def generate_key_from_list_explanations(process_mode, explanation_item_list, data_item):
    if process_mode == "prefix":
        medium_result_prefix = "For sentence:\n{text}\nGrammatical errors should be corrected as follows: \n"
        explanation_str = ""
        for item in explanation_item_list:
            explanation_str += f"{item['edit'].strip()}: {item['grammar'].strip()}\n"
        if explanation_str:
            return medium_result_prefix.format(text=data_item['text']) + explanation_str
        else:
            return ""
    elif process_mode == "":
        explanation_str = ""
        for item in explanation_item_list:
            explanation_str += f"{item['edit'].strip()}: {item['grammar'].strip()}\n"
        if explanation_str:
            return explanation_str
        else:
            return ""
    else:
        raise NotImplementedError(f"{process_mode} key generation method is not implemented by Edit-Based Database.")



def generate_key_from_edit_explanation(process_mode, explanation_item, data_item):
    if process_mode == "edit_raw" or process_mode == "":
        return explanation_item['edit'].strip() + '\n' + explanation_item["grammar"].strip()
    elif process_mode == "edit_grammar":
        return explanation_item["grammar"].strip()
    elif process_mode == "edit_full":
        return f"For sentence:\n{data_item['text']}\nWe should make a correction to the sentence: \n" + explanation_item['edit'].strip() + '\n' + explanation_item["grammar"].strip()
    elif process_mode == "prefix":
        medium_result_prefix = "For sentence:\n{text}\nGrammatical errors should be corrected as follows: \n"
        return medium_result_prefix.format(text=data_item['text']) + explanation_item["grammar"].strip()
    else:
        raise NotImplementedError(f"{process_mode} key generation method is not implemented by Edit-Based Database.")

class TextDataBase:
    def __init__(self, icl_config, dataset: Dataset, use="in_domain", need_database=True) -> None:
        self.icl_config = icl_config

        self.dataset_names = list(set(dataset['from']))

        # parameters of database construction
        self.database_parameters_str = icl_config.database_parameters
        self.database_parameters = json.loads(icl_config.database_parameters)

        # spacy model for tokenize
        language_set = list(set([LANGUAGE[name] for name in self.dataset_names]))
        assert len(language_set) == 1
        self.language = language_set[0]

        # dataset type and key type
        self.use = use
        assert use in ['in_domain', 'cross_domain']
        self.key_type = icl_config.in_domain_example_mode if use=="in_domain" else icl_config.cross_domain_example_mode
        if self.key_type == "default":
            self.key_type = DEFAULT_DATABASE_TYPE
        if self.key_type == "bm25":
            self.key_type = "text"

        # filter from args
        if self.use == 'in_domain':
            self.filter_name = self.icl_config.in_domain_filter
        else:
            self.filter_name = self.icl_config.cross_domain_filter

        # result directory
        os.makedirs(icl_config.database_dir, exist_ok=True)

        database_name, medium_result_name = self._get_database_and_medium_res_name()
        self.database_dir = os.path.join(icl_config.database_dir, "database" , database_name)
        self.medium_result_dir = os.path.join(icl_config.medium_result_dir, medium_result_name)

        # word tokenizer
        self.editor = Editor(self.dataset_names[0])
        logger.info(f"Initialize spacy tokenizer by {self.dataset_names[0]}.")
        # embedding model (lazy init)
        # self.embed_model = None
        # generate model (lazy init)
        # self.generate_model = None

        # retriever (lazy init)
        self.index = None
        self.retriever = None

        # enable log in medium result directory while constructing database
        setup_log(self.medium_result_dir)
        logger.info(f"Initialize {self.key_type} data base for {self.dataset_names} used by {use} selection.")
        logger.info(f"Medium result will be saved in {self.medium_result_dir}")
        self.dataset: Dataset = self._filter_dataset(dataset)
        logger.info(f"[NUM] database length {len(self.dataset)}")
        if need_database:
            self._construct_basic_database()
        # after construction, log will be redirected to output directory
        setup_log(icl_config.output_dir)
    
    def _filter_dataset(self, dataset: Dataset):
        # statistics
        total_num = len(dataset)
        error_num = sum([1 for item in dataset if item['text'] != item['label']])
        correct_num = total_num - error_num

        logger.info(f"[NUM] Total {total_num}/ Erroneous {error_num} / Correct {correct_num}")

        def correct_filter(item):
            if item['text'] == item['label']:
                return True
            else:
                return False
        def error_filter(item):
            return not correct_filter(item)
        
        if self.filter_name == 'correct':
            filter_func = correct_filter
        else:
            filter_func = error_filter

        logger.info(f"Filter dataset by {self.filter_name} sampler.")

        num_limit = self.database_parameters['num_limit']
        min_len, max_len = self.database_parameters['min_len'], self.database_parameters['max_len']

        # load cache filtered index
        hash_obj = hashlib.sha256()
        filter_hash_config = [
            self.database_parameters_str,
            self.filter_name
        ] + self.dataset_names
        hash_obj.update('-'.join(filter_hash_config).encode('utf-8'))
        data_index_name = self.dataset_names[0] + '-' + hash_obj.hexdigest()[:8]

        data_index_save_dir = os.path.join(self.icl_config.database_dir, "dataindex", data_index_name)
        os.makedirs(data_index_save_dir, exist_ok=True)
        data_index_config = {
            "dataset_names": '-'.join(self.dataset_names),
            "filter_name": self.filter_name,
            "database_parameters": self.database_parameters_str
        }
        data_index_config_file = os.path.join(data_index_save_dir, "config.json")
        data_index_file = os.path.join(data_index_save_dir, "indices.json")

        if os.path.exists(data_index_config_file):
            loaded_data_index_config = json.load(open(data_index_config_file))
            consistent = True
            if len(data_index_config.keys()) != len(loaded_data_index_config.keys()):
                consistent = False
            for key in data_index_config:
                if data_index_config[key] != loaded_data_index_config[key]:
                    consistent = False
                    break
            if consistent:
                logger.info(f"[DATABASE] Loading filtered dataset indices from {data_index_file}")
                indices = json.load(open(data_index_file))
                return dataset.select(indices)
            
        # compute filtered index
        indices = []
        for i, item in enumerate(tqdm(dataset, desc="Token Length Recognize")):
            text = item['label']
            tokens = self.editor._split_sentence(text)
            if min_len <= len(tokens) <= max_len and filter_func(item):
                indices.append(i)
        if len(indices) > num_limit:
            random.shuffle(indices)
            indices = indices[:num_limit]

        # cache filtered index
        with open(data_index_file, 'w') as f:
            json.dump(indices, f, indent=4)
        with open(data_index_config_file, 'w') as f:
            json.dump(data_index_config, f, indent=4)

        logger.info(f"[DATABASE] Length before filter {len(dataset)}; After filter {len(indices)}")

        return dataset.select(indices)

    
    def _get_database_and_medium_res_name(self):
        prefix = self.key_type + '-' + self.filter_name + '-' + '_'.join([n for n in self.dataset_names]) + '-'

        database_hash = hashlib.sha256()
        if self.key_type == DATABASE_TYPE.text:
            database_hash_config = [
                self.icl_config.embedding_model,
                self.key_type,
                self.database_parameters_str,
            ] + self.dataset_names
        else:
            database_hash_config = [
                self.icl_config.assist_model,
                self.icl_config.assist_prompt,
                self.icl_config.embedding_model,
                self.icl_config.medium_result_postprocess, 
                self.key_type,
                self.database_parameters_str,
            ] + self.dataset_names
        database_hash.update('-'.join(database_hash_config).encode('utf-8'))
        database_name = prefix + database_hash.hexdigest()[:8]

        medium_hash = hashlib.sha256()
        medium_hash_config = [
            self.icl_config.assist_model,
            self.icl_config.assist_prompt,
            self.key_type,
        ] + self.dataset_names
        medium_hash.update('-'.join(medium_hash_config).encode('utf-8'))
        medium_name = prefix + medium_hash.hexdigest()[:8]

        return database_name, medium_name
    
    def _check_or_save_medium_config(self, medium_res_dir):
        medium_config = {
            "assist_model": self.icl_config.assist_model,
            "assist_prompt": self.icl_config.assist_prompt,
            "key_type": self.key_type,
            "dataset_names": '-'.join(self.dataset_names)
        }
        config_file = os.path.join(medium_res_dir, "config.json")
        
        consistent = False
        loaded_medium_config = {}
        if os.path.exists(config_file):
            loaded_medium_config = json.load(open(config_file))
            for key in medium_config:
                if len(loaded_medium_config.keys()) != len(medium_config.keys()):
                    break
                if loaded_medium_config[key] != medium_config[key]:
                    break
            else:
                consistent = True

        if not consistent:
            logger.info(f"[WARNING]: Detect unmatched cache of medium results at {config_file}, new results will cover them.")
            logger.info(f"[WARNING] Un matched: {medium_config} and {loaded_medium_config}")
            with open(config_file, 'w') as f:
                json.dump(medium_config, f, ensure_ascii=False, indent=4)

    def load_or_build_index(self, documents):
        """
        Load or build index according to if the database file exists.
        """
        # Optional: add 'show_progress_bar=False' to source code of HuggingFaceEmbedding._embed, in call of self._model.encode
        embed_model = HuggingFaceEmbedding(model_name=self.icl_config.embedding_model, embed_batch_size=self.icl_config.embedding_batch_size, trust_remote_code=True)
        
        # Create ServiceContext without LLM
        service_context = ServiceContext.from_defaults(llm=MockLLM(), embed_model=embed_model, text_splitter=None, chunk_size_limit=None)
            
        if os.path.exists(self.database_dir) and len(os.listdir(self.database_dir)) > 0:
            # load
            logger.info(f"[DATABASE] Loading the existing {self.key_type} database from {self.database_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=self.database_dir)
            index = load_index_from_storage(storage_context, service_context=service_context)
        else:
            # create
            logger.info("[DATABASE] No existing database, start to construct...")
            
            # create from documents
            index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
            index.storage_context.persist(persist_dir=self.database_dir)
            
            logger.info(f"[DATABASE] Construction complete in {self.database_dir} and length {documents}.")
        
        del embed_model
        
        return index
    
    def _self_dataset_filter(self):
        new_dataset = self.dataset.select()
        self.dataset = new_dataset


    def _construct_basic_database(self):
        # save settings
        os.makedirs(self.medium_result_dir, exist_ok=True)
        self._check_or_save_medium_config(self.medium_result_dir)
        self.medium_result_jsonl = os.path.join(self.medium_result_dir, 'medium.jsonl')
        
        # if need relation, init the chat model to acquire relations.
        if self.key_type in [DATABASE_TYPE.relation, DATABASE_TYPE.edit, DATABASE_TYPE.rule_relation]:
            prompt_mode = self.icl_config.assist_prompt
            pipeline = TextGeneration(
                model_name=self.icl_config.assist_model,
                do_sample=False,
                return_full_text=False,
            )
            templator = ExplainationTemplate(prompt_mode)
            edit_mode = templator.get_edit_extract_mode()
            logger.info(f"[DATABASE]Editor has been set to {edit_mode} mode to extract edits from text for further explanation.")
            self.editor.set_edit_mode(edit_mode=edit_mode)
        else:
            pipeline = None
            templator = None

        # generation result file

        # if relation need to be get...
        medium_res = []

        if self.key_type in [DATABASE_TYPE.relation, DATABASE_TYPE.edit]:
            logger.info(f"Start to get medium result of relations for {len(self.dataset)} examples for databse of {'_'.join(self.dataset_names)}.")
            logger.info(f"Current ICL configs: {self.icl_config}")
            logger.info(f"Prompt Mode: {prompt_mode}")   

            medium_res = extract_matching_valid_lines(self.dataset, self.medium_result_jsonl)

            if pipeline.mode == "API" and len(medium_res) != len(self.dataset):
            # API-based generation
                # generate for all data, cache in api_cache.json
                api_res_cache_file = os.path.join(self.medium_result_dir, 'api_cache.json')

                data_for_api = []
                for item in tqdm(self.dataset, desc="Preprocessing"):
                    comparation = self.editor.compare_text_by_edit(item['text'], item['label'])
                    medium_item = dict(item)
                    system, query = templator.format({'text': item['text'], 'hypothesis': item['label']}, error_list=comparation)
                    item_for_api = {
                        "id": item["id"],
                        "text": item["text"],
                        "sentence": [
                            {'role': "system", 'content': system},
                            {'role': "user", 'content': query}
                        ]
                    }
                    data_for_api.append(item_for_api)

                # raise PermissionError("For now we do not allow using api for database construction")
                results = pipeline(data_for_api, api_res_cache_file)
                # convert to standard results
                with open(self.medium_result_jsonl, 'w') as f:
                    for i, item in enumerate(data_for_api):
                        assert results[i]["id"] == item["id"]
                        medium_item = dict(item)
                        response = results[i]["result"].strip()
                        generated_text = templator.postprocess(response)
                        medium_item['medium'] = generated_text
                        f.write(json.dumps(medium_item, ensure_ascii=False) + '\n')
                        f.flush()


            medium_res = extract_matching_valid_lines(self.dataset, self.medium_result_jsonl)
            f = rewrite_jsonl_with_valid_lines(medium_res, self.medium_result_jsonl)
            
            if len(medium_res) < len(self.dataset):
                for item in tqdm(self.dataset.select(range(len(medium_res), len(self.dataset))), desc="Medium Result"):
                    comparation = self.editor.compare_text_by_edit(item['text'], item['label'])
                    medium_item = dict(item)
                    system, query = templator.format({'text': item['text'], 'hypothesis': item['label']}, error_list=comparation)
                    # system = "You, a language expert, can briefly explain how to judge a sentence is grammatically correct and why some corrections are essential."
                    if pipeline.chat_template():
                        messages = [
                            {'role': "system", 'content': system},
                            {'role': "user", 'content': query}
                        ]
                    else:
                        messages = system + '\n' + query
                    outputs = pipeline(
                        messages,
                    )

                    logger.info(f'[Original Content][id:{item["id"]}][from:{item["from"]}]')
                    logger.info(messages[-1]['content'])
                    logger.info('[Answer]')
                    logger.info(outputs[0]["generated_text"])
                    
                    generated_text = templator.postprocess(outputs[0]["generated_text"].strip())
                    medium_item['medium'] = generated_text
                    f.write(json.dumps(medium_item, ensure_ascii=False) + '\n')
                    f.flush()
                    medium_res.append(medium_item)
            f.close()
            assert len(medium_res) == len(self.dataset)

        elif self.key_type == DATABASE_TYPE.rule_relation:
            medium_res = extract_matching_valid_lines(self.dataset, self.medium_result_jsonl)
            f = rewrite_jsonl_with_valid_lines(medium_res, self.medium_result_jsonl)
            if len(medium_res) < len(self.dataset):
                for item in tqdm(self.dataset.select(range(len(medium_res), len(self.dataset))), desc="Medium Result"):
                    comparation = self.editor.compare_text_by_edit(item['text'], item['label'])
                    medium_item = dict(item)
                    medium_item['medium'] = item["text"] + '\n' + comparation
                    f.write(json.dumps(medium_item, ensure_ascii=False) + '\n')
                    f.flush()
                    medium_res.append(medium_item)
            f.close()
            assert len(medium_res) == len(self.dataset)

        # construct vector store according to text (text mode) or medium text (*relation mode)
        used_data_num = 0
        # prepare dataset as vector
        documents = []
        
        for idx, item in enumerate(tqdm(self.dataset, desc="Documents for database")):
            if self.key_type in [DATABASE_TYPE.rule_relation, DATABASE_TYPE.relation]:
                assert item['id'] == medium_res[idx]['id']

                ## parsing the results
                if templator.explanation_structure() == "structured":
                    explanation_parser = ExplanationsParser(templator=templator, editor=self.editor)
                    explanations = explanation_parser.initial_parse(item, medium_res[idx]['medium'].strip())
                    key_ = generate_key_from_list_explanations(
                        self.icl_config.medium_result_postprocess, 
                        explanations,
                        item
                    )
                else:
                    key_ = generate_key_from_whole_explanation(
                        self.icl_config.medium_result_postprocess, 
                        medium_res[idx]['medium'].strip(),
                        item
                    )

                # when the explanation extraction failed and return [] for explanations, generated key will also be blank
                if key_.strip() == "":
                    continue

                documents.append(
                    Document(
                        text=key_,
                        metadata={
                            "id": item["id"],
                            "text": item["text"],
                            "label": item["label"],
                            "from": item["from"]
                        },
                        excluded_embed_metadata_keys=["id", "text", "label", "from"],
                        text_template="{content}"
                    )
                )
            elif self.key_type in [DATABASE_TYPE.edit]:
                # edit database mode
                assert item['id'] == medium_res[idx]['id']
                ## parsing the results
                explanation_parser = ExplanationsParser(templator=templator, editor=self.editor)
                explanations = explanation_parser.initial_parse(item, medium_res[idx]['medium'].strip())
                ## add to database by edit
                if explanations:
                    used_data_num += 1
                    if idx == len(self.dataset) - 1:
                        logger.info(f"Edit database use {used_data_num}/{idx+1} from original filtered database due to postprocess")
                for i, explanation_item in enumerate(explanations):
                    key_ = generate_key_from_edit_explanation(self.icl_config.medium_result_postprocess, explanation_item, item)

                    if key_.strip() == "":
                        continue

                    documents.append(
                        Document(
                            text=key_,
                            metadata={
                                "id": f'{item["id"]}_edit{i}',
                                "text": item["text"],
                                "label": item["label"],
                                "edit": explanation_item["edit"],
                                "grammar": explanation_item["grammar"],
                                "data_item_id": item["id"],
                                "from": item["from"]
                            },
                            excluded_embed_metadata_keys=["id", "text", "label", "from", "edit", "grammar", "data_item_id"],
                            text_template="{content}"
                        )
                    )

            # only use text as key text
            elif self.key_type == DATABASE_TYPE.text:
                documents.append(
                    Document(
                        text=item['text'],
                        metadata={
                            "id": item["id"],
                            "text": item["text"],
                            "label": item["label"],
                            "from": item["from"]
                        },
                        excluded_embed_metadata_keys=["id", "text", "label", "from"],
                        text_template="{content}"
                    )
                )
            else:
                raise NotImplementedError()

        
        # embed and index
        del pipeline
        logger.info(f"build index for database, database has items {len(documents)}...")
        self.index = self.load_or_build_index(documents=documents)


    def _init_retriever(self):
        assert self.index is not None, "Indexer are not initalized before retrieve."
        if self.use == "in_domain":
            topk = self.icl_config.in_domain_example_num
            mode = self.icl_config.in_domain_example_mode
        elif self.use == "cross_domain":
            topk = self.icl_config.cross_domain_example_num
            mode = self.icl_config.cross_domain_example_mode
        else:
            raise NotImplementedError(f'un recognized use of database: {self.use}')
        
        if mode == "bm25":
            # bm25 based retriever
            # stem_language = STEM_LANG[self.language]
            # if stem_language:
            #     stemmer = Stemmer.Stemmer("english")
            #     self.retriever = BM25Retriever.from_defaults(
            #         docstore=self.index.docstore, similarity_top_k=topk, stemmer=stemmer, language=stem_language
            #     )
            # else:
            self.retriever = BM25Retriever.from_defaults(
                docstore=self.index.docstore, similarity_top_k=topk, tokenizer=self.editor.get_tokenizer()
            )
        else:
            # vector based retriever
            self.retriever = self.index.as_retriever(similarity_top_k=topk)

    def retrieve(self, query):
        if self.retriever is None:
            logger.info(f"First use database of {self.database_dir}, initialize retriever.")
            self._init_retriever()
        
        # logger.info(query)
        results = self.retriever.retrieve(query)
        meta_data = []
        for res in results:
            item = res.metadata
            item["key"] = res.get_content()
            item["similarity"] = res.get_score()
            # logger.info(item["key"])
            # logger.info(item["similarity"])
            meta_data.append(item)
        return meta_data        
