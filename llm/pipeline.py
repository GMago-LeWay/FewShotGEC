from typing import Any
import transformers
import os
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import gc
from openai import OpenAI
import threading
from typing import List, Dict
import time
import datasets
from tqdm import tqdm
import json
import hashlib
import logging

API_MODELS_URL = {
    "deepseek": "https://api.deepseek.com",
    "gpt-4o-mini": "",
    "gpt-4o-mini-batch": "",
}

API_MODEL_NAME_MAP = {
    "deepseek": "deepseek-chat",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini-batch": "gpt-4o-mini-2024-07-18",
}

PROHIBITED = False

logger = logging.getLogger(__name__)

# stopping criteria
class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, answer_end) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.answer_end = answer_end
        self.check_len = int(1.5 * len(self.answer_end)) if self.answer_end else 1

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        if self.answer_end in generated_text[-self.check_len:]:
            return True
        return False


class OpenAIChat:
    def __init__(self, model_name, max_new_tokens, processor_num=5, max_retries=10) -> None:
        self.api_name = model_name.split('/')[-1].strip()
        self.token = open(os.path.join(model_name, "token")).read().strip()

        if API_MODELS_URL[self.api_name]:
            self.client = OpenAI(api_key=self.token, base_url=API_MODELS_URL[self.api_name])
        else:
            self.client = OpenAI(api_key=self.token)

        self.process = processor_num       # max parallel API call number
        self.max_retries = max_retries      # max retries times for one single

        # parameters
        self.max_tokens = max_new_tokens

    def chat(self, single_messages):
        response = self.client.chat.completions.create(
            model=API_MODEL_NAME_MAP[self.api_name],
            messages=single_messages,
            max_tokens=self.max_tokens,
            stream=False,
        )
        return response.choices[0].message.content

    def chat_dataset_parallel(self, data, cache_file) -> List[Dict[str, str]]:
        if PROHIBITED:
            raise PermissionError("The online api cannot be called for now. Please refer to llm/pipeline.py")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_results = json.load(f)
        else:
            cached_results = [None] * len(data)
        results = cached_results.copy()


        pending_indices = [i for i, r in enumerate(results) if r is None]

        lock = threading.Lock() 
        semaphore = threading.Semaphore(self.process)  

        pbar = tqdm(total=len(pending_indices), desc="Processing") 

        def controlled_worker(index: int, item: Dict[str, str], retries: int = 0):
            semaphore.acquire()
            try:
                result = self.chat(item["sentence"])
                with lock:
                    results[index] = {"id": item["id"], "result": result}
                    self._save_cache(results, cache_file)
            except Exception as e:
                if retries < self.max_retries:
                    #
                    time.sleep(2 ** retries)  
                    controlled_worker(index, item, retries + 1)
                else:
                    with lock:
                        results[index] = {"id": item["id"], "result": f"Error occurred after {retries} retries: {str(e)}"}
                        print(f"Error occurred after {retries} retries: {str(e)}")
                        
                        self._save_cache(results, cache_file)
            finally:
                semaphore.release()
                pbar.update(1) 

        threads = []
        for index in pending_indices:
            item = data[index]
            thread = threading.Thread(target=controlled_worker, args=(index, item))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        pbar.close()  

        return results
    
    def save_batch_file(self, data, request_file):
        requests = []
        with open(request_file, 'w') as f:
            for i, item in enumerate(data):
                request_item = {"custom_id": f"request-{i}-{item['id']}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": API_MODEL_NAME_MAP[self.api_name], "messages": item["sentence"], "max_tokens": self.max_tokens}}
                requests.append(request_item)
                f.write(json.dumps(request_item, ensure_ascii=False) + '\n')

        return requests
        

    def chat_dataset_batch(self, data, cache_file) -> List[Dict[str, str]]:
        cache_dir = os.path.dirname(cache_file)
        request_file = os.path.join(cache_dir, "request.jsonl")
        self.save_batch_file(data, request_file)

        dataset_name = os.path.basename(cache_dir)

        # if there is a batch id
        batch_id_file = os.path.join(cache_dir, "batch_id.txt")
        if os.path.exists(batch_id_file):
            batch_id = open(batch_id_file).read().strip()
        else:
            batch_input_file = self.client.files.create(
                file=open(request_file, "rb"),
                purpose="batch"
            )
            batch_id = batch_input_file.id
            with open(batch_id_file, 'w') as f:
                f.write(batch_id)
        

        results = []

        # find batch status
        try:
            status = self.client.batches.retrieve(batch_id)
            logger.info(status.status)
            logger.info(status.completed_at)
        except Exception as e:
            logger.info(e)

        # batch_info = self.client.batches.create(
        #     input_file_id=batch_id,
        #     endpoint="/v1/chat/completions",
        #     completion_window="24h",
        #     metadata={
        #         "description": f"For results of {cache_dir}"
        #     }
        # )
        # # item["sentence"] item["id"]
        # logger.info(batch_info)

        return results

    def _save_cache(self, results, cache_file):
        with open(cache_file, 'w') as f:
            json.dump(results, f)


class TransformerTextGeneration:
    def __init__(
            self, 
            model_name,
            max_new_tokens=512,
            do_sample=None, 
            temperature=None,
            top_k=None, 
            top_p=None, 
            return_full_text=False, 
            num_return_sequences=1, 
            stop_string=None, 
            **kwargs
        ) -> None:

        # save all the parameters
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.return_full_text = return_full_text
        self.num_return_sequences = num_return_sequences
        self.stop_string = stop_string

        
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            device_map='auto',
            **kwargs
        )

        self.reset_stop_string(stop_string=stop_string)


    def reset_stop_string(self, stop_string):
        self.stop_string = stop_string
        # stop conditions
        self.stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(self.pipeline.tokenizer, stop_string)])

        # token terminators
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
        ]
        if "Llama" in self.model_name:
            self.terminators.append(self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        if stop_string:
            end_token = self.pipeline.tokenizer.convert_tokens_to_ids(stop_string)
            if end_token:
                # end token is one standard token in tokenizer
                self.terminators.append(end_token)
                self.stopping_criteria = None
        else:
            # no end text
            self.stopping_criteria = None
    
    def chat_template(self):
        return self.pipeline.tokenizer.chat_template


    def __call__(self, inputs) -> Any:
        # judge if the type of inputs is dialogue
        is_dialogue = False
        if type(inputs) == str:
            pass
        elif type(inputs[0]) == dict:
            is_dialogue = True
        else:
            raise NotImplementedError("The pipeline should get inputs of str | list[dict[str]], got {type}".format(type=type(inputs)))   

        return self.pipeline(
            inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_k=self.top_k if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            return_full_text=self.return_full_text,
            num_return_sequences=self.num_return_sequences,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            eos_token_id=self.terminators,
            stopping_criteria=self.stopping_criteria
        )
    
    def __del__(self):
        del self.pipeline
        gc.collect()
        if torch.cuda:
            torch.cuda.empty_cache()
        self.pipeline = None


class TextGeneration:
    def __init__(
            self, 
            model_name,
            max_new_tokens=512,
            do_sample=None, 
            temperature=None,
            top_k=None, 
            top_p=None, 
            return_full_text=False, 
            num_return_sequences=1, 
            stop_string=None,
            **kwargs
        ) -> None:
        self.base_name = model_name.split('/')[-1].strip()
        if self.base_name in API_MODELS_URL:
            self.mode = "API"
            self.generation_pipe = OpenAIChat(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
            )
        else:
            self.mode = "HuggingFace"
            self.generation_pipe = TransformerTextGeneration(
                model_name,
                max_new_tokens,
                do_sample, 
                temperature,
                top_k, 
                top_p, 
                return_full_text, 
                num_return_sequences, 
                stop_string, 
                **kwargs
            )
        
    def reset_stop_string(self, stop_string):
        if self.mode == "HuggingFace":
            self.generation_pipe.reset_stop_string(stop_string)

    def chat_template(self):  
        if self.mode == "API":
            return True
        else:
            return self.generation_pipe.chat_template()
        
    def __call__(self, inputs, cache_file=None):
        if self.mode == "API":
            if "batch" in self.base_name:
                return self.generation_pipe.chat_dataset_batch(inputs, cache_file=cache_file)
            else:
                return self.generation_pipe.chat_dataset_parallel(inputs, cache_file=cache_file)
        else:
            return self.generation_pipe(inputs)
        


if __name__ == "__main__":
    # your model with token save as a file in the directory
    model_name = "/data/xxx/models/gpt-4o-mini"

    data = [
        {"id": 1, "sentence": [{"role": "user", "content": "Hello, how are you?"}]},
        {"id": 2, "sentence": [{"role": "user", "content": "What's the weather like today?"}]}
    ]

    chat_client = OpenAIChat(model_name, max_new_tokens=1024, processor_num=5) 
    results = chat_client.chat_dataset_parallel(data, cache_file="results/temp_cache.json")
    print(results)
