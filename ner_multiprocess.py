import json
import logging
import os
import torch
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from stanza import DownloadMethod
import stanza
from tqdm import tqdm

logging.basicConfig(filename='logs/ner_mp_processed_file.log',
                    level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] %(message)s')


class Entity:
    def __init__(self, name: str):
        self.name: str = name
        self.label2count: dict[str, int] = defaultdict(int)
        self.count: int = 0

    def to_json(self):
        return {
            'name': self.name,
            'label2count': self.label2count,
            'count': self.count
        }


def ner_from_files(args):
    file_paths, gpu_id, output_dir = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    for file_path in tqdm(file_paths, desc=f'GPU {gpu_id}: '):
        # Dictionary to store named entities and their counts
        entity_name2entity: dict[str, Entity] = {}

        try:
            logging.info(f'GPU {gpu_id} is processing {file_path} as whole.')
            # Create a NER processor for Chinese with GPU enabled
            nlp = stanza.Pipeline(lang='zh', processors='tokenize,ner', use_gpu=True,
                                  download_method=DownloadMethod.REUSE_RESOURCES,
                                  logging_level='WARN')

            with open(file_path, 'r', encoding='utf-8') as infile:
                text = infile.read()

            # Process the text with NER
            doc = nlp(text)

            # Collect named entities and update the count in the dictionary
            for sent in doc.sentences:
                for entity in sent.ents:
                    # Create entity if it does not exist
                    if entity.text not in entity_name2entity:
                        entity_name2entity[entity.text] = Entity(name=entity.text)
                    # Update the count of entities
                    entity_name2entity[entity.text].label2count[entity.type] += 1
                    entity_name2entity[entity.text].count += 1

        except RuntimeError:
            logging.warning(f'GPU {gpu_id} failed processing {file_path} as whole.')
            logging.info(f'GPU {gpu_id} falls back to processing {file_path} as lines.')

            torch.cuda.empty_cache()

            nlp = stanza.Pipeline(lang='zh', processors='tokenize,ner', use_gpu=True,
                                  download_method=DownloadMethod.REUSE_RESOURCES,
                                  logging_level='WARN')

            with open(file_path, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()

            # Process the text with NER
            for line in lines:
                doc = nlp(line)

                # Collect named entities and update the count in the dictionary
                for sent in doc.sentences:
                    for entity in sent.ents:
                        # Create entity if it does not exist
                        if entity.text not in entity_name2entity:
                            entity_name2entity[entity.text] = Entity(name=entity.text)
                        # Update the count of entities
                        entity_name2entity[entity.text].label2count[entity.type] += 1
                        entity_name2entity[entity.text].count += 1

        # Dump the entity dict for current file
        output_path = f'{file_path.replace(input_dir, output_dir, 1)}.json'
        final_output_dir = os.path.dirname(output_path)
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir)
        with open(output_path, 'w') as writer:
            json.dump([entity.to_json() for entity in entity_name2entity.values()], writer, ensure_ascii=False)

        # Release GPU resources for each document
        torch.cuda.empty_cache()


if __name__ == "__main__":
    input_dir = "output/extracted_texts"
    output_dir = 'output/entities'

    all_files = [os.path.join(root, file) for root, _, files in os.walk(input_dir) for file in files]
    num_gpus = 4
    file_groups = [all_files[i::num_gpus] for i in range(num_gpus)]

    with Pool(num_gpus) as p:
        p.map(ner_from_files, [(file_group, i, output_dir) for i, file_group in enumerate(file_groups)])