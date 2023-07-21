import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass

import stanza
import torch
from stanza import DownloadMethod
from tqdm import tqdm

logging.basicConfig(filename='logs/ner_processed_file.log',
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

    @staticmethod
    def from_json(json_data: dict):
        new_entity = Entity(name=json_data['name'])
        new_entity.label2count = json_data['label2count']
        new_entity.count = json_data['count']
        return new_entity


def ner_from_files(input_dir: str, output_dir: str):
    for root, _, files in tqdm(os.walk(input_dir), total=len(os.listdir(input_dir)), desc='Total: '):
        logging.info(f'Processing dir {root}.')

        for file in tqdm(files, desc='Subdir: '):
            # Dictionary to store named entities and their counts
            entity_name2entity: dict[str, Entity] = {}

            file_path = os.path.join(root, file)

            # First try to process the file as whole.
            try:
                logging.info(f'Processing file {file_path} as whole.')
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

            # If CUDA OOM, process the lines separately.
            except RuntimeError:
                logging.warning(f'Failed processing file {file_path} as whole.')
                logging.info(f'Processing file {file_path} as lines.')

                # Release GPU resources for previous failed process.
                torch.cuda.empty_cache()

                # Create a NER processor for Chinese with GPU enabled
                nlp = stanza.Pipeline(lang='zh', processors='tokenize,ner', use_gpu=True,
                                      download_method=DownloadMethod.REUSE_RESOURCES,
                                      logging_level='WARN')

                with open(file_path, 'r', encoding='utf-8') as infile:
                    lines = infile.readlines()

                # Process the text with NER
                for line in tqdm(lines, desc=f'File {file_path}: '):
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

    ner_from_files(input_dir, output_dir)

    # dict(sorted(entity_name2count.items(), key=lambda item: item[1], reverse=True))
