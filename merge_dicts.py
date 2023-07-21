import json
import logging
import os

from tqdm import tqdm

from ner import Entity

logging.basicConfig(filename='logs/merge_dict.log',
                    level=logging.DEBUG,
                    format='[%(asctime)s] [%(levelname)s] %(message)s')


def merge(input_dir: str, output_path: str) -> None:
    entity_name2entity: dict[str, Entity] = {}

    for root, _, files in tqdm(os.walk(input_dir), total=len(os.listdir(input_dir)), desc='Total: '):

        for file in tqdm(files, desc='Subdir: '):
            file_path = os.path.join(root, file)
            logging.info(f'Processing {file_path}.')

            with open(file_path, 'r') as reader:
                entities_data = json.load(reader)
                for entity_data in entities_data:
                    logging.debug(entity_data)
            return


if __name__ == '__main__':
    input_dir = 'output/entities'
    output_path = 'output/merged_entity_dict.json'

    merge(input_dir, output_path)
