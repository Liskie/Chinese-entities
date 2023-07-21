import json
import logging
import os

from tqdm import tqdm

from ner import Entity

logging.basicConfig(filename='logs/merge_dict.log',
                    level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] %(message)s')

logger = logging.getLogger()

def merge(input_dir: str, output_path: str) -> None:
    entity_name2entity: dict[str, Entity] = {}

    for root, _, files in tqdm(os.walk(input_dir), total=len(os.listdir(input_dir)), desc='Total: '):

        for file in tqdm(files, desc='Subdir: '):
            file_path = os.path.join(root, file)
            logger.info(f'Processing {file_path}.')

            with open(file_path, 'r') as reader:
                entities_data = json.load(reader)

            for entity_data in entities_data:
                new_entity = Entity.from_json(entity_data)
                if new_entity.name not in entity_name2entity:
                    entity_name2entity[new_entity.name] = new_entity
                else:
                    existing_entity = entity_name2entity[new_entity.name]
                    for label, count in new_entity.label2count.items():
                        existing_entity.label2count[label] += count
                    existing_entity.count += new_entity.count

    entity_name2entity = dict(sorted(entity_name2entity.items(), key=lambda item: item[1].count, reverse=True))

    with open(output_path, 'w') as writer:
        json.dump([entity.to_json() for entity in entity_name2entity.values()], writer, ensure_ascii=False)


if __name__ == '__main__':
    input_dir = 'output/entities'
    output_path = 'output/merged_entity_dict.json'

    merge(input_dir, output_path)
