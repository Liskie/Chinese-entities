import json
import logging
import os

import jsonlines
from tqdm import tqdm

from ner import Entity

excluded_labels: list[str] = [
    'CARDINAL',
    'ORDINAL',
    'DATE',
    'PERCENT',
    'QUANTITY',
    'TIME',
    'MONEY',

]

entity_name2references: dict[str, list[str]] = {
    '堪萨斯州': ['堪萨斯', '堪萨斯州', '美国堪萨斯州', '(堪萨斯', '美国堪萨斯'],
    '铁达尼号': ['铁达尼号', '铁达尼'],
    '密歇根州': ['密歇根州', '密歇根']
}


def filter_entities(input_path: str, output_path: str, topn: int = 10000) -> None:
    with open(input_path, 'r') as reader:
        entities_data = json.load(reader)

    logger.info(f'Loading entities from {input_path}.')

    logger.info('Filtering entities.')
    entities: list[Entity] = [Entity.from_json(entity_data) for entity_data in entities_data]
    filtered_entities: list[Entity] = []

    for entity in tqdm(entities, desc='Filtering entities: '):
        if entity.name.isascii():
            continue

        entity.label2count = dict(sorted(entity.label2count.items(), key=lambda x: x[1], reverse=True))
        if list(entity.label2count.keys())[0] in excluded_labels:
            continue

        filtered_entities.append(entity)

    logger.info(f'Dumping filtered {topn} entities into {output_path}.')

    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all([entity.to_json() for entity in filtered_entities[:topn]])


if __name__ == '__main__':
    logging.basicConfig(filename='logs/filter.log',
                        level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)s] %(message)s')

    logger = logging.getLogger()

    topn = 10000
    input_dir = 'output/merged_entity_dict.json'
    output_path = f'output/top{topn}_entities_zh.jsonl'

    filter_entities(input_dir, output_path, topn=topn)
