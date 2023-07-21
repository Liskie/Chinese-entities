import json
import jsonlines
import requests

from ner import Entity


class AlignedEntity(Entity):

    def __init__(self, name: str):
        super().__init__(name)


def search_entity(entity: Entity) -> str | None:
    """
    This function searches the name of a given entity on wikidata.
    If any related entity on wikidata matches the given one,
    update the name of the given entity with the search result.
    """
    url = "https://www.wikidata.org/w/api.php"
    query = entity.name
    params = {
        "action": "wbsearchentities",
        "language": "zh",
        "search": query,
        "format": "json",
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "python-requests",
    }

    response = requests.get(url=url, headers=headers, params=params)
    if response.status_code == 200:
        data = json.loads(response.text)
        print(json.dumps(data, indent=4, ensure_ascii=False))
        if len(data["search"]) > 0:

            entity.wikidata_id = data["search"][0]["id"]
            entity.wikidata_name = data['search'][0]['match']['text']

            return
    return None


def align_entities(entities, limit=10):
    aligned_entities = {}
    for entity in entities[:limit]:
        wikidata_id = search_entity(entity)
        if wikidata_id:
            aligned_entities[entity] = wikidata_id
            print(f"{entity} -> {wikidata_id}")
        else:
            print(f"{entity} not found in Wikidata")
    return aligned_entities


def load_entities(input_path: str) -> list[Entity]:
    entities: list[Entity] = []

    with jsonlines.open(input_path, 'r') as reader:
        for entity_data in reader:
            entities.append(Entity.from_json(entity_data))

    return entities


def dump_entities(entities: list[Entity], output_path: str) -> None:
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(entities)


if __name__ == '__main__':
    # input_path = 'output/top100000_entities_zh.jsonl'
    # output_path = 'output/top100000_entities_zh_aligned.jsonl'
    #
    # entities = load_entities(input_path)
    #
    # aligned_entities = align_entities(entities, limit=10)
    #
    # dump_entities(aligned_entities, output_path)

    entity: Entity = Entity(name='毛泽东')

    print(search_entity(entity))


