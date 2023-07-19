import json
import os
import stanza
from stanza import DownloadMethod
from tqdm import tqdm


def ner_from_files(input_path):
    # Create a NER processor for Chinese
    nlp = stanza.Pipeline(lang='zh', processors='tokenize,ner', use_gpu=True,
                          download_method=DownloadMethod.REUSE_RESOURCES)

    # Dictionary to store named entities and their counts
    entities_dict = {}

    for root, _, files in tqdm(os.walk(input_path), desc='Total: '):
        tqdm.write(f'root {root} {files}')
        for file in tqdm(files, desc='Subdir: '):

            file_path = os.path.join(root, file)
            tqdm.write(f'subdir {file} {file_path}')

            with open(file_path, 'r', encoding='utf-8') as infile:
                text = infile.read()

            # Process the text with NER
            doc = nlp(text)

            # Collect named entities and update the count in the dictionary
            for sent in doc.sentences:
                for entity in sent.ents:
                    entity_text = entity.text
                    entity_label = entity.type

                    # Update the count of named entities in the dictionary
                    if entity_label in entities_dict:
                        entities_dict[entity_label] += 1
                    else:
                        entities_dict[entity_label] = 1

    return dict(sorted(entities_dict.items(), key=lambda item: item[1], reverse=True))


if __name__ == "__main__":
    input_dir = "output/extracted_texts"

    entity2count = ner_from_files(input_dir)

    print("Named Entities and their counts:")
    for entity, count in entity2count.items():
        print(f"{entity}: {count} times")

    with open('output/entity2count.json', 'w') as writer:
        json.dump(entity2count, writer)


