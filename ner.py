import json
import os
import stanza
import torch
from stanza import DownloadMethod
from tqdm import tqdm

def ner_from_files(input_path):
    # Dictionary to store named entities and their counts
    entities_dict = {}

    for root, _, files in tqdm(os.walk(input_path), total=len(os.listdir(input_path)), desc='Total: '):
        for file in tqdm(files, desc='Subdir: '):
            # Create a NER processor for Chinese with GPU enabled
            nlp = stanza.Pipeline(lang='zh', processors='tokenize,ner', use_gpu=True,
                                  download_method=DownloadMethod.REUSE_RESOURCES,
                                  logging_level='WARN')

            file_path = os.path.join(root, file)

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

            # Release GPU resources for each document
            torch.cuda.empty_cache()

    return dict(sorted(entities_dict.items(), key=lambda item: item[1], reverse=True))


if __name__ == "__main__":
    input_dir = "output/extracted_texts"

    entity2count = ner_from_files(input_dir)

    print("Named Entities and their counts:")
    for entity, count in entity2count.items():
        print(f"{entity}: {count} times")

    with open('output/entity2count.json', 'w') as writer:
        json.dump(entity2count, writer)


