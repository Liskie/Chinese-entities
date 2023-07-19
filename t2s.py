import os
import opencc
from tqdm import tqdm

def convert_to_simplified_chinese(input_path, output_path):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create an OpenCC converter for traditional Chinese to simplified Chinese
    converter = opencc.OpenCC('t2s')

    for item in tqdm(os.listdir(input_path), desc="Processing files"):
        item_path = os.path.join(input_path, item)
        output_item_path = os.path.join(output_path, item)

        if os.path.isdir(item_path):
            # If it's a subdirectory, recursively call the function
            convert_to_simplified_chinese(item_path, output_item_path)
        else:
            # If it's a file, convert its content to simplified Chinese and save to output directory
            with open(item_path, 'r', encoding='utf-8') as infile:
                content = infile.read()

            simplified_content = converter.convert(content)

            with open(output_item_path, 'w', encoding='utf-8') as outfile:
                outfile.write(simplified_content)

if __name__ == "__main__":
    input_dir = "data/text"
    output_dir = "output/extracted_texts"

    convert_to_simplified_chinese(input_dir, output_dir)

