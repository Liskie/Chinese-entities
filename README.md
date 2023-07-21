# Top 10000 Popular Chinese Entities

执行步骤：
1. 下载 zh-wikidump
2. WikiExtractor 提取 dump 中的 text
3. OpenCC 将全部文本转为简体中文
    ```shell
    python t2s.py
    ```
4. Stanza 对文本进行 NER，统计实体的出现频数 (4*RTX TITAN XP 24G 花费 7.5 小时完成，可优化)
    ```shell
    python ner.py
    ```
   or if you have multiple GPUs:
    ```shell
    python ner_multiprocess.py
    ```
5. 按频数从高到低选取前 10000 个中文实体
    ```shell
    python merge_dicts.py
    python filter.py 
    ```