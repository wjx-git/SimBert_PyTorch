import json
import re


def to_simbert_data():
    """根据蚂蚁金服的竞赛数据构造simBert训练数据"""
    fr = open('atec/atec_nlp_sim_train_all.csv', 'r', encoding='utf-8')
    fw = open('data_similarity.json', 'a', encoding='utf-8')
    for line in fr.readlines():
        item = line.strip().split('\t')
        if item[-1] == '1' and not re.search(r'\*\*\*', item[1]+item[2]):
            json.dump({"text": item[1], "synonyms": [item[2]]}, fw, ensure_ascii=False)
            fw.write('\n')
    fw.close()
    fr.close()


if __name__ == '__main__':
    to_simbert_data()
