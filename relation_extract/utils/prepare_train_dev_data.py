import os
import json
import re
import sys
import copy

CUR_DIR=os.path.dirname(os.path.abspath(__file__))
SAVE_DIR=os.path.join(os.path.dirname(CUR_DIR),'train_data')
TRAIN_FILE=os.path.join(SAVE_DIR,'train.txt')
DEV_FILE=os.path.join(SAVE_DIR,'dev.txt')
RELATION_FILE=os.path.join(SAVE_DIR,'relations.json')

if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)

sys.path.append(os.path.dirname(CUR_DIR))

from basic_models import tokenization
import basic_config

config=basic_config.Config()
tokenizer=tokenization.FullTokenizer(config.vocab_file)


# 将数据转化为用于训练的格式
#  w1 w2 w3 w4 ... \t entity1_start entity2_start class1 ...
def get_train_data():
    raw_train_file='/home/wangjian0110/data/baidu_kg/train_data.json'

    total_relations=set() #所有的关系
    total_tokens=[] # 所有的规整化字符
    total_pair_pos_info=[] # 所有的实体位置信息,{'实体位置'：关系名}

    # 用于训练
    for line in read_lines(raw_train_file):
        line=json.loads(line)
        tokens,entity2pos,pairs,relations=convert_to_train_line(line)
        total_relations=total_relations|relations
        pos_info={}
        for e1,e2 in pairs:
            if (e1 not in entity2pos) or (e2 not in entity2pos):
                continue
            else:
                e1_pos=entity2pos[e1]
                e2_pos=entity2pos[e2]
                pos_info[(e1_pos,e2_pos)]=pairs[(e1,e2)]
        if len(pos_info)>0:
            total_tokens.append(tokens)
            total_pair_pos_info.append(pos_info)
    
    total_relations=sorted(list(total_relations))
    relation2id=get_relation_to_id(total_relations)
    with open(RELATION_FILE,'w',encoding='utf8') as f:
        print("Relation number:{}".format(len(relation2id)))
        print("Save relation2id...")
        json.dump(relation2id,f)
    assert len(total_tokens)==len(total_pair_pos_info)
    sample_num=0
    relation_num=0
    with open(TRAIN_FILE,'w',encoding='utf8') as f:
        for tokens,entity_info in zip(total_tokens,total_pair_pos_info):
            sample_num+=1
            relation_num+=len(entity_info)
            line=get_one_line_sample(tokens,entity_info,relation2id)
            f.write(line+'\n')
    print("Train file has {} samples, {} relations.".format(sample_num,relation_num))
def get_dev_data():
    raw_dev_file='/home/wangjian0110/data/baidu_kg/dev_data.json'

    total_tokens=[] # 所有的规整化字符
    total_pair_pos_info=[] # 所有的实体位置信息,{'实体位置'：关系名}

    with open(RELATION_FILE,'r',encoding='utf8') as f:
        relation2id=json.load(f)

    # 验证集
    for line in read_lines(raw_dev_file):
        line=json.loads(line)
        tokens,entity2pos,pairs,relations=convert_to_train_line(line)
        pos_info={}
        for e1,e2 in pairs:
            if (e1 not in entity2pos) or (e2 not in entity2pos):
                continue
            else:
                e1_pos=entity2pos[e1]
                e2_pos=entity2pos[e2]
                pos_info[(e1_pos,e2_pos)]=pairs[(e1,e2)]
        if len(pos_info)>0:
            total_tokens.append(tokens)
            total_pair_pos_info.append(pos_info)
    
    assert len(total_tokens)==len(total_pair_pos_info)
    sample_num=0
    relation_num=0
    with open(DEV_FILE,'w',encoding='utf8') as f:
        for tokens,entity_info in zip(total_tokens,total_pair_pos_info):
            sample_num+=1
            relation_num+=len(entity_info)
            line=get_one_line_sample(tokens,entity_info,relation2id)
            f.write(line+'\n')
    print("Dev file has {} samples, {} relations.".format(sample_num,relation_num))

def get_one_line_sample(tokens,entity_pos_info,relation2id):
    token_text=" ".join(tokens)+'\t'
    info_string=''
    for entity_pos in entity_pos_info:
        relation_id=relation2id[entity_pos_info[entity_pos]]
        e1_pos,e2_pos=entity_pos
        info_string+=str(e1_pos[0])+" "+str(e1_pos[1])+" "+str(e2_pos[0])+" "+str(e2_pos[1])+" "+str(relation_id)+' '
    return token_text+info_string

def convert_to_train_line(input_line):
    relations=set()
    text=normalize_text(input_line['text'])
    spo_list=input_line['spo_list']
    entity=set() #实体
    pairs={} # 实体对
    for sub_list in spo_list:
        e1=sub_list['object'].lower().strip()
        e2=sub_list['subject'].lower().strip()
        r_name=sub_list['predicate'].strip()
        relations.add(r_name)
        entity.add(normalize_text(e1))
        entity.add(normalize_text(e2))
        pairs[(e1,e2)]=r_name
    tokens,entity2pos=convert_tokens(text,entity)
    return tokens,entity2pos,pairs,relations
def convert_tokens(raw_string,total_entity):
    raw_raw_string=raw_string
    # 剔除异常实体
    all_entity=copy.copy(total_entity)
    for entity in list(total_entity):
        n_entity=regex_normal(entity)
        if not re.search(n_entity, raw_string):
            print("Search entity error!")
            print('raw_string:{}'.format(raw_string))
            print('entity:{}'.format(entity))
            all_entity.discard(entity)
        else:
            # 在实体的前后同为英文或者同为数字，加空格
            span=re.search(n_entity,raw_string).span()
            s_status=False
            e_status=False
            if span[0]!=0:
                if raw_string[span[0]].isdigit() and raw_string[span[0]-1].isdigit():
                    s_status=True
                if ('a'<=raw_string[span[0]]<='z')  and ('a'<=raw_string[span[0]-1]<='z'):
                    s_status=True
            if span[1]!=len(raw_string):
                if raw_string[span[1]].isdigit() and raw_string[span[1]-1].isdigit():
                    e_status=True
                if ('a'<=raw_string[span[1]]<='z') and ('a'<=raw_string[span[1]-1]<='z'):
                    e_status=True
            s1=raw_string[:span[0]]+" "+raw_string[span[0]:span[1]] if s_status else raw_string[:span[1]]
            s2=" "+raw_string[span[1]:] if e_status else raw_string[span[1]:]
            raw_string=s1+s2
            raw_string=re.sub('\s+',' ',raw_string)
    # 将原始字符串转化为token
    tokens=tokenizer.tokenize(raw_string)

    entity2pos={}
    for entity in all_entity:
        entity_tokens=tokenizer.tokenize(entity)
        start=None
        end=None
        for index in range(len(tokens)):
            sub=tokens[index:index+len(entity_tokens)]
            if entity_tokens==sub:
                start=index
                end=index+len(entity_tokens)
        if start is None:
            print("Match entity ERROR!")
            print('Raw tokens{}'.format(tokens))
            print('ENtity:{}'.format(entity_tokens))
        else:
            entity2pos[entity]=(start,end)
    return tokens,entity2pos


def read_lines(file):
    with open(file,'r',encoding='utf8') as f:
        for line in f:
            line=line.strip()
            if line:
                yield line
def regex_normal(input_string):
    e=input_string
    e=re.sub('\\\\','\\\\\\\\',e)
    e=re.sub('\\(','\\(',e)
    e=re.sub('\\)','\\)',e)
    e=re.sub('\\+','\\+',e)
    e=re.sub('\\]','\\]',e)
    e=re.sub('\\[','\\[',e)
    e=re.sub('\\*','\\*',e)
    e=re.sub('\\?','\\?',e)
    e=re.sub('\\.','\\.',e)
    e=re.sub('\\^','\\^',e)
    e=re.sub('\\$','\\$',e)
    e=re.sub('\\|','\\|',e)
    e=re.sub('\\}','\\}',e)
    e=re.sub('\\{','\\{',e)
    return e

def get_relation_to_id(relation_list):
    relation2id={}
    for r in relation_list:
        relation2id[r]=len(relation2id)
    return relation2id

def normalize_text(input_text):
    # 规整化输入文本
    # 小写
    input_text=input_text.lower()
    #英文，数字前后加空格
    input_text=re.sub("([0-9]+)",r' \1 ',input_text)
    input_text=re.sub("([a-z]+)",r' \1 ',input_text)
    input_text=re.sub("\s+",' ',input_text)
    return input_text.strip()
if __name__=="__main__":
    get_train_data()
    get_dev_data()