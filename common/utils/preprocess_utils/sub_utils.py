import re
import os
import sys
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import chain
from collections import defaultdict, OrderedDict
from typing import List, Union, Tuple, Dict, Optional, Generic, TypeVar
from konlpy.tag import (
    # Okt,  # Java install issue: https://muten.tistory.com/13#recentComments
    Kkma,
    Komoran
)
from transformers import AutoModel

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.dirname(os.path.abspath(os.path.dirname(dirname)))
sys.path.append(dirname)

from common.utils.file_utils import *
from common.constants import *


# OKT = Okt() # '50분에' 와 같은 경우 '에'를 'Foreign'으로 detect 하는 경우들이 생김
KKMA = Kkma()   # '발보나의'와 같은 경우 '의'를 조사로 판단하지 않고, '나의'로 잘라 'NNM"으로 판단해버림.
KOMORAN = Komoran()
        
special_family = ['SF', 'SP', 'SE', 'SO']

char_pair = {
    '‘': '’', 
    "“": "”",
    '(': ')',
    '"': '"',
    '[': ']'
    # '▲': '▲', 
    # '△': '△' 
}

char_pair_rev = {v: k for k, v in char_pair.items()}

excessive_spaces = r'\s{2,}'
hanzi_index = '[一-龥]'

unicode_units = '㎕ ㎖ ㎗ ℓ ㎘ ㏄ ㎣ ㎤ ㎥ ㎦ ㎙ ㎚ ㎛ ㎜ ㎝ ㎞ ㎟ ㎠ ㎡ ㎢ ㏊ ' + \
            '㎍ ㎎ ㎏ ㏏ ㎈ ㎉ ㏈ ㎧ ㎨ ㎰ ㎱ ㎲ ㎳ ㎴ ㎵ ㎶ ㎷ ㎸ ㎹ ㎀ ㎁ ' + \
            '㎂ ㎃ ㎄ ㎺ ㎻ ㎼ ㎽ ㎾ ㎿ ㎐ ㎑ ㎒ ㎓ ㎔ Ω ㏀ ㏁ ㎊ ㎋ ㎌ ㏖ ' + \
            '㏅ ㎭ ㎮ ㎯ ㏛ ㎩ ㎪ ㎫ ㎬ ㏝ ㏐ ㏓ ㏃ ㏉ ㏜ ㏆'
unicode_units = unicode_units.split(' ')

df_dictionary = {
    'Text' : [],
    'Label': []
}

T = TypeVar('T')

model = AutoModel.from_pretrained(f"monologg/kocharelectra-small-discriminator")


#%%

class ClassKMOU:

    pattern = r"<[0-9A-Za-z가-힣\s]+:[A-Z]+>"

    def __init__(self, queue: str, tagged: str):

        # <<BBC:OG>> -> <BBC:OG>
        if '<<' in tagged:
            tagged = tagged.replace('<<', '<').replace('>>', '>')
            queue = queue.replace('<', '').replace('>', '')

        self.tagged = tagged
        matches = re.finditer(self.pattern, self.tagged)

        self.patterns = [m.group() for m in matches]
        self.queue = queue
        self.index_dict_tmp, self.post_dict_tmp = [defaultdict(list) for _ in range(2)] # temporary
        self.ner_dict, self.index_dict, self.post_dict = [defaultdict(list) for _ in range(3)]

        self.arrange_info()

    def arrange_info(self):
        self.patterns = list(OrderedDict.fromkeys(self.patterns)) # ['<서:PS>', '<서:PS>']

        for pattern1 in self.patterns:
            entity_mixed = pattern1.split('<')[1].split(':')
            entity, ner = entity_mixed[0], entity_mixed[1].split('>')[0]
            # entity = entity.replace(' ', '')  # 이토 히로부미
            self.ner_dict[entity].append(ner)
            matches1 = re.finditer(pattern1, self.tagged)
            matches_idcs1 = [match.span() for match in matches1]

            for _ , end in matches_idcs1:

                while True:
                    if self.tagged[end] != '<':
                        next_word_t = self.tagged[end]
                        matches2 = re.finditer(entity, self.queue)
                        matches_idcs2 = [match.span() for match in matches2]
                        cnt = 0

                        while len(matches_idcs2) != cnt:
                            text_span = matches_idcs2[cnt]
                            next_word_q = self.queue[text_span[1]]
                            if next_word_t == next_word_q:
                                self.index_dict_tmp[entity].append(text_span)
                                self.post_dict_tmp[entity].append(next_word_t)
                            cnt += 1
                        break
                    else:
                        end += 1
        
    def remove_subset(self):
        
        def key_len(item):
            return len(item[0])

        ner_dict = dict(sorted(self.ner_dict.items(), key=key_len))
        for _ , ner in enumerate(ner_dict):    # ner : '교육청'
            ner_dict_cp = deepcopy(ner_dict)
            del ner_dict_cp[ner]
            left_keys = list(ner_dict_cp.keys())
            subsets = [x for x in left_keys if ner in x]    # ['지역교육청']

            # 소수의 경우 -> 정확성을 위해 for문을 여러번 쓰기는 함. 
            if len(subsets) != 0:
                index_info1 = self.index_dict_tmp[ner]

                for idx1, (start1, end1) in enumerate(index_info1): 
                    for subset in subsets:
                        index_info2 = self.index_dict_tmp[subset]

                        cnt = 0
                        for start2, end2 in index_info2:
                            if start2 <= start1 and end2 >= end1:
                                cnt += 1
                            
                        if not cnt:
                            self.index_dict[ner].append((start1, end1))
                            self.post_dict[ner].append(self.post_dict_tmp[ner][idx1])
                            
            else:
                self.index_dict[ner].extend(self.index_dict_tmp[ner])
                self.post_dict[ner].extend(self.post_dict_tmp[ner])

    def josa_detector(self, text):
        """
        NAVERx창원대 데이터 셋과는 다르게, 일부 날짜 관련에만 조사가 붙어 있어서, 
        이 경우에는 rule-based로 조사를 탐지하고자 함. 
        """
        if text[-2:] in ['부터', '까지']: return True, text[-2:]
        elif text[-1] == '도': return True, text[-1]
        else: return False, None

    def last_josa_handling(self):
        # ex) 1월5일부터 -> 1월5일
        self.index_dict_cp, self.ner_dict_cp = defaultdict(list), defaultdict(list)
        
        for key, value in self.index_dict.items():
            logical, josa = self.josa_detector(key)
            if logical:
                new_value = []
                for v in value:
                    v_updated = (v[0], v[1]-len(josa))
                    new_value.append(v_updated)
                key_updated = key[:-len(josa)]
                self.ner_dict_cp[key_updated] = self.ner_dict[key]
                self.index_dict_cp[key_updated] = new_value
                continue
            self.index_dict_cp[key].extend(value)
            self.ner_dict_cp[key] = self.ner_dict[key]
        
        self.index_dict, self.ner_dict = self.index_dict_cp, self.ner_dict_cp
        del self.index_dict_cp, self.ner_dict_cp

    @property
    def get_output(self):   # ex) {'교육청': ['OG'], '지역교육청': ['OG']}
        self.remove_subset()
        self.last_josa_handling()
        return self.ner_dict, self.index_dict, self.post_dict
    
    def debug(self):
        cnt = 0
        for key in self.ner_dict.keys():
            value = self.index_dict[key]
            cnt += len(key) * len(list(set(value)))
        return cnt
        

def na_appear(word: str, tag_idx: str) -> bool:
    # ex) 크누스트는
    pos = KOMORAN.pos(word)
    if (len(pos) == 1) and (pos[-1][-1] == 'NA') and word[-1] in '은는이가':
        print(f'{word}: {tag_idx}')
        return True
    return False


def irregular_bio_tag(string: Union[str, int], identity: str) -> List:
    string_len = len(string) if isinstance(string, str) else string
    label = [f'{identity}_I' for _ in range(string_len)]
    label[0] = (label[0][:-1] + 'B')
    return label


def postprocess_terminal(word: str, tag: Optional[str] = None) -> Tuple:

    # OKT, KKMA, KOMORAN 
    # OKT, KKMA => 잘못 잡아내는 경우들이 존재.

    # Okt: https://hyk0425.tistory.com/14
    # Kkma: http://kkma.snu.ac.kr/documents/?doc=postag
    # Komoran: https://docs.komoran.kr/firststep/postypes.html

    pos_print = ''
    josa_family = ['JK', 'JX', 'JC'] 
    last_pos = KOMORAN.pos(word)[-1]

    # output1: 조사, output2: 특수문자
    output1, output2 = (False, ), \
        (True, ) if any(substring in last_pos[1] for substring in special_family) else (False, )  # 특수문자

    if output2[0]:
        word, special = word[:-1], word[-1]
        if special not in '"\'':
            pos_print += special
            if special in '·~-' and tag != '-':
                return (False, '')
        if len(word) == 0:
            return (True, pos_print)

    output1 = (False, )
    while True:

        if tag is not None: 
            temp = KOMORAN.pos(word)
            if 'NUM' in tag:
                # 숫자와 관련된 클래스의 경우, 기존의 RULE과 반하는 경우들이 생긴다. 
                if len(temp) != 1:
                    front, back = temp[:-2], temp[-2:]
                    front_bind = list(zip(*front))
                    back_bind = list(zip(*back))
                    if ('VC' in back_bind[1][0]) and (back_bind[1][1][0] == 'E'):
                        # if back_bind[0][-1] not in 'ㄴㄹ':
                        front = ''.join(front_bind[0])
                        back = word[word.find(front)+len(front):]
                        word = front
                        pos_print += back
                        output1 = (bool(sum((True, ) + output1)), )

            # KOMORAN으로 조사 Parsing을 할 수 없는 경우
            if temp[-1][0] in exceptional:
                if temp[-1][-1] in ['NNP', 'NNB']:
                    while True: 
                        next_temp = KOMORAN.pos(temp[-1][0])
                        if len(next_temp[-1][0]) != len(temp[-1][0]):
                            if any(substring in next_temp[-1][1] for substring in josa_family):
                                pos_print += next_temp[-1][0]
                                output1 = (bool(sum((True, ) + output1)), )
                                break
                        else:
                            pos_print += temp[-1][0]
                            output1 = (bool(sum((True, ) + output1)), )
                            break
            
        tagged = KOMORAN.pos(word)
        front, back = tagged[:-1], tagged[-1]

        if back[0] in 'ㄴㄹ': # 구호션 -> 구효서 + ㄴ
            break

        front_exist = (len(front) != 0)

        # MM: '강성범(15점)이' <- OKT, KKMA, KOMORAN 중 그 어느 것도 '이'를 잡아내지 못하는 것을 방지
        if front_exist:
            josa_family.append('MM')

        if any(substring in back[1] for substring in josa_family):
            output1 = (bool(sum((True, ) + output1)), )
            pos_print += back[0]
            if not front_exist:
                break
        else:
            break

        josa_family = ['JK', 'JX', 'JC']

        # 달루완을 -> [('덜', 'MAG'), ('루', 'NNG'), ('와', 'JKB'), ('ㄴ', 'JX'), ('을', 'JKO')]
        if (front[-1][0] in 'ㄴㄹ') or (len(front[-1][0]) == 1 and 'NN' in front[-1][1]):   
            break

        word = ''.join(list(zip(*front))[0])

    # exceptional case
    if '이었다' in word:
        return (True, '이었다')

    return (bool(sum(output1 + output2)), ) + (pos_print, )


def revise_label(tags: List, 
                 word: str, 
                 logical: bool, 
                 pos: str) -> List:
    
    if logical:
        ot_length = len(pos)
        tags[-ot_length:] = [O_TAG] * ot_length
    
    # ex) '"중"
    frontal = word[0] if len(pos) == 0 else word[:-len(pos)][0]
    if frontal[0] in "“‘([":  
        open_length = len([x for x in word if x == frontal[0]])
        close_length = len([x for x in word if x == char_pair[frontal[0]]])
        if (open_length == close_length):
            return tags
        else: 
            tags = [tags[0]] + tags[2:]
            tags.insert(0, O_TAG)
    elif frontal[0] in "▲△-\"": # △필리핀
        if len(word) == 1: 
            return tags
        tags = [tags[0]] + tags[2:]
        tags.insert(0, O_TAG)
            
    return tags


def naver_operation_chunk(word: str, 
                          tag_entity: str, 
                          tag: str) -> List:
    tags_temp = irregular_bio_tag(len(word), tag_entity)
    if word in determined:  # Parsing이 rule-based로 해결될 수 없는 경우 그냥 명시를 함. 
        pos_revised = determined[word]
    else:
        pos_revised = postprocess_terminal(word, tag)
    tags = revise_label(tags_temp, word, *pos_revised)
    return tags


def consider_previous(contiguous_info: List, 
                     tags: List) -> List:
    if len(contiguous_info) > 0:    # B -> O
        contiguous_cp = contiguous_info.copy()
        for idx, element in enumerate(contiguous_cp):
            if element[-1] == 'B' and idx != 0:
                contiguous_info[idx] = contiguous_info[idx].replace('_B', '_I')
        if contiguous_info[-1] != 'O':
            tags[0] = tags[0].split('_')[0] + '_I' if tags[0] != 'O' else 'O'
        contiguous_info.extend(tags)
        tags = contiguous_info
    return tags


def split_by_enter(corpus: List) -> List:
    valid, valid_list = [], []
    for c in corpus:
        if c != '\n':
            valid.append(c)
        else:
            valid_list.append(valid)
            valid = []
    return valid_list


def strip_remove(text):
    text = text.strip()
    text = re.sub(excessive_spaces, ' ', text)
    return text


def return_hanzi_index(sentence: str) -> List:
    return [x.span() for x in re.finditer(hanzi_index, sentence)]


def return_time_josa_index(sentence: str) -> List:
    spans = []
    spans.extend([x.span() for x in re.finditer('부터', sentence)])
    spans.extend([x.span() for x in re.finditer('까지', sentence)])
    spans.extend([x.span() for x in re.finditer('부턴', sentence)])
    return spans


def remove_hanzi_and_josa(sentence: str, new_tags_vis: List) -> List:
    hanzi_index = return_hanzi_index(sentence)
    josa_index = return_time_josa_index(sentence)
    caution_index = hanzi_index + josa_index
    if len(caution_index) != 0:
        for start, end in caution_index:
            try:
                if sentence[end] == ' ' and new_tags_vis[end] != 'O':
                    new_tags_vis[end] = 'O' # '지난해부터 2021년까지' 와 같은 경우를 고려 
            except: # 조사가 나오고 문장이 끝나는 경우. 
                pass
            new_tags_vis[start:end] = ['O'] * (end-start)
    del hanzi_index, josa_index

    return new_tags_vis


def flatten(int_list: List) -> List:
    flatten_list = []
    for il in int_list:
        if isinstance(il, int):
            flatten_list.append(il)
        else:
            flatten_list.extend(il)
    return flatten_list


def show_all(results: List) -> None:
    # For debugging
    for result in results:
        print(result, sep='\n')
    print('====================')


#%%

# NOTE: Stats & Figures

def draw_subplot(axs: np.ndarray, 
                idx: int, 
                subidx: int, 
                data_name: str, 
                keys: List, 
                values: List, 
                colors: List) -> Generic[T]:
    axs[idx, subidx].bar(keys, values, color=colors)    
    axs[idx, subidx].set_title(f"# of entities/{['tokenized', 'raw'][subidx]} in {data_name}", fontsize=10)
    axs[idx, subidx].tick_params(axis='x', labelsize=5)
    axs[idx, subidx].set_ylim(0, max(values) * 1.5)
    axs[idx, subidx].annotate(f'# of data: {sum(values)}', xy=(3.5, max(values) * 1.2), fontsize=5)

    for i, y_value in enumerate(values):
        axs[idx, subidx].text(keys[i], y_value, str(y_value), horizontalalignment='center', fontsize=5)

    return axs


def plotting(fig_path: str, 
             overall_info: Dict) -> None:

    def df_correction(df_temp, df, col_name):
        df_temp = pd.DataFrame(df_temp).T
        df_temp.rename(index={0:col_name}, inplace=True)
        df_temp.columns = df.columns
        df = pd.concat((df, df_temp), axis=0)
        return df

    dfs = []
    total_cnt1, total_cnt2 = [0] * 6, [0] * 6

    data_keys = list(overall_info.keys())
    nrow = len(data_keys)

    colors = sns.color_palette('hls', 6)
    fig, axs = plt.subplots(nrow+1, 2)

    dict1_O_cnt = []
    sentence_cnt = []

    for idx, dk in enumerate(data_keys):
        data = overall_info[dk]
        sen_len = data['sentence_cnt']
        count_dict1, count_dict2 = data['tokenized'], data['not_tokenized']
        O_cnt = count_dict1.pop('O')
        dict1_O_cnt.extend([O_cnt, 0])
        sentence_cnt.extend([sen_len]*2)
        del count_dict1['TOTAL'], count_dict2['TOTAL']

        
        df = pd.DataFrame([count_dict1, count_dict2]).T.fillna(0).astype(int)
        df.columns = [f'{dk} entities/tokenized', f'{dk} entities/raw']
        dfs.append(df)

        keys = df.index.tolist()
        
        info1, info2 = df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()

        total_cnt1 = list(map(lambda x, y: x+y, total_cnt1, info1))
        total_cnt2 = list(map(lambda x, y: x+y, total_cnt2, info2))

        axs = draw_subplot(axs, idx, 0, dk, keys, info1, colors)
        axs = draw_subplot(axs, idx, 1, dk, keys, count_dict2.values(), colors)

    axs = draw_subplot(axs, len(overall_info), 0, 'total', keys, total_cnt1, colors)
    axs = draw_subplot(axs, len(overall_info), 1, 'total', keys, total_cnt2, colors)

    fig.tight_layout(h_pad=0.1, w_pad=0.1)

    dfs = pd.concat(dfs, axis=1)

    dfs = df_correction(dict1_O_cnt, dfs, 'O')
    dfs = df_correction(dfs.sum(axis=0), dfs, 'TOTAL')
    dfs = df_correction(sentence_cnt, dfs, 'SENTENCE_LEN')

    os.system(f'mkdir -p {fig_path}')

    dfs.to_csv(os.path.join(fig_path, 'df.csv'))
    fig.savefig(os.path.join(fig_path, 'fig.png'), dpi=3000)

    plt.close()


def get_tokenized_stats(tokenized: List) -> List:
    total_cnt, tokenized_cnt = 0, defaultdict(int)
    for instance in tokenized:
        not_o = [x for x in instance if x != 'O']
        for no in not_o:
            tokenized_cnt[no[:3]] += 1
        total_cnt += len(instance)
    valid_sum = sum(list(tokenized_cnt.values()))
    tokenized_cnt['O'] = total_cnt - valid_sum
    tokenized_cnt['TOTAL'] = total_cnt
    return tokenized_cnt


def get_raw_stats(not_tokenized: List) -> Dict:
    total_cnt, not_tokenized_cnt = 0, defaultdict(int)
    for nt in not_tokenized:
        for key, value in nt.items():
            not_tokenized_cnt[key] += value
            total_cnt += value
    not_tokenized_cnt['TOTAL'] = total_cnt  
    return not_tokenized_cnt


def get_public_dset_stats(out_path: str, 
                          fig_path: str) -> None:

    data = dict()
    list_dir = os.listdir(out_path)

    # File open
    for ld in list_dir:
        path = os.path.join(out_path, ld)
        datum = read_file(path)
        dname = ld.split('.')[0]
        data[dname] = datum

    # 통계량 정리
    overall_info = dict()
    for dname, v in data.items():
        overall_info[dname] = dict()

        sentence, tokenized_ne, not_tokenized_ne = v

        tokenized_dict = get_tokenized_stats(tokenized_ne)
        not_tokenized_dict = get_raw_stats(not_tokenized_ne)

        if dname == 'kmou':
            tokenized_dict['NUM'] = 0
            not_tokenized_dict['NUM'] = 0

        tokenized_dict = dict(sorted(tokenized_dict.items()))
        not_tokenized_dict = dict(sorted(not_tokenized_dict.items()))
        
        overall_info[dname]['tokenized'] = tokenized_dict
        overall_info[dname]['not_tokenized'] = not_tokenized_dict
        overall_info[dname]['sentence_cnt'] = len(tokenized_ne)
        
    plotting(fig_path, overall_info)


def judge_o_tag_only(data: List) -> bool:
    total_length = len(data)
    O_length = len([x for x in data if x == 'O'])
    return (total_length == O_length)


def balance_otag_with_others(datum: Tuple) -> Tuple:

    label_only = datum[1]
    o_tag_only = [i for i, x in enumerate(label_only) if judge_o_tag_only(x)]

    print(f'O_tag_only: {len(o_tag_only)}, Remainder: {len(label_only)-len(o_tag_only)}')
    
    if len(o_tag_only) > len(label_only):
        raise ValueError('Need to consider this case.')

    return datum


def update_dictionary(datum: List) -> Dict:
    text, label = datum[:2]
    
    df_dictionary['Text'].extend(text)
    df_dictionary['Label'].extend(label)
        
    return df_dictionary


def make_df(data: Dict) -> pd.DataFrame:

    df_dict = {
        'Text': data['Text'], 
        'Label': data['Label']
    }

    df = pd.DataFrame(df_dict)

    shuffled_idx = np.random.permutation(df.index)
    df = df.reindex(shuffled_idx)

    return df


def remove_longer_than_max_length(data: Dict, fn: str, cnt: int) -> Dict:
    
    text, label = data['Text'], data['Label']
    fd = fn.split('.')[0]

    valid_idx = [i for i, x in enumerate(label) if len(x) <= model.config.max_position_embeddings-2]    # CLS, SEP

    original_length = len(text) - cnt
    alive_length = len(valid_idx) - cnt

    text = list(map(lambda x: text[x], valid_idx))
    label = list(map(lambda x: label[x], valid_idx))

    data['Text'] = text
    data['Label'] = label
    
    print(f'{fd}: {alive_length} data is alive among {original_length} data')
    print('================================================================')

    return data
    

def data_split_and_gather(in_path: str) -> None:

    out_path = os.path.join(in_path, 'preprocessed')
    split_path = os.path.join(in_path, 'completed')

    data = []
    cnt = 0

    for fn in sorted(os.listdir(out_path)):
        full_path = os.path.join(out_path, fn)
        if os.path.isdir(full_path):
            continue
        datum = list(read_file(full_path))
        datum = balance_otag_with_others(datum)
        datum = update_dictionary(datum)
        datum = remove_longer_than_max_length(datum, fn, cnt)
        cnt += len(datum['Text'])
        
    data = make_df(datum)    
          
    total_len = data.shape[0]

    train_data = data.iloc[:int(total_len*0.9), :]
    valid_data = data.iloc[int(total_len*0.9):]

    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)

    train_data['Label'] = train_data['Label'].str.join(' ')
    valid_data['Label'] = valid_data['Label'].str.join(' ')

    os.makedirs(split_path, exist_ok=True)

    write_file(train_data, split_path+'/train_data', 'parquet')
    write_file(valid_data, split_path+'/valid_data', 'parquet')


def calculate_unk(in_path: str, 
                  tokenizer: Generic[T]):

    """
    UNK의 발생빈도를 확인하기 위함. 
    """

    completed_path = in_path + '/completed'
    unk_path = in_path + '/unk'

    os.system(f'mkdir -p {unk_path}')

    train_data = read_file(completed_path + '/train_data.parquet')
    valid_data = read_file(completed_path + '/valid_data.parquet')

    data_tolist = train_data['Text'].tolist()
    data_tolist += valid_data['Text'].tolist()
    
    unk_token_id = tokenizer.tokenizer.unk_token_id

    tokenized = list(map(lambda x: tokenizer.tokenizer(x)['input_ids'], data_tolist))

    tokenized_cat = list(chain(*tokenized))
    
    unk_included_overall = len([x for x in tokenized_cat if unk_token_id == x])
    unk_included_sentence_idx = [i for i, x in enumerate(tokenized) if unk_token_id in x]
    unk_included_sentence_seq = [x for i, x in enumerate(tokenized) if unk_token_id in x]

    original_sentence = list(map(lambda x: data_tolist[x], unk_included_sentence_idx))
    re_translate = [tokenizer.tokenizer.decode(x) for x in unk_included_sentence_seq]

    write_file([original_sentence, re_translate], unk_path, 'txt')    

    print(f'[UNK] 토큰 수 / 전체 토큰: {unk_included_overall} / {len(tokenized_cat)} ( {unk_included_overall/len(tokenized_cat) * 100}% )')
    print(f'[UNK] 토큰 포함한 문장 수 / 전체 문장: {len(unk_included_sentence_idx)} / {len(tokenized)} ( {len(unk_included_sentence_idx) / len(tokenized) * 100}% )')