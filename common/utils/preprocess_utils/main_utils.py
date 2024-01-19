import os
import re
import sys
import pandas as pd
from copy import deepcopy
from typing import List, Dict, TypeVar, Generic

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)

from common.constants import *
from common.utils.preprocess_utils.sub_utils import *
from itertools import chain
from collections import defaultdict


T = TypeVar('T')


pd.set_option('display.max_rows', None) # For debugging


#%%
        

out_dicts = {
    'c1_s': [], 'c2_s': [], 'c3_s': [], 'c4_s': [], # sentences
    'c1_t': [], 'c2_t': [], 'c3_t': [], 'c4_t': [], # tags w/o tokenization
    'c1_w': [], 'c2_w': [], 'c3_w': [], 'c4_w': []  # tags w/ tokenization
}   


#%% 

def iterate_naver(item: Generic[T], 
                tokenizer: Generic[T], 
                need_debug: bool):

    """
    Example

    item
        - text: '비토리오 양일 만에 영사관 감호 용퇴, 항룡 압력설 의심만 가율 '
        - words: ['비토리오', '양일', '만에', '영사관', '감호', '용퇴,', '항룡', '압력설', '의심만', '가율']
        - tags: ['PER_B', 'DAT_B', '-', 'ORG_B', 'CVL_B', '-', '-', '-', '-', '-']
    """
  
    text, words, tags = item.text, item.words, item.tags

    assert text.strip() == ' '.join(words), 'Sth wrong on text part'
    assert len(words) == len(tags), 'Sth wrong on original data'

    tags.append('TMP')  # Added for iteration 

    sentence_tokenized = tokenizer.tokenize(text)

    new_tags = []
    word_tagged = defaultdict(int)    # For stats
    word_accum, flag = '', True
    contiguous_info = []

    for idx, word in enumerate(words):
        word_tokenized = ''.join(tokenizer.tokenize(word))

        # 레이블링이 너무 이상한 것들이 등장하는 경우
        if any(case in word_tokenized for case in incomprehensible):
            return 

        try:
            tag_entity, tag_bio = tags[idx].split('_')
            if tag_entity in NE_TARGET:
                word_accum += word_tokenized
                word_tagged[tag_entity] += 1

                if na_appear(word_tokenized, tags[idx]):
                    return

                flag = True
                try:
                    next_tag_entity, next_tag_bio = tags[idx+1].split('_')
                    if (tag_entity == next_tag_entity):
                        flag = not ( (tag_bio == 'B' and next_tag_bio == 'I') or \
                                    (tag_bio == 'I' and next_tag_bio == 'I') ) 
                        if not flag:
                            tags_temp = naver_operation_chunk(word_tokenized,
                                                            tag_entity, 
                                                            tags[idx])
                            contiguous_info.extend(tags_temp)                   
                except: 
                    pass

                if flag:
                    tags_temp = naver_operation_chunk(word_tokenized,
                                                    tag_entity, 
                                                    tags[idx])
                    new_tags.extend(consider_previous(contiguous_info, tags_temp))
                    word_accum, flag = '', True
                    contiguous_info = []
            else:
                new_tags.extend([O_TAG] * len(word_tokenized))

        except: # '-'
            new_tags.extend([O_TAG] * len(word_tokenized))  
        
    new_tags = [LABEL[x] for x in new_tags]
    new_tags_vis = list(map(lambda x: LABEL_INV[x], new_tags))

    assert len(sentence_tokenized) == len(new_tags), 'Unexpected Error'

    out_dicts['c1_s'].append(text)
    out_dicts['c1_t'].append(new_tags_vis)
    out_dicts['c1_w'].append(word_tagged)

    if need_debug:
        show_all([pd.DataFrame([sentence_tokenized, new_tags_vis]).T, tags, words])


#%%

def iterate_nikl(items: List[Dict], 
                tokenizer: Generic[T], 
                need_debug: bool):

    """
    Example

    items[0]
        - 'id': 'NLRW2100000013.1.1.1'
        - 'form': '태안군의회, 2019년‘군민중심’의정성과 빛났다!'
        - 'word': [{'id': 1, 'form': '태안군의회,', 'begin': 0, 'end': 6}, {'id': 2, 'form': '2019년‘군민중심’의정성과', 'begin': 7, 'end': 22}, {'id': 3, 'form': '빛났다!', 'begin': 23, 'end': 27}]
        - 'NE': [{'id': 1, 'form': '태안군의회', 'label': 'OGG_POLITICS', 'begin': 0, 'end': 5}, {'id': 2, 'form': '2019년', 'label': 'DT_YEAR', 'begin': 7, 'end': 12}]
    """

    new_tags = []
    word_tagged = defaultdict(int)    # For stats

    for item in items:
        sentence, word, ne = item['form'], item['word'], item['NE']

        # 단위 Unicode를 포함하고 있거나 
        # 공백이 두 개 이상이라 Rule을 깨버려 전처리를 어렵게 만들거나, 
        # 만 나이 표현으로 인해 noise가 끼는 경우 삭제 ( 만10~19세 : 만10, 19세 )
        if any([unit for unit in unicode_units if unit in sentence]) or \
            len(re.findall(excessive_spaces, sentence)) != 0 or \
            len([x.span() for x in re.finditer('만[0-9]+', sentence)]):
            continue

        sentence = strip_remove(sentence)  #  ' 수산·어촌분야 예산이...' 

        new_tags = [LABEL[O_TAG]] * len(sentence)
        nes = [] # For debugging

        for ne_item in ne:
            form, label, begin, end = ne_item['form'], ne_item['label'][:2], ne_item['begin'], ne_item['end']
            form = re.sub(excessive_spaces, ' ', form)

            if label in LABEL_ALIGN:
                label = LABEL_ALIGN[label]
                tag_values = irregular_bio_tag(end - begin, label)
                new_tags[begin : end] = list(map(lambda x: LABEL[x], tag_values))
                nes.append(ne_item)
                word_tagged[label] += 1

        new_tags_vis = list(map(lambda x: LABEL_INV[x], new_tags))

        assert len(tokenizer.tokenize(sentence)) == len(new_tags), 'Unexpected Error'

        new_tags_vis = remove_hanzi_and_josa(sentence, new_tags_vis)

        out_dicts['c2_s'].append(sentence)
        out_dicts['c2_t'].append(new_tags_vis)
        out_dicts['c2_w'].append(word_tagged)

        if need_debug:
            show_all([pd.DataFrame([tokenizer.tokenize(sentence), new_tags_vis]).T, item['form'], nes])


#%% 

def iterate_kmou(items: List, 
                tokenizer: Generic[T], 
                need_debug: bool):

    """
    Example

    items:
        - index 0: '; 나도 때늦은 홍길동이보다는 이 사...쯤은 알고 있오.\n'
        - index 1: '$나도 때늦은 <홍길동:PS>이보다는...쯤은 알고 있오.\n'
        - index 2, 3, ... :  '1\t나\tNP\tO\n', '1\t도\tJX\tO\n', ...
            - 종성이 분리되어 기록되거니, '했'을 '하', '았'으로 분리하여 기록되어있음.
    """

    word_tagged = defaultdict(int)    # For stats

    original, tagged = items[:2]
    # individual_tags = [x.strip() for x in items[2:]]

    assert original[:2] == '; ', 'Unexpected case appears.'
    assert tagged[0] == '$', 'Unexpected case appears.'

    original = strip_remove(original[2:])
    tagged = strip_remove(tagged[1:])

    if any([unit for unit in unicode_units if unit in original]):
        return

    sentence_tokenized = tokenizer.tokenize(original)

    CKMOU = ClassKMOU(original, tagged)

    ner_info, idx_info, next_info = CKMOU.get_output

    new_tags = [O_TAG] * len(sentence_tokenized)

    for key, value in ner_info.items(): # {'홍길동': ['PS']}

        if len(value) != 0:
            assert len(list(set(value))) == 1, 'At least all same.'

        v = LABEL_ALIGN[value[0]]
        word_tagged[v] += 1

        iis, _ = idx_info[key], next_info[key]
        
        for ii in iis:
            start, end = ii
            ners = irregular_bio_tag(end-start, v)
            new_tags[start:end] = ners

    new_tags = list(map(lambda x: LABEL[x], new_tags))
    new_tags_vis = list(map(lambda x: LABEL_INV[x], new_tags))

    assert CKMOU.debug() == len([x for x in new_tags if x != 1]), 'Unexpected Error'

    new_tags_vis = remove_hanzi_and_josa(original, new_tags_vis)

    out_dicts['c3_s'].append(original)
    out_dicts['c3_t'].append(new_tags_vis)
    out_dicts['c3_w'].append(word_tagged)

    if need_debug:
        show_all([pd.DataFrame([sentence_tokenized, new_tags_vis]).T, tagged, original])



#%%

def iterate_klue(items: Dict, 
                 tokenizer: Generic[T], 
                 need_debug: bool):

    """
    Example
    
    - items: Dict
        - sentence: '특히 <영동고속도로:LC> <강릉:LC> 방향 <문막휴게소:LC>에서 <만종분기점:LC>까지 <5㎞:QT> 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다.'
        - tokens: ['특', '히', ' ', '영', '동', '고', '속', '도', '로', ' ', '강', '릉', ' ', '방', ...]
        - ner_tags: [12, 12, 12, 2, 3, 3, 3, 3, 3, 12, 2, 3, 12, 12, ...]
    """

    word_tagged = defaultdict(int)    # For stats

    sentence, tokens, ner_tags = list(items.values())
    assert len(tokens) == len(ner_tags), 'Strange'

    if any([unit for unit in unicode_units if unit in sentence]):
        return 

    xa_zero = sentence.find('\xa0')
    if xa_zero != -1:
        sentence = sentence.replace('\xa0', ' ')
        xa_zero = [i for i, x in enumerate(tokens) if x == '\xa0']
        for xaz in xa_zero:
            tokens[xaz] = ' '

    for LA in LABEL_ALIGN:
        cnt = len(re.findall(LA, sentence))
        word_tagged[LABEL_ALIGN[LA]] += cnt

    sentence_aligned = ''.join(tokens)
    sa_len = len(sentence_aligned)
    
    if sa_len != len(sentence_aligned.strip()):
        lstripped = sentence_aligned.lstrip()
        ner_tags = ner_tags[sa_len-len(lstripped):]
        
        rstripped = sentence_aligned.rstrip()
        ner_tags = ner_tags[:-(sa_len-len(rstripped))]

    sentence_aligned = sentence_aligned.strip()

    sentence_tokenized = tokenizer.tokenize(sentence_aligned)

    if len(sentence_aligned) != len(strip_remove(sentence_aligned)):    # Abandon 
        return 

    new_tags = list(map(lambda x: KLUE_LABEL[x], ner_tags))
    new_tags_vis = list(map(lambda x: LABEL_INV[x], new_tags))
    
    assert len(sentence_tokenized) == len(new_tags), 'Unexpected Error'

    ntv_copy = new_tags_vis[:]
    new_tags_vis = remove_hanzi_and_josa(sentence_aligned, new_tags_vis)
    if ntv_copy != new_tags_vis:    # <중국 후난(湖南)성 창샤(長沙)시 우자링(五家岭)가:LC>
        return

    out_dicts['c4_s'].append(sentence_aligned)
    out_dicts['c4_t'].append(new_tags_vis)
    out_dicts['c4_w'].append(word_tagged) 

    if need_debug:
        show_all([pd.DataFrame([sentence_tokenized, new_tags_vis]).T, sentence])