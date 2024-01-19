
# 장소, 일시, 사람, 회사명
NE_TARGET = ['PER', 'LOC', 'DAT', 'TIM', 'ORG', 'NUM']

LABEL_ALIGN = {
    'PS': 'PER', # 인물
    'LC': 'LOC', # 장소
    'DT': 'DAT', # 날짜 (일시)
    'TI': 'TIM', # 시간 (일시)
    'OG': 'ORG',  # 기관 (회사명)
    'QT': 'NUM'
}

O_TAG = 'O'

LABEL = {
    'O': 1,
    'PER_I': 2, 
    'PER_B': 3,
    'LOC_I': 4,
    'LOC_B': 5,
    'DAT_I': 6, 
    'DAT_B': 7, 
    'TIM_I': 8,
    'TIM_B': 9, 
    'ORG_I': 10,
    'ORG_B': 11, 
    'NUM_I': 12, 
    'NUM_B': 13
}

LABEL_INV = {v: k for k, v in LABEL.items()}

KLUE_LABEL = {
    0: 7,   # B-DT -> DAT_B
    1: 6,   # I-DT -> DAT_I 
    2: 5,   # B-LC -> LOC_B 
    3: 4,   # I-LC -> LOC_I 
    4: 11,   # B-OG -> ORG_B
    5: 10,   # I-OG -> ORG_I
    6: 3,   # B-PS -> PER_B
    7: 2,   # I-PS -> PER_I
    8: 13,   # B-QT -> NUM_B
    9: 12,   # I-QT -> NUM_I
    10: 9,  # B-TI -> TIM_B
    11: 8,  # I-TI -> TIM-I
    12: 1   # O -> O
}

JONGSUNG_ORD = {
    'ㄴ': 4, 
    'ㄹ': 8
}
