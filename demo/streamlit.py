import re
import hydra
import json
import requests
import streamlit as st
from typing import Dict
from annotated_text import annotated_text
from hydra.core.global_hydra import GlobalHydra 


NTH = 'NTH'

GlobalHydra.instance().clear()  # streamlit 과의 호환 

@hydra.main(version_base=None, config_path='../', config_name='config')
def streamlit_activate(cfg: Dict):

    config = cfg.stage4_cfg
    ml_server_exist = config.ml_server_exist
    host, port = list(config.app_server.values())

    url = 'ml_server' if ml_server_exist else 'predict/job_keyword'

    st.set_page_config(
        page_title = "Named Entity Recognition Demo", 
        page_icon = '❄️', 
        layout = 'centered', 
        initial_sidebar_state='expanded'
    )

    input_container = st.container()
    output_container = st.container()

    col1, col2 = st.columns(2, gap='small')
    with col1:
        keyword_button = st.button('키워드 검출')
    with col2:
        refresh_button = st.button('새로고침')    

    if refresh_button:
        st.experimental_rerun()      

    with input_container.expander('원본', expanded=True):
        user_input = st.text_area(label = '키워드를 추출할 문장 및 문서', 
                                  value = '백엔드 개발자 A싸는 10년에 걸쳐 삼성전자, 네이버, 카카오라는 회사에 근무했다.')
        st.write(f'입력: {user_input}')

    user_input = {
        'sentence': user_input
    }

    response = requests.post(url = f'http://{host}:{str(port)}/{url}', 
                            data = json.dumps(user_input))

    answers = response.json().get("response", [])
    
    if keyword_button:
        with output_container.expander('결과', expanded=True):

            if len(answers) == 0:
                answers = '추출할 개체명이 존재하지 않습니다.'
                st.write(answers)

            else:
                annotated = []

                idx_list = []
                for key, value in answers.items():
                    idcs = [(m.start(0), m.end(0), value) for m in re.finditer(key, user_input['sentence'])]
                    idx_list.extend(idcs)
                idx_list = sorted(idx_list, key=lambda x: (x[0], x[1]))

                if idx_list[0][0] != 0:
                    idx_list = [(0, 0, NTH)] + idx_list
                if idx_list[-1][-1] != len(user_input['sentence']):
                    idx_list.append((len(user_input['sentence']), len(user_input['sentence']), NTH))
                
                for front, back in zip(idx_list, idx_list[1:]):

                    if front[-1] == NTH:
                        annotated.append(user_input['sentence'][:back[0]])
                    else:
                        annotated.append((user_input['sentence'][front[0]:front[1]], front[2]))
                        annotated.append(user_input['sentence'][front[1]:back[0]])

                annotated_text(annotated)


# NOTE: Command on the root directory: streamlit run demo/streamlit.py
if __name__ == '__main__':
    streamlit_activate()    
