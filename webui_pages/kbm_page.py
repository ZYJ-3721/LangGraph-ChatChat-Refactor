import os
import time
import shutil
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from typing import Literal
from tqdm import tqdm

from webui_pages.utils import PLATFORMS_API_URL, PLATFORMS, get_embedding_models, get_embedding
from kbm.chunks_table import add_chunks_to_db, get_chunks_from_db, delete_chunks_from_db
from kbm.files_table import add_files_to_db, get_files_from_db, delete_files_from_db
from kbm.kbs_table import add_kb_to_db, get_kb_from_db, delete_kb_from_db
from kbm.base import KBS_ROOT, create_tables

from rag.utils import FILE_TYPES_LOADERS, load_document
from rag.utils import SPLITTERS, split_text
from rag.utils import VECTORSTORES, get_vectorstore

create_tables() # 预先创建数据库表

def build_gridOptions(
        df: pd.DataFrame,
        columns_config: dict[tuple[str, str], dict] = {},
        selection_mode: Literal["single", "multiple", "disabled"] = "multiple",
        use_checkbox: bool = False, header_checkbox: bool = False, autoHeight: bool = False):
    builder = GridOptionsBuilder.from_dataframe(df)
    builder.configure_default_column( # 中文
        filter=True, minWidth=80, maxWidth=200,
        wrapHeaderText=True, autoHeaderHeight=True)
    builder.configure_auto_height(autoHeight=autoHeight) 
    builder.configure_pagination(True, False, paginationPageSize=20) # 分页
    builder.configure_selection(selection_mode, use_checkbox, header_checkbox) # 选择
    builder.configure_column("No.", "序号", maxWidth=80, suppressMovable=True, cellStyle={'textAlign': 'center'})
    builder.configure_column("id", "标识", hide=True) # 隐藏id标识列
    for (col, header), kw in columns_config.items(): # 每列配置
        builder.configure_column(col, header, **kw)
    return builder.build() # 返回配置字典

def save_file(file, save_path, overwrite=False):
    try:
        file_path = os.path.join(save_path, file.name)
        if os.path.isfile(file_path) and not overwrite:
            return False, file_path
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        return True, file_path
    except Exception as e:
        return False, file.name

def multi_thread_run(func, params, desc=None, max_workers=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = []
        futures = [executor.submit(func, **kwargs) for kwargs in params]
        for future in tqdm(as_completed(futures), total=len(params), desc=desc):
            try:
                results.append(future.result())
            except Exception as e:
                print(f'【ERROR!】{e}')
        return results

def format_selected_kb_name(kb_name):
    if kb_name == "新建知识库":
        return kb_name
    kb = get_kb_from_db(kb_name) # 从数据库表中获取知识库信息
    return f'{kb_name} ({kb["kb_type"]} & {kb["embedding_model"]})'


def kbm_page():
    st.title("知 识 库 管 理")
    if "selected_kb_name" not in st.session_state:
        st.session_state["selected_kb_name"] = "新建知识库"
    
    kb_name_list = ["新建知识库"] + os.listdir(KBS_ROOT)
    kb_name_list.remove("info.db") # 移除info.db
    selected_kb_name = st.selectbox(
        format_func=format_selected_kb_name,
        label="请选择或新建知识库", options=kb_name_list,
        index=kb_name_list.index(st.session_state["selected_kb_name"]))
    
    if selected_kb_name == "新建知识库":
        with st.expander("知识库配置", expanded=True):
            cols = st.columns(2)
            kb_name = cols[0].text_input("知识库名称", placeholder="暂不支持中文")
            kb_type = cols[1].selectbox("向量库类型", options=VECTORSTORES.keys())
            kb_desc = st.text_input("知识库简介", placeholder="描述知识库的主题", value=f"关于{kb_name}的知识库")
            cols1 = st.columns(2) # 第一行两列
            cols2 = st.columns(2) # 第二行两列
            platform = cols1[0].selectbox("选择平台", options=PLATFORMS)
            api_url = cols2[0].text_input("API URL", placeholder=PLATFORMS_API_URL[platform])
            api_key = cols2[1].text_input("API KEY", placeholder="EMPTY")
            if platform in ["Xinference", "Ollama"]:
                embedding_model = cols1[1].selectbox("选择模型", options=get_embedding_models(platform, api_url, api_key))
            else:
                embedding_model = cols1[1].text_input("选择模型", placeholder="（必填）")
            if st.button("确认创建", use_container_width=True):
                if not kb_name.strip():
                    st.error(f"知识库名称不能为空！")
                elif kb_name in kb_name_list:
                    st.error(f"知识库名称“{kb_name}”已存在！")
                else:
                    try: # 添加知识库信息到数据库表中
                        add_kb_to_db(kb_name, kb_type, kb_desc, platform, api_url, api_key, embedding_model)
                    except Exception as e:
                        st.error(e)
                        st.stop()
                    kb_path = os.path.join(KBS_ROOT, kb_name)
                    files_path = os.path.join(kb_path, "files")
                    vectorstores_path = os.path.join(kb_path, "vectorstores")
                    os.makedirs(kb_path, exist_ok=True)
                    os.makedirs(files_path, exist_ok=True)
                    os.makedirs(vectorstores_path, exist_ok=True)
                    st.success("知识库创建成功！3秒后自动跳转...")
                    time.sleep(1.5)
                    st.session_state["selected_kb_name"] = kb_name
                    st.rerun()
    else:
        try: # 从数据库表中获取选择的知识库信息
            kb = get_kb_from_db(selected_kb_name)
        except Exception as e:
            st.error(e)
            st.stop()
        
        with st.expander("知识库配置", expanded=False):
            cols = st.columns(2)
            kb_name = cols[0].text_input("知识库名称", value=kb["kb_name"], disabled=True)
            kb_type = cols[1].selectbox("向量库类型", options=kb["kb_type"], disabled=True)
            kb_desc = st.text_input("知识库简介", placeholder="描述知识库的主题", value=kb["kb_desc"])
            cols1 = st.columns(2) # 第一行两列
            cols2 = st.columns(2) # 第二行两列
            platform = cols1[0].selectbox("选择平台", options=PLATFORMS, index=PLATFORMS.index(kb["platform"]))
            api_url = cols2[0].text_input("API URL", placeholder=PLATFORMS_API_URL[platform], value=kb["api_url"])
            api_key = cols2[1].text_input("API KEY", placeholder="EMPTY", value=kb["api_key"])
            if platform in ["Xinference", "Ollama"]:
                embedding_model = cols1[1].selectbox("选择模型", options=kb["embedding_model"], disabled=True)
            else:
                embedding_model = cols1[1].text_input("选择模型", value=kb["embedding_model"], disabled=True)
            if st.button("确认修改", use_container_width=True):
                if kb_desc == kb["kb_desc"] and platform == kb["platform"] \
                    and api_url == kb["api_url"] and api_key == kb["api_key"]:
                    st.toast("⚠️没有检测到任何修改！")
                else:
                    try: # 添加知识库信息到数据库表中
                        add_kb_to_db(kb_name, kb_type, kb_desc, platform, api_url, api_key, embedding_model)
                    except Exception as e:
                        st.error(e)
                        st.stop()
                    st.toast("✅知识库配置修改成功！")
        
        with st.expander("文件加载器", expanded=False):
            cols = st.columns(3)
            selected_loader = cols[0].selectbox("选择文件加载方法", options=[])
        
        with st.expander("文本分割器", expanded=False):
            cols = st.columns(3)
            selected_splitter = cols[0].selectbox("选择文本分割方法", options=SPLITTERS.keys())
            chunk_size = cols[1].number_input("单段文本最大长度:", 10, 10000, 500)
            chunk_overlap = cols[2].number_input("相邻文本重合长度:", 0, chunk_size, int(chunk_size*0.1))
        
        kb_path = os.path.join(KBS_ROOT, kb_name)
        files_path = os.path.join(kb_path, "files")
        vectorstores_path = os.path.join(kb_path, "vectorstores")

        files = st.file_uploader("请上传知识文件", accept_multiple_files=True, type=FILE_TYPES_LOADERS.keys())

        cols = st.columns(3) # 知识库管理的三个按钮
        if cols[2].button("删除知识库", use_container_width=True):
            try: # 从数据库表中删除选择的知识库信息
                delete_kb_from_db(kb_name)
            except Exception as e:
                st.error(e)
                st.stop()
            shutil.rmtree(kb_path, ignore_errors=True)
            st.toast("✅知识库删除成功！3秒后自动跳转...")
            time.sleep(1.5)
            st.session_state["selected_kb_name"] = "新建知识库"
            st.rerun()
        
        if cols[1].button("x x x x x", use_container_width=True):
            pass
        
        if cols[0].button("添加到知识库", use_container_width=True, disabled=(len(files)==0)):
            with st.spinner("正在添加到知识库...", show_time=True):
                saved_files = multi_thread_run(
                    save_file, desc="\033[32mFile Saving",
                    params=[{"file": file, "save_path": files_path} for file in files])
                st.toast("✅文件保存完成！")
                loaded_docs = multi_thread_run(
                    load_document, desc="\033[32mFile Loading",
                    params=[{"file_path": sf[1]} for sf in saved_files if sf[0]])
                st.toast("✅文件加载完成！")

                from functools import partial
                split_text2 = partial(
                    split_text, splitter_type=selected_splitter,
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                splited_texts = multi_thread_run(
                    split_text2, desc="\033[32mText Splitting",
                    params=[{"documents": doc} for doc in loaded_docs if doc])
                splited_texts = sum(splited_texts, []) # 合并每个文档的文本块列表
                st.toast("✅文本分割完成！")

                try: # 获取嵌入模型，获取向量数据库，将分割后的文本块向量化并存储到向量数据库
                    embedding = get_embedding(platform, embedding_model, api_url, api_key)
                    vectorstore = get_vectorstore(kb_name, kb_type, embedding, vectorstores_path)
                    with tqdm(total=len(splited_texts), desc="\033[32mVector Storing") as pbar:
                        vectorstore.add_documents(splited_texts)
                        pbar.update(len(splited_texts))
                    st.toast("✅向量存储完成！")
                except Exception as e:
                    st.error(e)
                    st.stop()
                
                # 添加文件信息到数据库表中

        
        st.divider() # 水平分割线

        selected_rows = [] 
        df_files = pd.DataFrame(get_files_from_db(kb["id"]))
        if not len(df_files):
            st.info(f"知识库 **{kb_name}** 中暂无文件")
        else:
            st.info(f"知识库 **{kb_name}** 中的文件信息如下：")
            df_files.insert(0, 'No.', range(1, len(df_files) + 1))
            files_gridOptions = build_gridOptions(
                df_files, columns_config={
                    ("file_name", "文件名称"): {},
                    ("file_type", "文件类型"): {},
                    ("file_size", "文件大小"): {},
                    ("file_dir", "文件目录"): {},
                    ("kb_id", "所属知识库"): {"hide": True},
                    ("loader_name", "文件加载器"): {},
                    ("splitter_name", "文本分割器"): {},
                    ("chunk_count", "文本块数量"): {},
                    ("create_time", "创建时间"): {},
                    ("update_time", "更新时间"): {},
                })
            files_aggrid = AgGrid(df_files, gridOptions=files_gridOptions, height=300, theme="alpine")
            if files_aggrid.selected_rows is not None:
                selected_rows = files_aggrid.selected_rows
            
            cols = st.columns(3) # 知识库管理的三个按钮
            if cols[2].button("x x x x x 1", use_container_width=True):
                pass

            if cols[1].button("x x x x x 2", use_container_width=True):
                pass
            
            if cols[0].button("x x x x x 3", use_container_width=True):
                pass
            
        
        st.divider() # 水平分割线

        if not len(selected_rows):
            st.info(f"请选择需要查看的文件")
        else:
            selected_row_first = selected_rows.iloc[0].to_dict()
            df_chunks = pd.DataFrame(get_chunks_from_db(
                selected_row_first["kb_id"], selected_row_first["id"]))
            if not len(df_chunks):
                st.info(f"向量库中未检测到该文件的文本块")
            else:
                st.info(f"该文件的文本块信息如下：")
                df_chunks.insert(0, 'No.', range(1, len(df_chunks) + 1))
                chunks_gridOptions = build_gridOptions(
                    df_chunks, columns_config={
                        ("create_time", "创建时间"): {},
                        ("update_time", "更新时间"): {},
                    })
                chunks_aggrid = AgGrid(df_chunks, gridOptions=chunks_gridOptions, height=500, theme="alpine")

                cols = st.columns(3) # 知识库管理的三个按钮
                if cols[2].button("x x x x x 4", use_container_width=True):
                    pass

                if cols[1].button("x x x x x 5", use_container_width=True):
                    pass
                
                if cols[0].button("x x x x x 6", use_container_width=True):
                    pass
        
        st.divider() # 水平分割线