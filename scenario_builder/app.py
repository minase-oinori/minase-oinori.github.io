import streamlit as st
import pandas as pd
import random
import sys
import os
import json
import time
import datetime
import pathlib
import logging
from typing import Any

# Google Cloud & AI ライブラリ
from google.oauth2.service_account import Credentials
from google.cloud import storage
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import google.generativeai as genai
from google.generativeai.types import Part

# Google Sheets ライブラリ
import gspread
from gspread import utils as gspread_utils
import gspread_formatting

# 画像処理ライブラリ
import PIL.Image

# ----------------------------------------------------------------------
# Streamlit CloudのSecretsから認証情報を読み込む設定
# ----------------------------------------------------------------------
GCP_SERVICE_ACCOUNT = None
try:
    # Streamlit Cloudでの本番用
    GCP_SERVICE_ACCOUNT = st.secrets["gcp_service_account"]
except (FileNotFoundError, KeyError):
    st.error("Google Cloudの認証情報が設定されていません。StreamlitのSecretsにGCP_SERVICE_ACCOUNTを追加してください。")
    st.stop()

# 認証情報が設定されていない場合のエラー表示
if not GCP_SERVICE_ACCOUNT:
    st.error("Google Cloudの認証情報が空です。")
    st.stop()

# 文字列の認証情報を辞書に変換
try:
    credentials_info = json.loads(GCP_SERVICE_ACCOUNT)
    credentials = Credentials.from_service_account_info(credentials_info)
except json.JSONDecodeError as e:
    st.error(f"認証情報のJSON形式が正しくありません: {e}")
    st.stop()
except Exception as e:
    st.error(f"認証情報の処理中に予期せぬエラーが発生しました: {e}")
    st.stop()

# ----------------------------------------------------------------------
# これまでの backend.py の関数群
# ----------------------------------------------------------------------

def setup_and_auth(project_id, location='us-central1'):
    """各種認証とクライアントの初期化"""
    try:
        vertexai.init(project=project_id, location=location, credentials=credentials)
        gc = gspread.authorize(credentials)
        storage_client = storage.Client(project=project_id, credentials=credentials)
        
        # Google AI Python SDK を使う場合
        genai.configure(credentials=credentials)
        
        return gc, storage_client
    except Exception as e:
        st.error(f"Googleサービスへの接続中にエラーが発生しました: {e}")
        st.stop()


def convert_json_to_dict(json_data: str) -> dict[str, Any]:
    # マークダウンのコードブロック表記を削除
    if "```json" in json_data:
        json_data = json_data.replace("```json", "").replace("```", "")
    
    try:
        return json.loads(json_data)
    except json.JSONDecodeError as e:
        st.error(f"AIからの応答JSONの解析に失敗しました: {e}")
        st.write("AIの応答:")
        st.code(json_data)
        return None

@st.cache_data(ttl=3600)
def generate_and_extract_json(prompt: str, model_name: str) -> dict[str, Any]:
    retries = 3
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)

            json_response = convert_json_to_dict(response.text)
            if json_response:
                return json_response
            else:
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                else:
                    st.error("AIからの応答をJSONとして解析できませんでした。")
                    return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                st.error(f"Geminiへのリクエスト中にエラーが発生しました: {e}")
                return None
    return None


@st.cache_data(ttl=3600)
def generate_images(prompt, number_of_images, aspect_ratio, safety_filter_level, person_generation, seed, model_name, project_id, location):
    """Vertex AIのImagenモデルで画像を生成"""
    try:
        model = ImageGenerationModel.from_pretrained(model_name)
        images = model.generate_images(
            prompt=prompt,
            number_of_images=number_of_images,
            aspect_ratio=aspect_ratio,
            safety_filter_level=safety_filter_level,
            person_generation=person_generation,
            seed=seed,
            add_watermark=False
        )
        return images
    except Exception as e:
        st.error(f"画像生成中にエラーが発生しました: {e}")
        return []

def upload_blobs(storage_client, filenames: list[str], bucket_name: str, blob_prefixes: str):
    """GCSにファイルをアップロード"""
    bucket = storage_client.bucket(bucket_name)
    fname_to_gs_url = {}
    for filename in filenames:
        # 一時ファイルとして保存
        temp_path = f"/tmp/{os.path.basename(filename)}"
        filenames[filename].save(temp_path)

        blob_name = f"{blob_prefixes}/{os.path.basename(filename)}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(temp_path)
        os.remove(temp_path) # 一時ファイルを削除
        
        gs_url = f"[https://storage.googleapis.com/](https://storage.googleapis.com/){bucket_name}/{blob_name}"
        fname_to_gs_url[filename] = gs_url
    return fname_to_gs_url

# (その他のヘルパー関数もここに追加)
def get_curr_time(diff: int = 9) -> str:
    return (datetime.datetime.utcnow() + datetime.timedelta(hours=diff)).strftime('%Y%m%d%H%M')

def read_sheet(gc, spreadsheet_url: str, sheet_name: str) -> pd.DataFrame:
    try:
        spreadsheet = gc.open_by_url(spreadsheet_url)
        sheet = spreadsheet.worksheet(sheet_name)
        rows = sheet.get_all_values()
        if len(rows) > 1:
            df = pd.DataFrame(rows[1:], columns=rows[0])
            return df
        return pd.DataFrame()
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"スプレッドシートが見つかりません: {spreadsheet_url}")
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"シート '{sheet_name}' が見つかりません。")
    except Exception as e:
        st.error(f"シート読み込み中にエラー: {e}")
    return pd.DataFrame()

def export_dataframe(gc, url, name, df):
    try:
        out_ss = gc.open_by_url(url)
        worksheet = out_ss.add_worksheet(title=name, rows=len(df) + 1, cols=len(df.columns))
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())
        st.success(f"シート '{name}' をスプレッドシートに出力しました。")
    except Exception as e:
        st.error(f"スプレッドシートへのエクスポート中にエラー: {e}")

def load_dict_from_sheet(gc, url, sheet_name, key, value):
    df = read_sheet(gc, url, sheet_name)
    if not df.empty and key in df.columns and value in df.columns:
        return dict(zip(df[key], df[value]))
    return {}


# ----------------------------------------------------------------------
# Streamlit アプリケーション UI
# ----------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title('🎬 AI動画シナリオ＆画像生成アプリ')

# --- 1. 設定セクション ---
with st.sidebar:
    st.header('⚙️ 必須設定')
    project_id = st.text_input('Google Cloud Project ID', 'your-gcp-project-id')
    bucket_name = st.text_input('Google Cloud Storage Bucket', 'your-gcs-bucket-name')
    input_spreadsheet_url = st.text_input('入力元スプレッドシートURL', '[https://docs.google.com/spreadsheets/d/](https://docs.google.com/spreadsheets/d/)...')
    direction_output_spreadsheet_url = st.text_input('ディレクション出力先シートURL', '[https://docs.google.com/spreadsheets/d/](https://docs.google.com/spreadsheets/d/)...')
    output_spreadsheet_url = st.text_input('最終成果物シートURL', '[https://docs.google.com/spreadsheets/d/](https://docs.google.com/spreadsheets/d/)...')
    
    st.header('🛠️ 詳細設定')
    num_of_outputs = st.slider('ストーリーボードの出力数', 1, 5, 1)
    gemini_model_name = st.selectbox('Geminiモデル', ['gemini-1.5-flash-001', 'gemini-1.5-pro-001'])
    imagen_model_name = st.selectbox('Imagenモデル', ['imagen-3.0-generate-002'])

# --- セッション管理 ---
if 'gc' not in st.session_state:
    st.session_state.gc = None
    st.session_state.storage_client = None
    st.session_state.auth_done = False

# --- 認証ボタン ---
if st.sidebar.button("設定を適用してGoogleにログイン"):
    if not all([project_id, bucket_name, input_spreadsheet_url, direction_output_spreadsheet_url, output_spreadsheet_url]):
        st.sidebar.error("必須設定をすべて入力してください。")
    else:
        with st.spinner("Googleサービスに接続中..."):
            gc, storage_client = setup_and_auth(project_id)
            st.session_state.gc = gc
            st.session_state.storage_client = storage_client
            st.session_state.auth_done = True
        st.sidebar.success("ログイン成功！")

# --- アプリ本体 ---
if not st.session_state.auth_done:
    st.warning("サイドバーで設定を入力し、「設定を適用してGoogleにログイン」ボタンを押してください。")
else:
    # パラメータを辞書にまとめる
    params = {
        'PROJECT_ID': project_id, 'BUCKET_NAME': bucket_name, 'INPUT_SPREADSHEET_URL': input_spreadsheet_url,
        'DIRECTION_OUTPUT_SPREADSHEET_URL': direction_output_spreadsheet_url, 'OUTPUT_SPREADSHEET_URL': output_spreadsheet_url,
        'NUM_OF_OUTPUTS': num_of_outputs, 'GEMINI_MODEL_NAME': gemini_model_name, 'IMAGEN_MODEL_NAME': imagen_model_name,
        'SAFETY_FILTER_LEVEL': "block_some", 'PERSON_GENERATION': "allow_all", 'RANDOM_SEED': random.randint(1, 10000),
    }

    # --- 2. ディレクション作成セクション ---
    st.header('ステップ1：ディレクション作成')
    if st.button('① ディレクション作成を実行'):
        st.info("ディレクション作成を開始します... (この機能は現在実装中です)")
        # ここにディレクション作成のロジックを実装
        st.success("ディレクション作成が完了しました。（ロジックは実装されていません）")


    # --- 3. 画像生成セクション ---
    st.header('ステップ2：画像生成')
    edited_sheet_name = st.text_input('人間が編集したシート名を入力してください')
    if st.button('② 画像生成を実行'):
        if edited_sheet_name:
            st.info("画像生成を開始します... (この機能は現在実装中です)")
            # ここに画像生成のロジックを実装
            st.success("画像生成が完了しました。（ロジックは実装されていません）")
        else:
            st.error("編集したシート名を入力してください。")
