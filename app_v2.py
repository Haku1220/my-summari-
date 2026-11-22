import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageOps
from google import genai
from pypdf import PdfReader

# ページ全体のレイアウトと基本設定
st.set_page_config(
    page_title="Resumake",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =================【🎨 カスタムCSSによるシック・暖色系デザイン】=================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Noto Sans JP', sans-serif;
        background-color: #F8F5F2; /* 落ち着いたアイボリー・トープ */
        color: #4A3E3D; /* 深みのあるアッシュブラウン */
    }

    /* メインタイトル：シックなコッパーとテラコッタのグラデーション */
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #8C3838, #A75D5D, #C78B75);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        text-align: center;
    }
    .sub-title {
        color: #6D5D5C;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 2.5rem;
        letter-spacing: 0.05em;
    }

    /* シックなマットボタン（テラコッタブラウン） */
    .stButton > button {
        background: #9E4747 !important;
        color: #FDFBF7 !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important; /* 少し角を残してシックに */
        box-shadow: 0 4px 10px rgba(158, 71, 71, 0.15) !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 15px rgba(158, 71, 71, 0.25) !important;
        background: #803333 !important; /* より深い色へ */
    }

    /* アップローダー：極細のブロンズ枠とクリーンな白背景 */
    section[data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 1px solid #D3C1B5;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    </style>
""", unsafe_allow_html=True)
# =====================================================================

# タイトルエリアの描画
st.markdown('<h1 class="main-title">📖  Resumake Creator</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">日々の学びをスマートにドキュメント化</p>', unsafe_allow_html=True)

# APIキーの取得
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = None
    st.warning("⚠️ 管理者設定: Streamlit Cloudの Secrets に GEMINI_API_KEY を設定してください。")

# 画像とPDFを受け入れ
uploaded_files = st.file_uploader(
    "教科書、ノート、またはレジュメの画像・PDFを選択してください（最大10ファイル）",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.error("⚠️ 一度にアップロードできるファイルは最大10個までです。選択し直してください。")
    else:
        st.success(f"📁 {len(uploaded_files)} 個のファイルが正常に読み込まれました。")

        # ファイルプレビュー・一覧表示
        st.write("**📂 アップロード済みファイル一覧:**")
        for idx, file in enumerate(uploaded_files):
            file_type = file.name.split(".")[-1].lower()
            if file_type == "pdf":
                st.write(f"📄 ページファイル {idx + 1}: **{file.name}** (PDF文書 - 自動テキスト抽出)")
            else:
                st.write(f"📸 ページファイル {idx + 1}: **{file.name}** (画像データ - 自動OCR)")

        st.write("")
        if st.button("✨ クイズ・振り返り付きレジュメを生成する"):
            if not api_key:
                st.error("APIキーが設定されていません。アプリ設定のSecretsに登録してください。")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                all_extracted_texts = []

                try: 
                    # ファイルを1つずつループして処理
                    for idx, file in enumerate(uploaded_files):
                        file_name = file.name
                        file_type = file_name.split(".")[-1].lower()

                        if file_type == "pdf":
                            status_text.text(f"📄 {idx + 1}/{len(uploaded_files)} 個目のPDF「{file_name}」からテキストを解析中...")
                            reader = PdfReader(file)
                            pdf_text = ""
                            for page_num, page in enumerate(reader.pages):
                                extracted_text = page.extract_text()
                                if extracted_text:
                                    pdf_text += f"\n[PDFページ {page_num + 1}]\n{extracted_text}"
                            
                            if pdf_text.strip():
                                all_extracted_texts.append(f"--- [ファイル {idx + 1} ({file_name}) の内容] ---\n{pdf_text}")
                            else:
                                all_extracted_texts.append(f"--- [ファイル {idx + 1} ({file_name}) の内容] ---\n(PDFから直接テキストを抽出できませんでした)")
                        
                        else:
                            status_text.text(f"📸 {idx + 1}/{len(uploaded_files)} 個目の画像「{file_name}」からテキストをOCR変換中...")
                            image = Image.open(file)
                            image = ImageOps.exif_transpose(image).convert("RGB")

                            img = np.array(image)
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                            ocr_text = pytesseract.image_to_string(processed, lang="jpn", config="--psm 6")
                            if ocr_text.strip():
                                all_extracted_texts.append(f"--- [ファイル {idx + 1} ({file_name}) の内容] ---\n{ocr_text}")

                        progress_bar.progress(int((idx + 1) / len(uploaded_files) * 50))

                    combined_text = "\n\n".join(all_extracted_texts) if all_extracted_texts else "(テキストを読み取れませんでした)"

                    # Geminiへの指示
                    status_text.text("🤖 クイズとまとめを構成中...（およそ10秒お待ちください）")
                    progress_bar.progress(75)

                    client = genai.Client(api_key=api_key)
                    prompt = f"""
あなたは非常に優秀な家庭教師、および教材クリエイターです。
提供された複数のテキスト（教科書やノート、PDF文書の読み取りデータ）を元に、学生が直感的に理解でき、学習定着度を高められる高品質な「1つの統合レジュメ（要約学習プリント）」を作成してください。

【レジュメ作成・構成ルール】
1. **タイトルと導入**:
   全体像が一目でわかる大タイトルと、概要の導入を書いてください。

2. **視覚的な整理（表・グラフ）の積極的導入**:
   - データの比較、分類などがある場合は、必ずMarkdownの「表（テーブル）」形式に整理してください。
   - プロセスやシステムの関係性がある場合は、「Mermaid.js (graph TDなど)」コードブロックを使ってフロー図を書いてください。

3. **重要キーワード解説**:
   理解に不可欠な語句をピックアップし、優しく解説してください。

4. **最重要ポイントのまとめ（振り返りセクション）**:
   超重要エッセンスを3〜5項目の簡潔な箇条書きでまとめてください。

5. **理解度チェッククイズ（Comprehension Quiz）**:
   学んだ内容を確認するクイズを「3問」作成してください。
   - 設問のすぐ下に「<details><summary>👉 正解と解説を表示</summary>...解答解説...</details>」を設置して正解を隠してください。

【対象テキスト】
{combined_text}
"""
                    response = client.models.generate_content(
                        model='gemini-3.8-flash',
                        contents=prompt,
                    )

                    progress_bar.progress(100)
                    status_text.empty()

                    st.success("🎉 クイズ・まとめ付きレジュメの作成が完了しました！")
                    st.subheader("📚 作成された学習レジュメ")
                    st.markdown(response.text, unsafe_allow_html=True)

                except Exception as e:
                    status_text.empty()
                    st.error(f"エラーが発生しました: {e}")
