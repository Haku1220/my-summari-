import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageOps
from google import genai

# ページ全体のレイアウトと基本設定
st.set_page_config(
    page_title="AI教材クリエイター | スマートレジュメ作成",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =================【🎨 カスタムCSSによるデザイン調整】=================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Noto Sans JP', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        text-align: center;
    }
    .sub-title {
        color: #4B5563; 
        font-size: 1.05rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366F1, #3B82F6) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3) !important;
        background: linear-gradient(135deg, #4F46E5, #2563EB) !important;
    }
    
    section[data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px dashed #6366F1;
        border-radius: 12px;
        padding: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)
# =====================================================================

# タイトルエリアの描画
st.markdown('<h1 class="main-title">📖 AI Smart Resume Creator</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">教科書やノートを撮影・アップロードすると、Geminiが要点・図表・まとめ・理解度確認クイズを一括生成します</p>', unsafe_allow_html=True)

# APIキーの取得
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = None
    st.warning("⚠️ 管理者設定: Streamlit Cloudの Secrets に GEMINI_API_KEY を設定してください。")

# 最大10枚までの画像アップローダー
uploaded_files = st.file_uploader(
    "教科書やノートの画像を選択してください（最大10枚）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.error("⚠️ 一度にアップロードできる画像は最大10枚までです。選択し直してください。")
    else:
        st.success(f"📸 {len(uploaded_files)} 枚の画像が正常に読み込まれました。")
        
        # 3列構成で画像プレビューを表示
        cols = st.columns(min(len(uploaded_files), 3))
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 3]:
                img_preview = Image.open(file)
                img_preview = ImageOps.exif_transpose(img_preview).convert("RGB")
                st.image(img_preview, caption=f"ページ {idx + 1}", use_container_width=True)

        st.write("")
        if st.button("✨ クイズ・振り返り付きレジュメを生成する"):
            if not api_key:
                st.error("APIキーが設定されていません。アプリ設定のSecretsに登録してください。")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_ocr_texts = []
                
                try:
                    # 1. 各ページに対してOCR処理
                    for idx, file in enumerate(uploaded_files):
                        status_text.text(f"📖 {idx + 1}/{len(uploaded_files)} 枚目の画像をテキストに変換中...")
                        
                        image = Image.open(file)
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        
                        img = np.array(image)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                        
                        ocr_text = pytesseract.image_to_string(processed, lang="jpn", config="--psm 6")
                        if ocr_text.strip():
                            all_ocr_texts.append(f"--- [ページ {idx + 1} の内容] ---\n{ocr_text}")
                        
                        progress_bar.progress(int((idx + 1) / len(uploaded_files) * 50))
                    
                    combined_text = "\n\n".join(all_ocr_texts) if all_ocr_texts else "(画像からテキストを読み取れませんでした)"
                    
                    # 2. Geminiによる高度な要約＆クイズ生成プロンプト
                    status_text.text("🤖 クイズとまとめを構成中...（およそ10秒お待ちください）")
                    progress_bar.progress(75)
                    
                    client = genai.Client(api_key=api_key)
                    prompt = f"""
あなたは非常に優秀な家庭教師、および教材クリエイターです。
提供された複数のOCRテキスト（教科書やノートの複数ページ分の読み取りデータ）を元に、学生が直感的に理解でき、学習定着度を高められる高品質な「1つの統合レジュメ（要約学習プリント）」を作成してください。

【レジュメ作成・構成ルール】
1. **タイトルと導入**:
   何について書かれた内容なのか、全体像が一目でわかる大タイトルと、概要の導入を書いてください。

2. **視覚的な整理（表・グラフ）の積極的導入**:
   - データの比較、対比、分類、タイムラインなどがある場合は、箇条書きで済ませず、必ずMarkdownの「表（テーブル）」形式に整理してください。
   - プロセス、順序（ステップ）、因果関係、システムの関係性がある場合は、Markdown内でレンダリング可能な「Mermaid.js (graph TDなど)」コードブロックを使ってフロー図を書いてください。

3. **重要キーワード解説**:
   理解に不可欠な重要語句・専門用語をピックアップし、初心者向けに優しく解説してください。

4. **★最重要ポイントのまとめ（振り返りセクション）**:
   「これだけは絶対に覚えて帰るべき」という最重要エッセンスを、3〜5項目で簡潔な箇条書きでまとめてください。

5. **★理解度チェッククイズ（Comprehension Quiz）**:
   学んだ内容が定着したかを確認するためのクイズを「3問」作成してください。
   - 記述式、穴埋め式、3択式のいずれかで構成してください。
   - クイズ設問のすぐ下に、HTMLの折りたたみタグである「<details><summary>👉 正解と解説を表示</summary>...解答解説...</details>」を使用して、タップするまで正解が隠れるスマートな解説欄を設置してください。

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
