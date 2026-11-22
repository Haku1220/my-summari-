import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageOps
from google import genai

# ページ全体のレイアウトと基本設定
st.set_page_config(
    page_title="AI教材クリエイター | 新デザイン検証",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =================【🎨 カスタムCSSによる極上デザイン調整】=================
# ここで全体のフォント、カードの枠線、色のグラデーションなどを自由にカスタマイズしてください！
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Noto Sans JP', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }

    /* より洗練されたプレミアムヘッダーのデザイン */
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

    /* カードコンテナ（枠線とソフトシャドウ） */
    .custom-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.1);
        margin-bottom: 20px;
    }

    /* カスタムデザインボタン */
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

    /* ファイルアップローダー枠のデザイン */
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
st.markdown('<p class="sub-title">【デザイン確認用 ver 2.0】より美しいレイアウトと直感的なインタフェースへの挑戦</p>', unsafe_allow_html=True)

# APIキーの取得
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = None
    st.warning("⚠️ 管理者設定: Streamlit Cloudの Secrets に GEMINI_API_KEY を設定してください。")

# 複数画像アップローダーの配置
uploaded_files = st.file_uploader(
    "検証用の画像ファイルをアップロードしてください（最大10枚）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.error("⚠️ 一度にアップロードできる画像は最大10枚までです。選択し直してください。")
    else:
        st.success(f"📸 {len(uploaded_files)} 枚の画像が読み込まれました。")

        # プレビューコンテナ
        cols = st.columns(min(len(uploaded_files), 3))
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 3]:
                img_preview = Image.open(file)
                img_preview = ImageOps.exif_transpose(img_preview).convert("RGB")
                st.image(img_preview, caption=f"検証画像 {idx + 1}", use_container_width=True)

        st.write("")
        if st.button("✨ 統合レジュメの生成（テスト実行）"):
            if not api_key:
                st.error("APIキーが設定されていません。Colab環境でテストする場合は、コード内で直接渡すかSecretsを設定してください。")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                all_ocr_texts = []

                try:
                    for idx, file in enumerate(uploaded_files):
                        status_text.text(f"📖 OCR処理中 ({idx + 1}/{len(uploaded_files)} 枚目)...")

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

                    combined_text = "\n\n".join(all_ocr_texts) if all_ocr_texts else "(画像からの文字検出なし)"

                    status_text.text("🤖 Geminiによる新規デザインレジュメ生成中...")
                    progress_bar.progress(75)

                    client = genai.Client(api_key=api_key)
                    prompt = f"""
                    提供された複数のOCRテキストを元に、全体の流れが一目で理解できるように構造化した、高品質な「1つの統合レジュメ」を作成してください。
                    比較事項は必ずMarkdown表形式を使い、プロセスはMermaid形式でフロー図を書いてください。
                    \n\n
                    {combined_text}
                    """

                    response = client.models.generate_content(
                        model='gemini-3.8-flash',
                        contents=prompt,
                    )

                    progress_bar.progress(100)
                    status_text.empty()

                    st.success("🎉 新デザインでのレジュメ生成が正常に完了しました！")
                    st.subheader("📚 作成されたレジュメプレビュー")
                    st.markdown(response.text)

                except Exception as e:
                    status_text.empty()
                    st.error(f"エラーが発生しました: {e}")
