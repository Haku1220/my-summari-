import os
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from google import genai
from google.colab import userdata

os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/tessdata/'

try:
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"指定された画像が見つかりません: {IMAGE_PATH}")

    # 画像の読み込みと前処理
    image = Image.open(IMAGE_PATH)
    image = ImageOps.exif_transpose(image).convert("RGB")
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # OCR実行
    print(f"📖 画像から文字を読み取っています...")
    ocr_text = pytesseract.image_to_string(processed, lang=OCR_LANG, config="--psm 6")

    # Gemini APIの実行
    print("🤖 Geminiがクイズ・まとめ付きレジュメを作成しています...")
    api_key = userdata.get('GEMINI_API_KEY')
    client = genai.Client(api_key=api_key)

    # レジュメ作成に特化した高度なシステムプロンプト (クイズ・ポイントまとめ付き)
    prompt = f"""
あなたは非常に優秀な家庭教師、および教材クリエイターです。
提供されたOCRテキスト（教科書やノートの読み取りデータ）を元に、学生が直感的に理解でき、学習定着度を高められる高品質な「レジュメ（要約学習プリント）」を作成してください。

【レジュメ作成ルール】
1. **タイトルと概要**:
   何について書かれた内容なのか、一目でわかるタイトルと短い導入を書いてください。

2. **視覚的な整理（表・グラフ）の積極的導入**:
   - テキスト内に数値の比較、対比、分類、タイムラインなどがある場合は、箇条書きで済ませず、必ずMarkdownの「表（テーブル）」形式に整理してください。
   - 物事の流れ（フロー）、因果関係、構成要素のつながりなどがある場合は、Markdown内でレンダリング可能な「Mermaid.js (graph TDなど)」のコードブロックを使って図解を作成してください。

3. **重要キーワード解説**:
   理解に不可欠な専門用語や重要語句をピックアップし、初心者向けに優しく解説してください。

4. **★最重要ポイントのまとめ（振り返りセクション）**:
   この講義・ページにおいて「これだけは絶対に暗記・理解して帰るべき」という超重要エッセンスを、3〜5項目の簡潔な箇条書きでまとめてください。

5. **★理解度チェッククイズ（Comprehension Quiz）**:
   学習した内容がしっかりと身についているかテストするための「クイズ」を3問程度作成してください。
   - 記述式、穴埋め式、または3択式のいずれかで作成してください。
   - クイズセクションのすぐ下、または折りたたみ（HTML <details> タグなど）を使って、わかりやすい「解答」と「ワンポイント解説」を必ずセットで記述してください。

【対象テキスト】
{ocr_text}
"""

    response = client.models.generate_content(
        model='gemini-3.8-flash',
        contents=prompt,
    )

    print("\n✨ === [作成されたクイズ付きレジュメ] ===")
    print(response.text)
    print("==============================")

except Exception as e:
    print(f"❌ エラーが発生しました: {e}")
