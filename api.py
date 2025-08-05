from flask import Flask, request, jsonify
import requests
import ast
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/extract_event_info', methods=['POST'])
def extract_event_info():
    # Kiểm tra xem có file ảnh không
    if 'file' not in request.files:
        return jsonify({"error": "Missing image file"}), 400

    img_file = request.files['file']

    # Gửi ảnh đến API OCR (Dorify)
    ocr_access_token = "e6527ee62ccb177d35d7792de7e8ec815369da74ff000d68afd8308d60944838"
    ocr_url = "https://dorify.net/api/predictions/e22596ee-b79a-4206-a90d-bfbe9be092e9"
    ocr_headers = {"Authorization": f"Bearer {ocr_access_token}"}

    ocr_response = requests.post(ocr_url, headers=ocr_headers, files={"file": img_file})
    try:
        json_obj = ast.literal_eval(ocr_response.text)
        json_list = json_obj['data']
    except:
        return jsonify({"error": "OCR failed", "raw_response": ocr_response.text}), 500

    full_tokens = ' '.join([x['text'] for x in json_list])

    # Prompt cho LLM
    prompt_opening = (
        "Tôi đang cần trích xuất những thông tin sau: "
        "Tên sự kiện, Miêu tả sự kiện, Số nhà, Tên đường, Khu vực, Xã/phường, "
        "Quận/huyện, Tỉnh thành, Ngày bắt đầu, Ngày kết thúc, Số lượng. "
        "Trình bày gọn gàng như sau (không in đậm, không thêm ký tự đặc biệt, không chú thích, có thể sửa chính tả hoặc thêm từ cho trọn vẹn ý nghĩa): "
        "Tên sự kiện: ... Miêu tả sự kiện: ... Số nhà: ... Tên đường: ... Khu vực: ... "
        "Xã/phường: ... Quận/huyện: ... Tỉnh thành: ... Ngày bắt đầu: ... Ngày kết thúc: ... Số lượng: ... "
        "Đây là đoạn văn bao gồm các token lộn xộn từ một phiếu sự kiện, hãy sắp xếp và trích xuất các thông tin đã cho: "
    )
    prompt = prompt_opening + full_tokens

    # Gọi LLM
    llm_url = 'https://api.groq.com/openai/v1/chat/completions'
    llm_model = 'meta-llama/llama-4-scout-17b-16e-instruct'
    llm1 = 'gsk_kG9LhNvp9XJliInAA8CD'
    llm2 = 'WGdyb3FYcVYp7UnH26BVq5IcX9QkmxHZ'
    llm_access_token = llm1 + llm2

    llm_api_headers = {
        "Authorization": f"Bearer {llm_access_token}",
        "Content-Type": "application/json"
    }

    llm_api_body = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}]
    }

    llm_response = requests.post(llm_url, headers=llm_api_headers, data=json.dumps(llm_api_body))

    if llm_response.status_code != 200:
        return jsonify({"error": "LLM API failed", "response": llm_response.text}), 500

    try:
        llm_json_text = llm_response.json()['choices'][0]['message']['content']
    except:
        return jsonify({"error": "Could not parse LLM response", "raw": llm_response.text}), 500

    # Parse về dict
    data_dict = {
        "Tên sự kiện": "",
        "Miêu tả sự kiện": "",
        "Số nhà": "",
        "Tên đường": '',
        'Khu vực': '',
        'Xã/phường': '',
        'Quận/huyện': '',
        'Tỉnh thành': '',
        'Ngày bắt đầu': '',
        'Ngày kết thúc': '',
        'Số lượng': '',
    }

    for line in llm_json_text.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key in data_dict:
                data_dict[key] = value

    # Xử lý tỉnh thành, quận huyện, xã phường
    import pandas as pd
    from unidecode import unidecode
    from rapidfuzz.distance import JaroWinkler

    provinces = pd.read_csv("provinces.csv")
    districts = pd.read_csv("districts.csv")
    wards = pd.read_csv("wards.csv")

    province_input = data_dict['Tỉnh thành']
    district_input = data_dict['Quận/huyện']
    ward_input = data_dict['Xã/phường']

    def normalize(text):
        text = text.lower()
        for word in ["tỉnh", "thành", "phố", "quận", "huyện", "xã", "phường"]:
            text = text.replace(word, "")
        return unidecode(text).replace(" ", "").strip()

    def find_best_match(name_norm, df, col):
        if df.empty:
            return 0
        scores = df[col].apply(lambda x: JaroWinkler.similarity(name_norm, x.lower()))
        if scores.max() == 0:
            return 0
        best_idx = scores.idxmax()
        return df.loc[best_idx, 'MaSo']

    province_norm = normalize(province_input)
    province_code = find_best_match(province_norm, provinces, "EnglishName")
    print("Mã tỉnh:", province_code)

    filtered_districts = districts[districts["MaTinhThanh"] == province_code]
    district_norm = normalize(district_input)
    district_code = find_best_match(district_norm, filtered_districts, "EnglishName")
    print("Mã quận/huyện:", district_code)

    filtered_wards = wards[wards["MaQuanHuyen"] == district_code]
    ward_norm = normalize(ward_input)
    ward_code = find_best_match(ward_norm, filtered_wards, "EnglishName")
    print("Mã phường/xã:", ward_code)

    data_dict['MaTinhThanh'] = int(province_code)
    data_dict['MaQuanHuyen'] = int(district_code)
    data_dict['MaPhuongXa'] = int(ward_code)


    return jsonify(data_dict)


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
