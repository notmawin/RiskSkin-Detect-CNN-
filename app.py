from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage, TextMessage
from ultralytics import YOLO
from PIL import Image
import io
import os
import gc
from dotenv import load_dotenv

# 1. โหลดค่า Config
load_dotenv()
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
channel_secret = os.getenv('LINE_CHANNEL_SECRET')

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

app = Flask(__name__)

# 2. โหลด Model (โหลดครั้งเดียวตอนเริ่ม)
print("Loading module YOLOv11...")
model = YOLO('skin-cancer.pt') 
print("Success!")

@app.route("/", methods=['GET'])
def index():
    return "Bot is Running Live!", 200

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    
    # แปลงรูปภาพ
    image_bytes = io.BytesIO(message_content.content)
    image = Image.open(image_bytes)

    # --- 3. วิเคราะห์ (ปรับแต่งเพื่อประหยัด RAM บน Render) ---
    results = model.predict(
        source=image, 
        imgsz=320,       # ลดขนาดภาพเพื่อประหยัด RAM
        conf=0.25, 
        save=False, 
        device='cpu'     # บังคับใช้ CPU เท่านั้น
    )
    
    result = results[0]
    
    if len(result.boxes) > 0:
        box = result.boxes[0]
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0]) * 100

        # เตรียมคำตอบ
        reply_text = f"📄 ผลการตรวจจับ: พบ {class_name}\n📊 ความมั่นใจ: {confidence:.2f}%"
        
        # เงื่อนไขแบ่งตามประเภท
        if "akiec" in class_name or class_id == 0:
             reply_text += "\n\n⚠️ คำแนะนำ: มีความเสี่ยงสูง ควรพบแพทย์\n\nActinic Keratoses (akiec) คือเซลล์ผิวหนังที่เริ่มผิดปกติจากแสงแดด ถือเป็นระยะก่อนเป็นมะเร็ง"
        elif "bcc" in class_name or class_id == 1:
             reply_text += "\n\n⚠️ คำแนะนำ: มีอันตราย ควรพบแพทย์\n\nBasal Cell Carcinoma (bcc) มะเร็งผิวหนังที่พบบ่อยที่สุด มักโตช้าแต่ต้องรักษาเพื่อไม่ให้ทำลายเนื้อเยื่อรอบข้าง"
        elif "mel" in class_name or class_id == 4:
             reply_text += "\n\n🚨 คำแนะนำ: อันตรายมาก! ควรรีบพบแพทย์\n\nMelanoma คือมะเร็งผิวหนังชนิดที่รุนแรงที่สุด มีโอกาสกระจายตัวสูง"
        elif "bkl" in class_name or class_id == 2:
             reply_text += "\n\n✅ คำแนะนำ: ปกติ\n\nBenign keratosis-like lesions คือกระเนื้อตามวัย ไม่ใช่มะเร็ง"
        elif "df" in class_name or class_id == 3:
             reply_text += "\n\n✅ คำแนะนำ: ปกติ\n\nDermatofibroma คือตุ่มเนื้อนูนแข็ง ไม่ใช่มะเร็ง"
        elif "nv" in class_name or class_id == 5:
             reply_text += "\n\n✅ คำแนะนำ: ปกติ\n\nMelanocytic nevi คือไฝปกติที่พบได้ทั่วไป"
        elif "vasc" in class_name or class_id == 6:
             reply_text += "\n\n✅ คำแนะนำ: ปกติ\n\nVascular lesions คือปานแดงหรือเส้นเลือดฝอยขยายตัว"

        reply_text += "\n\n⛔ คำเตือน: ผลนี้เป็นการคัดกรองเบื้องต้นโดย AI เท่านั้น ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้"

    else:
        reply_text = "❌ ไม่พบรอยโรคที่ชัดเจน\n\nลองถ่ายใหม่ให้ชัดเจน แสงสว่างเพียงพอ และให้จุดที่สงสัยอยู่กลางภาพครับ"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
    
    # เคลียร์ RAM ทันทีหลังใช้เสร็จ (ป้องกัน SIGKILL)
    del results
    gc.collect()

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_text = event.message.text.lower()
    if user_text == "faq":
        reply = "❓ [FAQ]\n1. แม่นยำไหม? : เป็นการคัดกรองเบื้องต้นด้วย YOLOv11\n2. ถ่ายยังไง? : แสงชัด ไฝอยู่กลางภาพ"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
    elif user_text == "about":
        reply = "🛡️ [DermaCheck AI]\nพัฒนาโดยนักเรียนชั้น ม.4 เพื่อคัดกรองมะเร็งผิวหนังเบื้องต้นด้วยชุดข้อมูล HAM10000"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

if __name__ == "__main__":
    # การตั้งค่า Port สำหรับ Render (สำคัญมาก!)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
