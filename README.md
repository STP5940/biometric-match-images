# วิธีการติดตั้ง

สร้าง venv สำหรับโปรเจคบน python:
``` sh
python -m venv biometricenv
```

เปิดเข้าใช้งาน venv ที่สร้างไว้:
``` sh
# เปิดใช้ (Windows)
biometricenv\Scripts\activate
```
``` sh
# หรือ (macOS/Linux)
source biometricenv/bin/activate
```

สั่งติดตั้ง lib ที่ต้องใช้ในโปรเจค:
``` sh
pip install -r requirements.txt
```

เช็ค lib ที่ติดตั้งไว้:
``` sh
pip list
```

# วิธีการใช้งานเพื่อทดสอบ

* ถ้ายังไม่มีรูปที่ Crop แล้วใน cropped_images ให้ใช้คำนี้:

สั่ง Crop ใบหน้าจาก input_images ไปเก็บไว้ที่ cropped_images
```
python cropfaces.py
```

จัดเก็บ Biometric จากภาพที่ Crop ลงฐานข้อมูลจำลอง (ยกเลิกคอมเม้นต์เพื่อใช้งาน)
```
# ตัวอย่างการประมวลผลทั้งโฟลเดอร์
print("📷 Processing batch image example:")
results = biometric_converter.process_batch_image("cropped_images", "embeddings.pkl", save_mode="overwrite")
```
และรันคำสั่งเพื่อบันทึกลงฐานข้อมูล
```
python biometric.py
```

ทดสอบตรวจสอบภาพใบหน้าตัวอย่าง จากฐานข้อมูล (ยกเลิกคอมเม้นต์เพื่อใช้งาน)
```
# ตัวอย่างการโหลดและเปรียบเทียบ
loaded_embeddings = biometric_converter.load_embeddings("embeddings.pkl")
if loaded_embeddings:
    probe_result = biometric_converter.process_face_image("testFace.png")
    ...
```
และรันคำสั่งเพื่อตรวจสอบความเข้ากันของใบหน้าจากในฐานข้อมูลจำลอง
```
python biometric.py
```

# Dependency Development

* pip: 24.0
* Python: 3.12.4
