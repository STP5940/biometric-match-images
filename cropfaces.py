# cropfaces.py

import os
from PIL import Image
from facenet_pytorch import MTCNN

class FaceCropper:
    def __init__(self, margin: int = 20, image_size: int = 160):
        """
        เริ่มต้น FaceCropper ด้วยพารามิเตอร์ที่กำหนด
        
        Args:
            margin (int): ขอบเขตเพิ่มเติมรอบใบหน้า (พิกเซล)
            image_size (int): ขนาดภาพผลลัพธ์ (สี่เหลี่ยมจัตุรัส)
        """
        self.margin = margin
        self.image_size = image_size
        self.mtcnn = MTCNN(keep_all=False)
    
    def crop_single_face(self, image_path: str, output_path: str = None) -> str:
        """
        ครอบรูปใบหน้าจากภาพเดียว
        
        Args:
            image_path (str): เส้นทางไฟล์ภาพต้นทาง
            output_path (str): เส้นทางไฟล์ภาพผลลัพธ์ (ถ้าไม่กำหนดจะสร้างอัตโนมัติ)
        
        Returns:
            str: เส้นทางไฟล์ผลลัพธ์ หรือ None หากไม่พบใบหน้า
        """
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            # ตรวจจับใบหน้า
            boxes, _ = self.mtcnn.detect(image)

            if boxes is not None:
                box = boxes[0]  # ใช้ใบหน้าแรกที่พบ
                x1, y1, x2, y2 = [int(b) for b in box]

                # เพิ่ม margin และกันไม่ให้ออกนอกขอบภาพ
                x1 = max(x1 - self.margin, 0)
                y1 = max(y1 - self.margin, 0)
                x2 = min(x2 + self.margin, width)
                y2 = min(y2 + self.margin, height)

                # ครอบใบหน้า
                face_crop = image.crop((x1, y1, x2, y2))

                # ปรับขนาดภาพใบหน้า
                face_crop = face_crop.resize((self.image_size, self.image_size), Image.LANCZOS)

                # สร้างชื่อไฟล์ผลลัพธ์ถ้าไม่กำหนด
                if output_path is None:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_dir = os.path.dirname(image_path)
                    output_path = os.path.join(output_dir, f"{base_name}_cropped.png")

                # บันทึกภาพ
                face_crop.save(output_path)
                print(f"✅ บันทึกใบหน้า: {output_path}")
                return output_path
            else:
                print(f"❌ ไม่พบใบหน้าใน: {os.path.basename(image_path)}")
                return None

        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาดกับ {os.path.basename(image_path)}: {e}")
            return None
    
    def crop_faces_from_folder(self, input_folder: str, output_folder: str = None):
        """
        ครอบรูปใบหน้าจากทุกภาพในโฟลเดอร์
        
        Args:
            input_folder (str): โฟลเดอร์ต้นทาง
            output_folder (str): โฟลเดอร์ปลายทาง (ถ้าไม่กำหนดจะใช้โฟลเดอร์ต้นทาง)
        """
        # กำหนดโฟลเดอร์ปลายทาง
        if output_folder is None:
            output_folder = os.path.join(input_folder, "cropped_images")
        
        # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
        os.makedirs(output_folder, exist_ok=True)

        success_count = 0
        total_count = 0

        # วนลูปไฟล์ในโฟลเดอร์ต้นทาง
        for filename in os.listdir(input_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                total_count += 1
                input_path = os.path.join(input_folder, filename)
                output_filename = os.path.splitext(filename)[0] + ".png"
                output_path = os.path.join(output_folder, output_filename)

                result = self.crop_single_face(input_path, output_path)
                if result:
                    success_count += 1

        print(f"\n📊 สรุปผล: ประมวลผล {success_count}/{total_count} ภาพสำเร็จ")

    def set_parameters(self, margin: int = None, image_size: int = None):
        """
        เปลี่ยนพารามิเตอร์การทำงาน
        
        Args:
            margin (int): ขอบเขตใหม่
            image_size (int): ขนาดภาพใหม่
        """
        if margin is not None:
            self.margin = margin
        if image_size is not None:
            self.image_size = image_size
        print(f"⚙️ ตั้งค่าพารามิเตอร์: margin={self.margin}, image_size={self.image_size}")


# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # สร้าง instance ของ FaceCropper
    cropper = FaceCropper(margin=15, image_size=160)
    
    # วิธีที่ 1: ครอบรูปจากโฟลเดอร์
    cropper.crop_faces_from_folder("input_images", "cropped_images")
    
    # วิธีที่ 2: ครอบรูปเดี่ยว
    # cropper.crop_single_face("path/to/your/image.jpg", "output/cropped_face.png")
    
    # วิธีที่ 3: เปลี่ยนพารามิเตอร์
    # cropper.set_parameters(margin=20, image_size=200)