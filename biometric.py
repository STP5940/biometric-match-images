# biomatric_converter.py

import io
import os
import time
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
import pickle
from typing import Dict, List, Optional, Tuple


class BiometricConverter:
    def __init__(self, model_type: str = "vggface2", device: str = None):
        """
        เริ่มต้น Biometric Converter ด้วย FaceNet model

        Args:
            model_type (str): ประเภทโมเดล ('vggface2' หรือ 'casia-webface')
            device (str): device สำหรับ inference ('cuda' หรือ 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # โหลด FaceNet model
        self.model = InceptionResnetV1(pretrained=model_type).eval().to(self.device)

        print(f"✅ BiometricConverter initialized with {model_type} on {self.device}")

    def image_to_tensor(self, image_path: str) -> Optional[torch.Tensor]:
        """
        แปลงภาพเป็น tensor ที่เหมาะสมสำหรับ FaceNet

        Args:
            image_path (str): เส้นทางไฟล์ภาพ

        Returns:
            Optional[torch.Tensor]: Tensor ของภาพ หรือ None หากเกิดข้อผิดพลาด
        """
        try:
            # โหลดและแปลงภาพ
            image = Image.open(image_path).convert("RGB")

            # แปลงเป็น tensor และ normalize
            image_tensor = torch.tensor(np.array(image)).float()

            # เปลี่ยนรูปแบบจาก HWC to CHW
            image_tensor = image_tensor.permute(2, 0, 1)

            # Normalize (ค่าเฉลี่ยและ std จาก FaceNet)
            image_tensor = (image_tensor - 127.5) / 128.0

            # เพิ่ม batch dimension
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            return image_tensor

        except Exception as e:
            print(f"❌ Error converting image to tensor: {e}")
            return None
        
    def image_to_tensor_from_memory(self, image: Image.Image) -> Optional[torch.Tensor]:
        """
        แปลง PIL Image เป็น tensor ที่เหมาะสมสำหรับ FaceNet
        
        Args:
            image (Image.Image): PIL Image object
            
        Returns:
            Optional[torch.Tensor]: Tensor ของภาพ หรือ None หากเกิดข้อผิดพลาด
        """
        try:
            # แปลงเป็น tensor และ normalize
            image_tensor = torch.tensor(np.array(image)).float()

            # เปลี่ยนรูปแบบจาก HWC to CHW
            image_tensor = image_tensor.permute(2, 0, 1)

            # Normalize (ค่าเฉลี่ยและ std จาก FaceNet)
            image_tensor = (image_tensor - 127.5) / 128.0

            # เพิ่ม batch dimension
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            return image_tensor

        except Exception as e:
            print(f"❌ Error converting image to tensor from memory: {e}")
            return None

    def extract_embedding(self, image_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        สกัด facial embedding จากภาพ

        Args:
            image_tensor (torch.Tensor): Tensor ของภาพ

        Returns:
            Optional[np.ndarray]: Facial embedding (512-dim) หรือ None หากเกิดข้อผิดพลาด
        """
        try:
            with torch.no_grad():
                embedding = self.model(image_tensor)
                return embedding.cpu().numpy().flatten()

        except Exception as e:
            print(f"❌ Error extracting embedding: {e}")
            return None

    def process_face_image(self, image_path: str) -> Optional[Dict]:
        """
        ประมวลผลภาพเดียวและสกัด biometric features

        Args:
            image_path (str): เส้นทางไฟล์ภาพ

        Returns:
            Optional[Dict]: Dictionary ที่มี biometric data หรือ None หากเกิดข้อผิดพลาด
        """
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None

        # แปลงภาพเป็น tensor
        image_tensor = self.image_to_tensor(image_path)
        if image_tensor is None:
            return None

        # สกัด embedding
        embedding = self.extract_embedding(image_tensor)
        if embedding is None:
            return None

        # คำนวณ additional features
        embedding_norm = np.linalg.norm(embedding)
        normalized_embedding = (
            embedding / embedding_norm if embedding_norm > 0 else embedding
        )

        return {
            "embedding": embedding,
            "normalized_embedding": normalized_embedding,
            "embedding_norm": embedding_norm,
            "embedding_shape": embedding.shape,
            "model_type": self.model_type,
        }
    
    def process_face_image_from_memory(self, image_data: bytes) -> Optional[Dict]:
        """
        ประมวลผลภาพจาก memory data และสกัด biometric features
        
        Args:
            image_data (bytes): ข้อมูลภาพในรูปแบบ bytes
            
        Returns:
            Optional[Dict]: Dictionary ที่มี biometric data หรือ None หากเกิดข้อผิดพลาด
        """
        try:
            # แปลง bytes data เป็น PIL Image
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # แปลงภาพเป็น tensor
            image_tensor = self.image_to_tensor_from_memory(image)
            if image_tensor is None:
                return None

            # สกัด embedding
            embedding = self.extract_embedding(image_tensor)
            if embedding is None:
                return None

            # คำนวณ additional features
            embedding_norm = np.linalg.norm(embedding)
            normalized_embedding = (
                embedding / embedding_norm if embedding_norm > 0 else embedding
            )

            return {
                "embedding": embedding,
                "normalized_embedding": normalized_embedding,
                "embedding_norm": embedding_norm,
                "embedding_shape": embedding.shape,
                "model_type": self.model_type,
            }
            
        except Exception as e:
            print(f"❌ Error processing image from memory: {e}")
            return None

    def process_single_image(
        self, image_path: str, save_path: str = None, save_mode: str = "overwrite"
    ) -> Optional[Dict]:
        """
        ประมวลผลภาพเดียวและบันทึกผลลัพธ์ (ถ้าระบุ)

        Args:
            image_path (str): เส้นทางไฟล์ภาพ
            save_path (str): เส้นทางสำหรับบันทึกผลลัพธ์ (optional)
            save_mode (str): โหมดการบันทึก ('overwrite' หรือ 'insert')

        Returns:
            Optional[Dict]: Dictionary ของผลลัพธ์หรือ None หากเกิดข้อผิดพลาด
        """
        print(f"🔄 Processing single image: {image_path}")

        # สกัด biometric features
        result = self.process_face_image(image_path)

        if result is None:
            print(f"❌ Failed to process image: {image_path}")
            return None

        # บันทึกผลลัพธ์ถ้าระบุ
        if save_path:
            try:
                # สร้าง dictionary สำหรับบันทึก
                save_data = {os.path.basename(image_path): result}
                self.save_embeddings(save_data, save_path, mode=save_mode)
                print(f"💾 Saved result to: {save_path} (mode: {save_mode})")
            except Exception as e:
                print(f"⚠️  Warning: Could not save result: {e}")

        print(f"✅ Successfully processed: {image_path}")
        return result

    def process_batch_image(
        self, image_folder: str, output_file: str = None, save_mode: str = "overwrite"
    ) -> Dict[str, Dict]:
        """
        ประมวลผลภาพทั้งหมดในโฟลเดอร์

        Args:
            image_folder (str): โฟลเดอร์ที่มีภาพ
            output_file (str): ไฟล์สำหรับบันทึกผลลัพธ์ (optional)
            save_mode (str): โหมดการบันทึก ('overwrite' หรือ 'insert')

        Returns:
            Dict[str, Dict]: Dictionary ของผลลัพธ์โดยใช้ชื่อไฟล์เป็น key
        """
        results = {}
        processed_count = 0
        total_count = 0

        if not os.path.exists(image_folder):
            print(f"❌ Folder not found: {image_folder}")
            return results

        for filename in os.listdir(image_folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                total_count += 1
                image_path = os.path.join(image_folder, filename)

                result = self.process_face_image(image_path)
                if result:
                    results[filename] = result
                    processed_count += 1
                    print(f"✅ Processed: {filename}")
                else:
                    print(f"❌ Failed to process: {filename}")

        # บันทึกผลลัพธ์ถ้าระบุ output file
        if output_file and results:
            self.save_embeddings(results, output_file, mode=save_mode)

        print(
            f"\n📊 Batch processing complete: {processed_count}/{total_count} successful"
        )
        return results

    def save_embeddings(
        self, embeddings_dict: Dict, output_path: str, mode: str = "overwrite"
    ):
        """
        บันทึก embeddings ลงไฟล์

        Args:
            embeddings_dict (Dict): Dictionary ของ embeddings
            output_path (str): เส้นทางไฟล์ปลายทาง
            mode (str): โหมดการบันทึก ('overwrite' หรือ 'insert')
        """
        try:
            if mode == "insert" and os.path.exists(output_path):
                # โหลดข้อมูลที่มีอยู่
                existing_data = self.load_embeddings(output_path)
                if existing_data:
                    # ผสานข้อมูลใหม่กับข้อมูลที่มีอยู่
                    merged_data = {**existing_data, **embeddings_dict}
                    embeddings_dict = merged_data
                    print(f"📥 Loaded existing file with {len(existing_data)} entries")

            with open(output_path, "wb") as f:
                pickle.dump(embeddings_dict, f)

            print(f"💾 Saved {len(embeddings_dict)} embeddings to: {output_path}")
            print(f"📊 Mode: {mode}")

        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")

    def load_embeddings(self, input_path: str) -> Optional[Dict]:
        """
        โหลด embeddings จากไฟล์

        Args:
            input_path (str): เส้นทางไฟล์ต้นทาง

        Returns:
            Optional[Dict]: Dictionary ของ embeddings หรือ None หากเกิดข้อผิดพลาด
        """
        try:
            with open(input_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return None

    def compare_embeddings(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        เปรียบเทียบสอง embeddings และคืนค่า similarity score

        Args:
            embedding1 (np.ndarray): Embedding แรก
            embedding2 (np.ndarray): Embedding ที่สอง

        Returns:
            float: Cosine similarity score (0-1)
        """
        # Normalize embeddings
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)

        # คำนวณ cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)

        # แปลงเป็น probability-like score (0-1)
        return (similarity + 1) / 2

    def verify_identity(
        self,
        probe_embedding: np.ndarray,
        reference_embeddings: Dict,
        threshold: float = 0.6,
    ) -> Dict:
        """
        ตรวจสอบ identity โดยเปรียบเทียบกับ reference embeddings

        Args:
            probe_embedding (np.ndarray): Embedding ที่ต้องการตรวจสอบ
            reference_embeddings (Dict): Dictionary ของ reference embeddings
            threshold (float): Threshold สำหรับการยืนยัน identity

        Returns:
            Dict: ผลลัพธ์การยืนยัน identity
        """
        best_score = 0.0
        best_match = None

        for filename, ref_data in reference_embeddings.items():
            score = self.compare_embeddings(probe_embedding, ref_data["embedding"])

            if score > best_score:
                best_score = score
                best_match = filename

        is_verified = best_score >= threshold

        return {
            "verified": is_verified,
            "best_score": best_score,
            "best_match": best_match,
            "threshold": threshold,
        }

    def get_matches(
        self,
        probe_embedding: np.ndarray,
        reference_embeddings: Dict,
        threshold: float = 0.6,
    ) -> Dict:
        """
        หา matches ทั้งหมดที่มีคะแนน similarity มากกว่าหรือเท่ากับ threshold

        Args:
            probe_embedding (np.ndarray): Embedding ที่ต้องการตรวจสอบ
            reference_embeddings (Dict): Dictionary ของ reference embeddings
            threshold (float): คะแนน similarity ขั้นต่ำที่ต้องการ

        Returns:
            Dict: ผลลัพธ์ matches ที่ผ่านเกณฑ์
        """
        matches = []

        for filename, ref_data in reference_embeddings.items():
            score = self.compare_embeddings(probe_embedding, ref_data["embedding"])
            if score >= threshold:
                matches.append({"filename": filename, "score": score})

        # เรียงลำดับจากคะแนนสูงไปต่ำ
        matches.sort(key=lambda x: x["score"], reverse=True)

        return {
            "matches": matches,
            "total_matches": len(matches),
        }

    def get_model_info(self) -> Dict:
        """
        คืนค่าข้อมูลเกี่ยวกับโมเดล

        Returns:
            Dict: ข้อมูลโมเดล
        """
        return {
            "model_type": self.model_type,
            "device": self.device,
            "embedding_dim": 512,
            "model_architecture": "InceptionResnetV1",
        }


# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # สร้าง instance
    biometric_converter = BiometricConverter(model_type="vggface2")

    start_time = time.time()
    min_score = 0.6

    # สกัด biometric features
    print("⛏️ Extract biometric features: ")
    extractbiometric = biometric_converter.process_face_image(image_path="testFace.png")
    print(extractbiometric['embedding'])

    # save_mode: insert or overwrite

    # ตัวอย่างการประมวลผลทั้งโฟลเดอร์
    # print("📷 Processing batch image example:")
    # results = biometric_converter.process_batch_image("cropped_images", "embeddings.pkl", save_mode="overwrite")

    # ตัวอย่างการประมวลผลภาพเดียว
    # print("📷 Processing single image example:")
    # single_result = biometric_converter.process_single_image(
    #     image_path="testFace.png",
    #     save_path="embeddings.pkl",
    #     save_mode="insert"
    # )

    # ตัวอย่างการโหลดและเปรียบเทียบ
    # loaded_embeddings = biometric_converter.load_embeddings("embeddings.pkl")
    # if loaded_embeddings:
    #     probe_result = biometric_converter.process_face_image("testFace.png")
    #     if probe_result:
    #         # ใช้ฟังก์ชัน verify_identity แยก
    #         verification_result = biometric_converter.verify_identity(
    #             probe_result["embedding"], loaded_embeddings, threshold=0.8
    #         )

    #         # ใช้ฟังก์ชัน get_top_matches แยก
    #         matches_result = biometric_converter.get_matches(
    #             probe_result["embedding"], loaded_embeddings, threshold=min_score
    #         )

    #         print("=" * 60)
    #         print(f"🔍 Verified: {verification_result['verified']}")
    #         print(f"📊 Best Score: {verification_result['best_score']:.4f}")
    #         print(f"🎯 Best Match: {verification_result['best_match']}")
    #         print("")

    #         print(f"🏆 Min Score {min_score} Matches:")
    #         for i, match in enumerate(matches_result["matches"], 1):
    #             print(
    #                 f"{i}. {match['filename']} ความมั่นใจ: {match['score']:.4f}"
    #             )

    #         print("=" * 60)
    #         print(f"📈 Total matches: {matches_result['total_matches']}/{len(loaded_embeddings)}")

    #         processing_time = time.time() - start_time
    #         print(f"⏱️  เวลาประมวลผล: {processing_time:.2f} วินาที")
