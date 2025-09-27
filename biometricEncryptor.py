import json
import base64
import numpy as np
from phe import paillier
import os
import getpass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class biometricEncryptor:
    """
    Biometric encryption system using Paillier Homomorphic Encryption
    with standard key storage format (JSON + optional encryption)
    """

    def __init__(self, key_dir="keys", password=None):
        """
        Initialize the biometric encryptor

        Args:
            key_dir: Directory to store key files
            password: Optional password for private key encryption
        """
        self.private_key = None
        self.public_key = None
        self.key_dir = key_dir
        self.password = password

        # Create key directory if it doesn't exist
        os.makedirs(key_dir, exist_ok=True)

        self.public_key_path = os.path.join(key_dir, "public_key.json")
        self.private_key_path = os.path.join(key_dir, "private_key.json")

        # Try to load existing keys
        self._load_keys()

        # If keys don't exist, generate new ones
        if self.public_key is None or self.private_key is None:
            self._generate_keys()
            self._save_keys()

    def _generate_keys(self):
        """Generate new Paillier key pair with recommended key size"""
        print("Generating new Paillier key pair (2048-bit recommended)...")
        # Using 2048-bit keys for better security
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=2048
        )
        print("Key generation completed!")

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def _save_keys(self):
        """Save keys in standard JSON format"""
        # Save public key (unencrypted, as it's meant to be public)
        pub_data = {
            "algorithm": "Paillier",
            "key_size": 2048,
            "created": np.datetime64("now").astype(str),
            "parameters": {"n": int(self.public_key.n), "g": int(self.public_key.g)},
        }

        with open(self.public_key_path, "w") as f:
            json.dump(pub_data, f, indent=2)

        # Save private key (optionally encrypted)
        priv_data = {
            "algorithm": "Paillier",
            "key_size": 2048,
            "created": np.datetime64("now").astype(str),
            "parameters": {
                "p": int(self.private_key.p),
                "q": int(self.private_key.q),
                "public_key": {
                    "n": int(self.public_key.n),
                    "g": int(self.public_key.g),
                },
            },
        }

        if self.password:
            # Encrypt private key
            salt = os.urandom(16)
            key = self._derive_key(self.password, salt)
            fernet = Fernet(key)

            encrypted_data = fernet.encrypt(json.dumps(priv_data).encode())

            encrypted_priv_data = {
                "encrypted": True,
                "algorithm": "AES-256-CBC",
                "kdf": "PBKDF2-HMAC-SHA256",
                "iterations": 100000,
                "salt": base64.b64encode(salt).decode(),
                "data": base64.b64encode(encrypted_data).decode(),
            }

            with open(self.private_key_path, "w") as f:
                json.dump(encrypted_priv_data, f, indent=2)
            print("Private key saved with encryption")
        else:
            # Save without encryption (not recommended for production)
            with open(self.private_key_path, "w") as f:
                json.dump(priv_data, f, indent=2)
            print("Warning: Private key saved without encryption")

        print(f"Keys saved to {self.key_dir}/")

    def _load_keys(self):
        """Load keys from standard JSON files"""
        try:
            # Load public key
            if os.path.exists(self.public_key_path):
                with open(self.public_key_path, "r") as f:
                    pub_data = json.load(f)

                n = pub_data["parameters"]["n"]
                self.public_key = paillier.PaillierPublicKey(n=n)
                print("Public key loaded successfully")
            else:
                print("Public key file not found")
                return

            # Load private key
            if os.path.exists(self.private_key_path):
                with open(self.private_key_path, "r") as f:
                    priv_data = json.load(f)

                if priv_data.get("encrypted", False):
                    if not self.password:
                        self.password = getpass.getpass(
                            "Enter password for private key: "
                        )

                    salt = base64.b64decode(priv_data["salt"])
                    key = self._derive_key(self.password, salt)
                    fernet = Fernet(key)

                    encrypted_data = base64.b64decode(priv_data["data"])
                    decrypted_data = fernet.decrypt(encrypted_data)
                    priv_data = json.loads(decrypted_data.decode())

                p = priv_data["parameters"]["p"]
                q = priv_data["parameters"]["q"]
                self.private_key = paillier.PaillierPrivateKey(
                    public_key=self.public_key, p=p, q=q
                )
                print("Private key loaded successfully")

            else:
                print("Private key file not found")
                return

        except Exception as e:
            print(f"Error loading keys: {e}")
            self.public_key = None
            self.private_key = None

    def normalize_embedding(self, embedding):
        """
        Normalize embedding to unit norm
        """
        if isinstance(embedding, np.ndarray):
            embedding = embedding.flatten()
        else:
            embedding = np.array(embedding)

        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")

        normalized = embedding / norm
        return normalized

    def encrypt_embedding(self, embedding):
        """
        Encrypt a biometric embedding with normalization
        """
        # Normalize first
        normalized_embedding = self.normalize_embedding(embedding)

        # Convert to Python float list
        embedding_list = normalized_embedding.tolist()
        embedding_list = [float(x) for x in embedding_list]

        encrypted_embedding = [self.public_key.encrypt(x) for x in embedding_list]
        return encrypted_embedding

    def decrypt_embedding(self, encrypted_embedding):
        """
        Decrypt an encrypted biometric embedding

        Args:
            encrypted_embedding: list of encrypted numbers

        Returns:
            numpy array of decrypted values
        """
        decrypted_embedding = [self.private_key.decrypt(x) for x in encrypted_embedding]
        return np.array(decrypted_embedding)

    def encrypt_embedding_batch(self, embeddings):
        """
        Encrypt multiple embeddings at once

        Args:
            embeddings: 2D numpy array or list of lists

        Returns:
            List of lists of encrypted numbers
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        return [self.encrypt_embedding(embedding) for embedding in embeddings]

    def homomorphic_euclidean_distance(self, encrypted_vec, plain_vec):
        """
        Calculate Euclidean distance homomorphically
        Only one vector needs to be encrypted

        Args:
            encrypted_vec: encrypted embedding
            plain_vec: plaintext embedding (numpy array or list)

        Returns:
            Encrypted distance squared
        """
        if len(encrypted_vec) != len(plain_vec):
            raise ValueError("Vectors must have the same length")

        # Convert to Python float list (สำคัญ!)
        plain_vec = [float(x) for x in plain_vec]

        # Calculate Euclidean distance squared: Σ(encrypted_vec[i] - plain_vec[i])^2
        # Using the formula: Σ(E(a_i) * (-2*b_i) + E(b_i^2)) + Σ(b_i^2)
        distance_sq = self.public_key.encrypt(0.0)  # Start with encrypted zero

        for i in range(len(encrypted_vec)):
            # Term 1: E(a_i) * (-2*b_i) = E(-2*a_i*b_i)
            term1 = encrypted_vec[i] * (-2.0 * plain_vec[i])

            # Term 2: Encrypt b_i^2 (since we can't multiply two encrypted numbers)
            term2 = self.public_key.encrypt(plain_vec[i] * plain_vec[i])

            # Add both terms
            distance_sq = distance_sq + term1 + term2

        # Add the sum of squares of plain_vec (this is known plaintext)
        sum_plain_sq = sum(x * x for x in plain_vec)
        distance_sq = distance_sq + sum_plain_sq

        return distance_sq

    def homomorphic_dot_product(self, encrypted_vec, plain_vec):
        """
        Calculate dot product homomorphically
        Only one vector needs to be encrypted

        Args:
            encrypted_vec: encrypted embedding
            plain_vec: plaintext embedding (convert to Python float)

        Returns:
            Encrypted dot product
        """
        if len(encrypted_vec) != len(plain_vec):
            raise ValueError("Vectors must have the same length")

        # Convert plain_vec to Python float list
        plain_vec = [float(x) for x in plain_vec]

        dot_product = encrypted_vec[0] * plain_vec[0]  # E(a1 * b1)

        for i in range(1, len(encrypted_vec)):
            # Homomorphic multiplication: E(a_i) * b_i = E(a_i * b_i)
            term = encrypted_vec[i] * plain_vec[i]
            # Homomorphic addition: E(sum) + E(term) = E(sum + term)
            dot_product = dot_product + term

        return dot_product

    def homomorphic_similarity_search(self, encrypted_query, plain_database):
        """
        Perform similarity search between encrypted query and plaintext database

        Args:
            encrypted_query: encrypted query embedding
            plain_database: list of plaintext database embeddings

        Returns:
            List of encrypted similarity scores
        """
        similarities = []

        for plain_vec in plain_database:
            # Use dot product as similarity measure
            similarity = self.homomorphic_dot_product(encrypted_query, plain_vec)
            similarities.append(similarity)

        return similarities

    def serialize_encrypted_data(self, encrypted_data):
        """
        Serialize encrypted data for storage/transmission

        Args:
            encrypted_data: encrypted number or list of encrypted numbers

        Returns:
            Serialized string or list of strings
        """
        if isinstance(encrypted_data, list):
            return [self._serialize_single(x) for x in encrypted_data]
        else:
            return self._serialize_single(encrypted_data)

    def _serialize_single(self, encrypted_number):
        """Serialize a single encrypted number"""
        data = {
            "ciphertext": str(encrypted_number.ciphertext()),
            "exponent": encrypted_number.exponent,
        }
        return base64.b64encode(json.dumps(data).encode()).decode()

    def deserialize_encrypted_data(self, serialized_data):
        """
        Deserialize encrypted data

        Args:
            serialized_data: serialized string or list of strings

        Returns:
            Encrypted number or list of encrypted numbers
        """
        if isinstance(serialized_data, list):
            return [self._deserialize_single(x) for x in serialized_data]
        else:
            return self._deserialize_single(serialized_data)

    def _deserialize_single(self, serialized_string):
        """Deserialize a single encrypted number"""
        data = json.loads(base64.b64decode(serialized_string.encode()).decode())
        return paillier.EncryptedNumber(
            self.public_key, int(data["ciphertext"]), exponent=data["exponent"]
        )

    def save_encrypted_template(self, encrypted_embedding, file_path):
        """
        Save encrypted biometric template to file

        Args:
            encrypted_embedding: encrypted embedding
            file_path: path to save file
        """
        serialized = self.serialize_encrypted_data(encrypted_embedding)
        template_data = {
            "version": "1.0",
            "algorithm": "Paillier",
            "vector_length": len(encrypted_embedding),
            "created": np.datetime64("now").astype(str),
            "data": serialized,
        }

        with open(file_path, "w") as f:
            json.dump(template_data, f, indent=2)

        print(f"✅ Encrypted Template saved to {file_path}")

    def load_encrypted_template(self, file_path):
        """
        Load encrypted biometric template from file

        Args:
            file_path: path to template file

        Returns:
            Encrypted embedding
        """
        with open(file_path, "r") as f:
            template_data = json.load(f)

        return self.deserialize_encrypted_data(template_data["data"])

    def get_key_info(self):
        """Get information about the current keys"""
        if self.public_key:
            key_size = self.public_key.n.bit_length()
            return {
                "key_size": key_size,
                "algorithm": "Paillier",
                "public_key_file": self.public_key_path,
                "private_key_file": self.private_key_path,
            }
        return None

    def save_original_template(self, embedding, file_path):
        """
        บันทึก original embedding ลงไฟล์

        Args:
            embedding: ข้อมูลต้นฉบับ
            file_path: path เพื่อบันทึกไฟล์
        """
        if isinstance(embedding, np.ndarray):
            embedding_data = embedding.tolist()
            shape = embedding.shape
            dtype = str(embedding.dtype)
        else:
            embedding_data = embedding
            shape = [len(embedding)]
            dtype = "list"

        template_data = {
            "version": "1.0",
            "type": "original_embedding",
            "shape": shape,
            "dtype": dtype,
            "created": np.datetime64("now").astype(str),
            "data": embedding_data,
        }

        with open(file_path, "w") as f:
            json.dump(template_data, f, indent=2)

        print(f"✅ Original Template saved to {file_path}")

    def load_original_template(self, file_path):
        """
        โหลด original embedding จากไฟล์

        Args:
            file_path: path ของไฟล์

        Returns:
            numpy array ของข้อมูลต้นฉบับ
        """
        with open(file_path, "r") as f:
            template_data = json.load(f)

        embedding = np.array(template_data["data"])
        print(f"✅ Loaded original data from file {file_path}: shape {embedding.shape}")
        return embedding

    def homomorphic_verification(self, new_embedding, encrypted_template):
        """
        เปรียบเทียบ embedding ใหม่กับ encrypted template
        """
        # Convert และ normalize new embedding
        if isinstance(new_embedding, np.ndarray):
            new_embedding_array = new_embedding.astype(np.float64).flatten()
        else:
            new_embedding_array = np.array([float(x) for x in new_embedding])

        # Normalize ให้มี norm = 1
        original_norm = np.linalg.norm(new_embedding_array)
        new_embedding_normalized = new_embedding_array / original_norm

        # คำนวณ dot product
        encrypted_similarity = self.homomorphic_dot_product(
            encrypted_template, new_embedding_normalized.tolist()
        )

        return encrypted_similarity


# Example usage and testing
def testFormNew():
    """Example usage of the biometricEncryptor"""

    # Initialize with password protection (recommended)
    password = getpass.getpass(
        "Set password for private key (optional, press enter to skip): "
    )
    encryptor = biometricEncryptor(password=password if password else None)

    # Display key information
    key_info = encryptor.get_key_info()
    print(f"Using {key_info['key_size']}-bit Paillier keys")

    # Example: Face recognition embedding (128-dimensional)
    print("\n=== Biometric Encryption Demo ===")

    # สร้าง biometric จำลองเพื่อทดสอบ
    # Original biometric template - USE FLOAT64 INSTEAD OF FLOAT32
    original_embedding = np.random.randn(128).astype(np.float64)  # Changed to float64
    print(f"Original Embedding shape: {original_embedding.shape}")
    print(f"First 5 values: {original_embedding[:5]}")
    # Save original template for reference
    encryptor.save_original_template(original_embedding, "original_template.json")
    # Encryption
    print("\nEncrypting biometric template...")
    encrypted_embedding = encryptor.encrypt_embedding(original_embedding)
    # Save template
    encryptor.save_encrypted_template(encrypted_embedding, "encrypted_template.json")

def testFormFile():
    """Example usage of the biometricEncryptor"""

    # Initialize with password protection (recommended)
    password = getpass.getpass(
        "Set password for private key (optional, press enter to skip): "
    )
    encryptor = biometricEncryptor(password=password if password else None)

    # Display key information
    key_info = encryptor.get_key_info()
    print(f"Using {key_info['key_size']}-bit Paillier keys")

    # Example: Face recognition embedding (128-dimensional)
    print("\n=== Biometric Encryption Demo ===")

    # โหลดกลับมาเพื่อตรวจสอบ
    original_embedding = encryptor.load_original_template("original_template.json")
    print(f"Loaded Original Embedding shape: {original_embedding.shape}")
    print(f"First 5 values: {original_embedding[:5]}")
    print("Original Template loaded successfully!")
    # Load template
    print("\nLoading Encrypted Template...")
    encrypted_embedding = encryptor.load_encrypted_template("encrypted_template.json")
    print("Encrypted Template loaded successfully!")

    # ทดสอบกับ embedding ที่ต่าง
    threshold = 0.8  # กำหนด threshold สำหรับการตรวจสอบ
    print("\nTesting with DIFFERENT embedding:")
    similarity = encryptor.homomorphic_verification(
        original_embedding, encrypted_embedding
    )

    decrypted_similarity = encryptor.private_key.decrypt(similarity)
    print(f"📊 similarity (decrypted): {decrypted_similarity:.3f}")

    if decrypted_similarity > threshold:
        print("✅ VERIFICATION PASSED - คล้ายกันมาก")
    else:
        print("❌ VERIFICATION FAILED - ไม่คล้ายกัน")

    # Decryption with private key
    # print("\nDecrypting template...")
    # decrypted_embedding = encryptor.decrypt_embedding(encrypted_embedding)
    # print(f"Decrypted first 5 values: {decrypted_embedding[:5]}")


if __name__ == "__main__":
    # ทดสอบแบบสร้างข้อมูลใหม่
    # testFormNew()

    # ทดสอบแบบโหลดข้อมูลมาจากไฟล์
    testFormFile()