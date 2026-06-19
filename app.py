import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_tiny
from PIL import Image


MODEL_PATH  = "/home/thanhf/Downloads/dog_breed/best_model.pth"   # file .pth sau khi train
#MODEL_PATH  = "/home/thanhf/Downloads/dog_breed/best_model_convnext_tiny.pth"
DATASET_DIR = "/home/thanhf/Downloads/dog_breed/cropped_dogs"     # ImageFolder('dataset') — thẳng folder gốc
TOP_K       = 5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────
# CLASS NAMES
# Đọc subfolders của dataset/ theo thứ tự sort alphabet
# → khớp 100% với thứ tự class_to_idx của ImageFolder lúc train
# ──────────────────────────────────────────────
@st.cache_data
def load_class_names() -> list:
    if not os.path.isdir(DATASET_DIR):
        st.error(
            f"❌ Không tìm thấy thư mục `{DATASET_DIR}/`.\n\n"
            "Hãy đặt folder dataset cùng chỗ với app.py, "
            f"hoặc chỉnh `DATASET_DIR` trong CONFIG."
        )
        st.stop()

    classes = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    if len(classes) == 0:
        st.error(f"❌ Folder `{DATASET_DIR}/` không chứa subfolder nào.")
        st.stop()

    return classes          # vd: ['n02085620-Chihuahua', 'n02086240-Shih-Tzu', ...]


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
@st.cache_resource
def load_model(num_classes: int):
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    # Khi train dùng nn.Sequential bọc ngoài Linear
    # → checkpoint lưu key "classifier.2.1.weight" thay vì "classifier.2.weight"
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.0),          # index 0 — dropout (p có thể khác, không ảnh hưởng inference)
        nn.Linear(in_features, num_classes),  # index 1
    )

    if not os.path.isfile(MODEL_PATH):
        st.error(
            f"❌ Không tìm thấy `{MODEL_PATH}`.\n\n"
            "Hãy đặt file model.pth cùng thư mục với app.py."
        )
        st.stop()

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    # Hỗ trợ cả state_dict thuần lẫn dict bọc ngoài
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt:
                ckpt = ckpt[key]
                break

    model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()
    return model


# ──────────────────────────────────────────────
# TRANSFORM  (giống validation lúc train)
# ──────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def pretty(raw: str) -> str:
    """
    'n02085620-Chihuahua'        ->  'Chihuahua'
    'n02116738-African_hunting_dog' ->  'African Hunting Dog'
    """
    name = raw
    # Bỏ prefix nXXXXXXX- (đúng định dạng Stanford Dogs)
    if "-" in name and name.split("-")[0][1:].isdigit():
        name = "-".join(name.split("-")[1:])
    return name.replace("_", " ").replace("-", " ").title()


@torch.inference_mode()
def predict(image: Image.Image, model, class_names: list, top_k: int):
    tensor = val_transform(image).unsqueeze(0).to(DEVICE)
    probs  = torch.softmax(model(tensor), dim=1)[0]
    k      = min(top_k, len(class_names))
    top_probs, top_idx = torch.topk(probs, k)
    return [(class_names[i], p.item()) for i, p in zip(top_idx, top_probs)]


# ──────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Dog Breed Classifier",
        page_icon="🐶",
        layout="wide",
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        top_k = st.slider("Số giống hiển thị (Top-K)", 1, 10, TOP_K)
        st.divider()
        st.markdown("**Model:** ConvNeXt-Tiny")
        st.markdown("**Dataset:** Stanford Dogs 120")
        st.markdown(f"**Device:** `{DEVICE.upper()}`")

    # Load model + class names (cached)
    class_names = load_class_names()
    model       = load_model(num_classes=len(class_names))

    # Header
    st.title("🐶 Dog Breed Classifier")
    st.markdown(
        f"Upload ảnh chó – model **ConvNeXt-Tiny** sẽ dự đoán "
        f"trong số **{len(class_names)} giống** từ Stanford Dogs dataset."
    )
    st.divider()

    # Upload
    uploaded = st.file_uploader(
        "📁 Chọn ảnh (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is None:
        st.info("👆 Upload ảnh chó để bắt đầu.")
        return

    image = Image.open(uploaded).convert("RGB")

    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Ảnh đã upload")
        st.image(image, use_container_width=True)

    with col_res:
        st.subheader("Kết quả dự đoán")

        with st.spinner("🔍 Đang phân tích..."):
            results = predict(image, model, class_names, top_k)

        top_name, top_prob = results[0]

        # Top-1 card nổi bật
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a73e8, #0d47a1);
                border-radius: 12px;
                padding: 20px 24px;
                color: white;
                margin-bottom: 20px;
            ">
                <div style="font-size:13px; opacity:.75; margin-bottom:4px">
                    🏆 Dự đoán hàng đầu
                </div>
                <div style="font-size:30px; font-weight:700; letter-spacing:.3px">
                    {pretty(top_name)}
                </div>
                <div style="font-size:20px; margin-top:8px">
                    {top_prob * 100:.1f}% tin cậy
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Top-K progress bars
        st.markdown(f"**Top-{top_k} giống gần nhất:**")
        for rank, (name, prob) in enumerate(results, 1):
            st.progress(prob, text=f"{rank}. {pretty(name)} — **{prob*100:.1f}%**")

    st.divider()
    st.caption("Powered by ConvNeXt-Tiny · PyTorch · Streamlit")


if __name__ == "__main__":
    main()