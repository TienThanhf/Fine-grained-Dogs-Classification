# Fine-grained-Dogs-Classification

Dự án phân loại các giống chó sử dụng mô hình học sâu.

## Danh sách các file và thư mục chính
- **`cropped_dogs/`**: Dataset (tập dữ liệu ảnh các giống chó) sau khi đã được crop.
- **`app.py`**: Giao diện demo của ứng dụng được xây dựng bằng Streamlit.
- **`convnext-tiny.ipynb`**: Notebook dùng để huấn luyện (train) mô hình ConvNeXt.
- **`best_model.pth`**: File trọng số mô hình (weights) tối ưu sau khi huấn luyện, được lưu trữ bằng Git LFS do có dung lượng lớn (>100MB).

## Hướng dẫn upload file lớn (Model Weights .pth) lên GitHub bằng Git LFS

Do GitHub giới hạn dung lượng file upload thông thường dưới 100MB, các file mô hình như `best_model.pth` (khoảng 111MB) cần được upload thông qua **Git LFS (Large File Storage)**. 

Dưới đây là các bước thực hiện qua Terminal:

1. **Cài đặt Git LFS trên máy tính (Ubuntu/Debian):**
   ```bash
   sudo apt update && sudo apt install git-lfs -y
   ```

2. **Kích hoạt Git LFS trong thư mục dự án:**
   ```bash
   git lfs install
   ```

3. **Cấu hình Git LFS theo dõi định dạng `.pth`:**
   ```bash
   git lfs track "*.pth"
   ```
   *(Thao tác này sẽ tạo/cập nhật file `.gitattributes` để Git biết cần xử lý đặc biệt với các file `.pth`)*

4. **Thêm file cấu hình và file model vào hàng đợi:**
   ```bash
   git add .gitattributes best_model.pth
   ```

5. **Commit và đẩy lên GitHub:**
   ```bash
   git commit -m "Thêm model weights bằng Git LFS"
   git push origin main
   ```

