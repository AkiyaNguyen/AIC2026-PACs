# PACs Search UI

Giao diện Textual KIS cho retrieval backend của PACs AIC 2026.

## Chức năng

- Theo dõi trạng thái FastAPI backend.
- Gửi truy vấn tới `POST /search`.
- Điều chỉnh số candidate, số result, trọng số visual/ASR và ASR window.
- Xem danh sách keyframe theo rank và inspect đầy đủ metadata.
- Sao chép đáp án KIS `video_id, frame_idx`.
- Xuất toàn bộ ranked results thành CSV.

Backend hiện chưa cung cấp ảnh keyframe hoặc video endpoint, vì vậy UI dùng media placeholder và vẫn hiển thị toàn bộ metadata hiện có. Trường `thumbnail_url` đã được hỗ trợ ở phía UI để có thể hiển thị ảnh khi backend bổ sung media URL.

## Cấu hình

Sao chép `.env.example` thành `.env.local` nếu backend không chạy ở địa chỉ mặc định:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

FastAPI mặc định cho phép UI từ `http://localhost:3000` và `http://127.0.0.1:3000`. Có thể thay đổi bằng biến `CORS_ORIGINS` của backend, phân tách nhiều origin bằng dấu phẩy.

## Chạy local

```bash
npm install
npm run dev
```

Mở `http://localhost:3000`.

## Kiểm tra

```bash
npx tsc --noEmit
npm run build
```

Trên Windows ARM64, local development tự bỏ qua Cloudflare `workerd` vì package này không cung cấp binary tương ứng. Các môi trường x64 và hosted build vẫn dùng Cloudflare plugin.
