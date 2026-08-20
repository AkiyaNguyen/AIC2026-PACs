# PACs Search UI

Giao diện Textual KIS cho retrieval backend của PACs AIC 2026.

## Chức năng

- Theo dõi trạng thái FastAPI backend.
- Gửi truy vấn tới `POST /search`.
- Điều chỉnh candidate visual, số result, fusion visual/transcript và MiniLM/BM25.
- Xem danh sách keyframe theo rank và inspect đầy đủ metadata.
- Phát toàn bộ video từ đúng timestamp của keyframe, tua tự do hoặc nhảy ±5 giây.
- Theo dõi timestamp/frame ID tại playhead và sao chép đáp án KIS đã điều chỉnh.
- Sao chép đáp án KIS `video_id, frame_idx`.
- Xuất toàn bộ ranked results thành CSV.

UI hiển thị keyframe từ `thumbnail_url` do backend cung cấp và tự quay về media
placeholder nếu batch tương ứng chưa có trong `KEYFRAMES_ROOT`.
Khi kết quả có `video_url`, inspector dùng native video player với `preload=metadata`;
video không tự phát và chỉ tải các byte cần thiết khi xem hoặc tua.

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
