"""Explicit errors (§2.12). Every failure the owner can see carries a stable code and Vietnamese text.

The wording of input-check failures is Pending in the design (§15.3); the texts below are the Proposed wording.
"""

from __future__ import annotations

MESSAGES_VI: dict[str, str] = {
    # Input checks (§4.4) — Proposed wording
    "input_not_found": "Không tìm thấy tệp ảnh.",
    "input_unreadable": "Không mở được tệp này. Hãy chọn ảnh PNG hoặc WebP.",
    "input_too_small": "Ảnh quá nhỏ (cạnh ngắn nhất phải từ {min} px).",
    "input_too_large": "Ảnh quá lớn (cạnh dài nhất tối đa {max} px).",
    "input_no_alpha": "Ảnh không có nền trong suốt. Hãy xuất ảnh PNG có kênh alpha.",
    "input_checkerboard": "Nền ô vuông được vẽ thẳng vào ảnh, không phải nền trong suốt thật. "
                          "Hãy xuất lại ảnh với nền trong suốt.",
    "input_no_transparency": "Ảnh có kênh alpha nhưng gần như không có phần nào trong suốt.",
    "input_empty": "Không thấy nhân vật trong ảnh (ảnh gần như trong suốt hoàn toàn).",
    "input_many_figures": "Ảnh có {count} hình tách rời. Mỗi ảnh chỉ được có một nhân vật.",
    # Generation
    "no_character": "Hãy chọn ảnh nhân vật trước.",
    "skeleton_unconfirmed": "Hãy kiểm tra và xác nhận khung xương của nhân vật trước khi tạo.",
    "skeleton_invalid": "Khung xương không hợp lệ: các khớp phải nằm trong ảnh.",
    "skeleton_incomplete": "Khung xương cần ít nhất cổ, hông và cổ chân.",
    "unknown_animation": "Không có hoạt ảnh này.",
    "animation_not_ready": "Hoạt ảnh này chưa có trong phiên bản hiện tại.",
    "busy": "Đang có một việc khác chạy. Hãy chờ xong hoặc huỷ.",
    "gpu_out_of_memory": "Không đủ bộ nhớ GPU cho khung hình này.",
    "inference_failed": "Tạo khung hình bị lỗi: {detail}",
    "models_missing": "Thiếu mô hình AI trong thư mục cài đặt: {detail}",
    "frame_failed_after_retries": "Khung {frame} vẫn lỗi sau {tries} lần thử. Bạn có thể tạo lại khung này.",
    "cancelled": "Đã huỷ.",
    "rig_missing": "Hãy tạo khung nhân vật (bước 2) trước khi tạo hoạt ảnh.",
    "rig_failed": "Không dựng được khung nhân vật từ các bộ phận này. Hãy bấm “Tách lại”.",
    "unknown_part": "Không có phần này trên nhân vật.",
    "rig_edit_missing": "Chưa có tệp Photoshop các bộ phận. Hãy bấm “Sửa trong Photoshop” trước.",
    "rig_edit_unreadable": "Không đọc được tệp Photoshop đã sửa. Hãy lưu lại dạng PSD (Lưu, không phải Xuất).",
    "rig_edit_size": "Tệp Photoshop phải giữ nguyên kích thước khung. Đừng cắt hay đổi cỡ hình.",
    "rig_edit_failed": "Không dựng được khung nhân vật từ tệp đã sửa nên đã giữ các bộ phận cũ. "
                       "Hãy giữ tên các lớp, giữ lớp “Thân”, và giữ tay chân ở chỗ cũ.",
    "rig_layers_missing": "Khung nhân vật này làm bằng bản cũ, chưa có các bộ phận riêng. "
                          "Hãy bấm “Làm lại khung nhân vật” để tạo.",
    "rig_map_missing": "Không còn bản đồ bộ phận của nhân vật này. Hãy bấm “Tách lại” để tách các bộ phận lần nữa.",
    "rig_psd_busy": "Không ghi được tệp Photoshop các bộ phận. Hãy đóng tệp đó trong Photoshop rồi thử lại.",
    "nothing_to_redraw": "Khung hình làm từ khung nhân vật luôn giống nhau; chỉ bản hoàn thiện bằng AI mới vẽ lại được.",
    "no_frames": "Chưa có khung hình nào.",
    "no_candidate": "Khung này không có bản tạo lại để so sánh.",
    "frame_index": "Số khung không hợp lệ.",
    # Export
    "export_incomplete": "Chưa đủ 8 khung hợp lệ để xuất.",
    "export_bad_size": "Kích thước xuất không hợp lệ.",
    "export_write_failed": "Không ghi được tệp: {detail}",
    "export_verify_failed": "Tệp vừa ghi không hợp lệ: {detail}",
    "export_bad_folder": "Không dùng được thư mục xuất này.",
    "bad_request": "Yêu cầu không hợp lệ: {detail}",
}


class AppError(Exception):
    """An error with a stable code and a Vietnamese message for the owner."""

    def __init__(self, code: str, **params: object) -> None:
        self.code = code
        self.params = {k: v for k, v in params.items()}
        template = MESSAGES_VI.get(code, code)
        try:
            self.message = template.format(**self.params)
        except (KeyError, IndexError):
            self.message = template
        super().__init__(f"{code}: {self.message}")

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message, "params": {k: str(v) for k, v in self.params.items()}}
