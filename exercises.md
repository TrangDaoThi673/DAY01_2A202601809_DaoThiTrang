# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Tăng temperature làm câu trả lời chuyển từ chính xác, cố định sang linh hoạt và ngẫu nhiên hơn. Ở mức 0.0 – 0.5, mô hình ưu tiên các thông tin chuẩn mực, dễ lặp lại kết quả giữa các lần gọi. Lên mức 1.0 – 1.5, văn phong trở nên đa dạng, sáng tạo hơn nhưng dễ xuất hiện câu từ rời rạc hoặc thông tin sai thực tế (hallucination).*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Tôi sẽ chọn temperature = 0.0 đến 0.2 cho chatbot hỗ trợ khách hàng. Nguyên nhân là hệ thống chăm sóc khách hàng ưu tiên hàng đầu tính chính xác, tính nhất quán và tuân thủ chặt chẽ tài liệu/quy trình kinh doanh; việc đặt temperature thấp giúp hạn chế tối đa việc mô hình "sáng tạo" ra thông tin sai sự thật gây ảnh hưởng xấu tới doanh nghiệp.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *Chi phí đắt hơn: Bảng giá GPT-4o output token ($10.00 / 1M token) đắt gấp 10 lần so với GPT-4o-mini ($1.00 / 1M token) với cùng workload $10.000 \times 3 \times 350 = 10.500.000$ output token/ngày. Trường hợp GPT-4o xứng đáng: Phân tích hợp đồng pháp lý hoặc tổng hợp báo cáo tài chính phức tạp đòi hỏi khả năng suy luận logic sâu và chính xác tuyệt đối.Trường hợp nên dùng GPT-4o-mini: Phân loại ý định người dùng (intent classification), tóm tắt tin nhắn ngắn hoặc trả lời các câu hỏi FAQ đơn giản.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Ở system prompt giáo viên tiểu học, phản hồi ngắn gọn, dùng từ ngữ mộc mạc, hình ảnh ẩn dụ gần gũi (như cuốn sổ nhật ký chung) để giải thích. Ngược lại, prompt chuyên gia tài chính cho ra văn bản dài hơn, sử dụng thuật ngữ chuyên ngành (sổ cái phân tán, mật mã học, cơ chế đồng thuận). System prompt đóng vai trò làm bộ lọc định hình (persona filter), điều chỉnh trực tiếp tone giọng, độ sâu tri thức và tập từ vựng mà mô hình truy xuất từ cơ sở dữ liệu.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Với đoạn văn tiếng Việt 100 từ, ước lượng số từ / 0.75 ra khoảng 133 token, trong khi count_tokens (tiktoken BPE) đếm được thực tế khoảng 180 - 210 token (chênh lệch lớn khoảng 35% - 50%). Tiếng Việt tốn nhiều token hơn tiếng Anh vì các bộ mã hóa (tokenizer) của OpenAI chủ yếu tối ưu cho tiếng Anh; các từ tiếng Việt đa âm tiết và có dấu phụ thanh điệu thường bị tách thành nhiều sub-word hoặc byte-level token riêng lẻ.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng nhất trong các ứng dụng tương tác thời gian thực như Chatbot, nơi câu trả lời dài khiến Time-To-First-Token (TTFT) lớn dễ làm người dùng sốt ruột và nghĩ rằng ứng dụng bị treo. Ngược lại, Non-streaming phù hợp hơn khi xử lý ngầm (background jobs), gọi API lấy dữ liệu JSON structured output, hoặc thực hiện các tác vụ phân tích batch cần nhận đủ toàn bộ response mới xử lý tiếp.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff giúp phân tán khoảng thời gian thử lại của các client, cho phép hệ thống server có đủ thời gian "thở" và tự phục hồi sau sự cố quá tải. Nếu hàng nghìn client cùng retry với delay cố định giống nhau (ví dụ 1 giây), chúng sẽ đồng loạt tạo ra các cơn sóng request trùng thời điểm (Thundering Herd Problem), tiếp tục làm sập hệ thống và khiến server không bao giờ gượng dậy được.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Trả lời ngắn gọn, súc tích trong dưới 3 câu: Giúp kiểm soát chi phí token đầu ra và giữ giao diện chat gọn gàng, phù hợp trải nghiệm người dùng di động. Ưu tiên phản hồi bằng tiếng Việt: Định hướng ngôn ngữ cố định cho mô hình ngay cả khi người dùng vô tình nhập thuật ngữ tiếng Anh lai tiếng Việt.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất: Lịch sử hội thoại bị giới hạn cứng 3 lượt cuối (history[-6:]), dẫn đến việc mô hình sẽ quên sạch ngữ cảnh hoặc các yêu cầu ban đầu của người dùng nếu cuộc trò chuyện kéo dài. Đề xuất cải thiện: Triển khai cơ chế Summarization Memory.Mô tả cách thực hiện: Khi độ dài history vượt quá 6 messages, gọi một hàm chạy ngầm dùng gpt-4o-mini để tóm tắt 4 message cũ nhất thành một đoạn summary ngắn, sau đó lưu summary này vào System Prompt và chỉ giữ lại 2 message mới nhất trong history.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
