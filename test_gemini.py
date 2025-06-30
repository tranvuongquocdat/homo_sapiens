import google.generativeai as genai
import time
import os
from typing import Optional

class GeminiSpellChecker:
    def __init__(self, api_key: Optional[str] = None):
        """
        Khởi tạo Gemini API
        Args:
            api_key: API key cho Gemini. Nếu None, sẽ lấy từ biến môi trường GEMINI_API_KEY
        """
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key is None:
                raise ValueError("Vui lòng cung cấp API key hoặc set biến môi trường GEMINI_API_KEY")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Prompt template từ Flutter app
        self.prompt_template = """Bạn là một bot kiểm tra chính tả và ngữ pháp tiếng Việt. Nhiệm vụ của bạn là chỉnh sửa văn bản có lỗi sai do lặp chữ, sai chính tả hoặc cấu trúc câu chưa chính xác. Sau đây là một số ví dụ để tham khảo:
"aaaaxxaaaaannnnnccccccjjcccccoooooookkmmmmmmmm" → "ăn cơm"
"ddddxdooooccccccnnnlnnnaaayyyyyy" → "đọc này"
"xxxxxxjjjiiiiiiinnnnnnnnfkdcccccccchhhhhhaaaaaooooo" → "xin chào"
"ddddddjaiiiiiiiiiaaaaaaanmaaaaaannnnnnn" → "đi ăn"

Chỉ xuất ra câu đã được chỉnh sửa, đảm bảo chính tả, ngữ pháp và dấu câu chuẩn xác.
Ví dụ: "Ttttrrrooooiioiimmmmmuuuuujfuuuuaaaaa," sẽ trở thành "Trời mưa!"

Bây giờ, hãy áp dụng quy tắc này để chỉnh sửa văn bản sau: {text}"""

    def correct_text(self, text: str) -> tuple[str, float]:
        """
        Chỉnh sửa văn bản bằng Gemini API
        Args:
            text: Văn bản cần chỉnh sửa
        Returns:
            tuple: (kết quả chỉnh sửa, thời gian phản hồi)
        """
        prompt = self.prompt_template.format(text=text)
        
        # Đo thời gian bắt đầu
        start_time = time.time()
        
        try:
            # Gọi API
            response = self.model.generate_content(prompt)
            
            # Đo thời gian kết thúc
            end_time = time.time()
            response_time = end_time - start_time
            
            return response.text.strip(), response_time
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            return f"Lỗi: {str(e)}", response_time

    def test_multiple_cases(self, test_cases: list[str]) -> None:
        """
        Test nhiều trường hợp và hiển thị kết quả
        """
        print("=" * 80)
        print("🧪 GEMINI API SPEED TEST - KIỂM TRA CHÍNH TẢ TIẾNG VIỆT")
        print("=" * 80)
        print(f"Model: {self.model.model_name}")
        print("-" * 80)
        
        total_time = 0
        successful_requests = 0
        
        for i, test_text in enumerate(test_cases, 1):
            print(f"\n📝 Test Case {i}:")
            print(f"Input:  '{test_text}'")
            
            result, response_time = self.correct_text(test_text)
            total_time += response_time
            
            if not result.startswith("Lỗi:"):
                successful_requests += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"Output: '{result}' {status}")
            print(f"⏱️  Response Time: {response_time:.3f}s")
            print("-" * 40)
        
        # Thống kê tổng kết
        print(f"\n📊 THỐNG KÊ:")
        print(f"Tổng số test cases: {len(test_cases)}")
        print(f"Thành công: {successful_requests}")
        print(f"Thất bại: {len(test_cases) - successful_requests}")
        print(f"Tổng thời gian: {total_time:.3f}s")
        print(f"Thời gian trung bình: {total_time/len(test_cases):.3f}s/request")
        
        if successful_requests > 0:
            success_rate = (successful_requests / len(test_cases)) * 100
            print(f"Tỷ lệ thành công: {success_rate:.1f}%")


def main():
    """
    Hàm main để chạy test
    """
    # Khởi tạo spell checker
    try:
        # Thay đổi API_KEY ở đây hoặc set biến môi trường GEMINI_API_KEY
        API_KEY = "AIzaSyBa0RIPexzkRmQU-hXizc54O0yypUyJCF8"
        
        spell_checker = GeminiSpellChecker(api_key=API_KEY)
        
        # Các test cases mẫu
        test_cases = [
            "xxxxxiiiinnnnnlllefglllllllooooooiiii",
            "ddddxdooooccccccnnnlnnnaaayyyyyy", 
            "aaaaaaaffnnnnnnnggggggoooooefoooooonnnnn",
            "xxxxxxiiiiinnnnncccchaaaaaoooo"
        ]
        
        # Chạy test
        spell_checker.test_multiple_cases(test_cases)
        
        print("\n" + "=" * 80)
        print("✨ Test hoàn thành! Kiểm tra kết quả ở trên.")
        print("=" * 80)
        
    except ValueError as e:
        print(f"❌ Lỗi cấu hình: {e}")
        print("\n💡 Hướng dẫn:")
        print("1. Lấy API key từ https://aistudio.google.com/app/apikey")
        print("2. Set biến môi trường: export GEMINI_API_KEY='your_api_key'")
        print("3. Hoặc thay đổi biến API_KEY trong code")
        
    except Exception as e:
        print(f"❌ Lỗi không mong đợi: {e}")


if __name__ == "__main__":
    main()
