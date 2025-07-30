import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepSeek Tutor")
        self.messages = []

        # Загрузка модели и токенизатора
        model_dir = "./deepseek-model"  # Updated path
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        # Генерация приветственного сообщения
        welcome_prompt = "Explain the Present Simple tense in English."
        self.messages.append({"role": "user", "content": welcome_prompt})
        response = self.generate_response()
        self.messages.append({"role": "assistant", "content": response})

        # Настройка графического интерфейса
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        # Область чата
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        layout.addWidget(self.chat_history)
        
        # Поле ввода и кнопка отправки
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        input_layout.addWidget(self.input_field)
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.on_send)
        input_layout.addWidget(send_button)
        layout.addLayout(input_layout)
        
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Отображение начального сообщения
        self.update_chat_history()

    def generate_response(self):
        """Генерирует ответ модели на основе списка сообщений."""
        input_tensor = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(
            input_tensor,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_tokens = outputs[0][input_tensor.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response

    def on_send(self):
        """Обрабатывает отправку пользовательского сообщения."""
        user_text = self.input_field.text().strip()
        if user_text:
            self.messages.append({"role": "user", "content": user_text})
            response = self.generate_response()
            self.messages.append({"role": "assistant", "content": response})
            self.update_chat_history()
            self.input_field.clear()

    def update_chat_history(self):
        """Обновляет область чата с историей сообщений."""
        text = ""
        for msg in self.messages:
            if msg["role"] == "user":
                text += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                text += f"Assistant: {msg['content']}\n\n"
        self.chat_history.setText(text)
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())