import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PyQt5 import QtCore, QtWidgets

# -----------------------------
# 1) Блок загрузки модели
# -----------------------------
MODEL_DIR = "deepseek-model"

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

# Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем модель на это устройство
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
)
model.to(device)
model.eval()

def generate_deepseek(prompt: str,
                      max_new_tokens: int = 150,
                      temperature: float = 0.7) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Отрезаем исходный prompt из начала
    return full[len(prompt):].strip()

# -----------------------------
# 2) GUI на PyQt5
# -----------------------------
class SciFiChatWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("DeepSeek Tutor")
        self.resize(800, 600)
        self.setStyleSheet("""
            QWidget { background-color: #0b0e11; color: #00ffff; font-family: 'Consolas'; }
            QTextEdit { background: #081014; border: 2px solid #00ffff; border-radius: 5px; }
            QLineEdit { background: #081014; border: 2px solid #00ffff; border-radius: 3px; padding: 4px; }
            QPushButton { background: #00ffff; color: #0b0e11; padding: 6px 12px; border-radius: 3px; }
            QPushButton:hover { background: #33ffff; }
        """)

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("DEEPSEEK AI TUTOR")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.chat_area = QtWidgets.QTextEdit()
        self.chat_area.setReadOnly(True)
        layout.addWidget(self.chat_area, 1)

        hbox = QtWidgets.QHBoxLayout()
        self.input_line = QtWidgets.QLineEdit()
        self.input_line.setPlaceholderText("Напишите сообщение...")
        hbox.addWidget(self.input_line, 1)
        send_btn = QtWidgets.QPushButton("Send")
        send_btn.clicked.connect(self.on_send)
        hbox.addWidget(send_btn)
        layout.addLayout(hbox)

        # Первое сообщение от бота
        welcome = (
            "<b>DeepSeek:</b> Привет! Я ваша локальная модель DeepSeek — "
            "ваш AI-репетитор по 12 английским временам. "
            "Начнём с <i>Present Simple</i>:<br>"
            "• Регулярные действия и общие истины.<br>"
            "• Пример: “<b>I eat breakfast every day.</b>”<br><br>"
            "Задавайте свои предложения — я помогу их исправить!"
        )
        self.chat_area.append(welcome)

    def on_send(self):
        text = self.input_line.text().strip()
        if not text:
            return
        self.chat_area.append(f"<b>You:</b> {text}")
        self.input_line.clear()

        try:
            reply = generate_deepseek(text)
        except Exception as e:
            reply = f"[Ошибка генерации: {e}]"
        self.chat_area.append(f"<b>DeepSeek:</b> {reply}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = SciFiChatWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
