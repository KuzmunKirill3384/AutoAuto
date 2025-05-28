# 🤖 4‑Wheel Robot: от ручного управления до автономии


> **Коротко:** этот репозиторий собирает воедино **три** скрипта, которые превращают обычное шасси 4×4 на Raspberry Pi в маленького обучаемого робота.
>
> 1. `main.py` — ручной режим с геймпадом и автоматическим логированием изображений;
> 2. `train_robot_model.py` — обучение MobileNetV3‑Small классификатора действий;
> 3. `run_robot_model.py` — автономное следование модели на борту Pi.


---

## 📂 Содержание репозитория

| Файл/директория        | Что внутри                                | Зачем нужно                                                                  |
| ---------------------- | ----------------------------------------- | ---------------------------------------------------------------------------- |
| `main.py`              | Джойстик → робот; делает **фото‑датасет** | Собираем примеры трёх действий *(`forward`, `rotate_left`, `rotate_right`)*  |
| `train_robot_model.py` | PyTorch‑скрипт для Desktop/GPU            | Обучение CNN + экспорт `robot_action_cnn.pth` и `robot_action_cnn_script.pt` |
| `run_robot_model.py`   | Инференс на Raspberry Pi 4 / Pi Zero 2 W  | Делает предсказание > задаёт ШИМ ногам                                       |
| `dataset/`             | JPEG‑файлы вида `forward_001.jpg`…        | Сырые данные для обучения                                                    |
| `models/`              | Готовые веса сети (генерируется)          | Берётся `run_robot_model.py`                                                 |

*(Таблица минимальна — только чтобы быстро сориентироваться.)*

---

## 🛠️ Аппаратные требования

* **Шасси:** любое 4‑колёсное с двумя Н‑мостами (L298N) и ШИМ‑пинами.
* **Одноплатник:** Raspberry Pi 4 B ⬆ или Pi Zero 2 W (≈ 25 FPS в инференсе).
* **Камера:** Picamera2 или CSI‑совместимая; fallback — USB+OpenCV.
* **Геймпад:** любой, который видит `pygame` (X‑Box, DualShock и т.д.).

> **GPIO‑пины** зашиты в коде (`main.py`, `run_robot_model.py`).
> Если у вас другая распайка — поправьте константы `FL_IN1 … RR_EN`.

---

## 💾 Установка зависимостей

### На Raspberry Pi (сбор датасета **и** автономия)

```bash
sudo apt update && sudo apt install python3-pip python3-pygame python3-picamera2
pip3 install torch torchvision pillow opencv-python RPi.GPIO tqdm
```

### На рабочем столе (обучение)

```bash
conda create -n robot python=3.10
conda activate robot
pip install torch torchvision pillow tqdm
```

---

## 🚀 Быстрый старт

1. **Соберите датасет**

   ```bash
   # на Raspberry Pi (приставьте роботу трассу, возьмите геймпад)
   python3 main.py --out dataset
   # 📸 Скрипт сохраняет ~15 кадров/сек только когда вы жмёте стики
   ```
2. **Перекиньте снимки** на PC/Mac и **обучите модель**

   ```bash
   python3 train_robot_model.py --data ~/Downloads/pi  # путь к папке с JPEG
   # ↳ получим robot_action_cnn.pth (лучшие веса) и robot_action_cnn_script.pt
   ```
3. **Загрузите веса на Pi** и запустите автономию

   ```bash
   scp robot_action_cnn.pth pi@raspberrypi.local:~/robot
   ssh pi@raspberrypi.local
   python3 run_robot_model.py
   ```

---

## 🏗️ Под капотом

### main.py — ручное управление + логирование

* Читает оси геймпада через `pygame` (мертвую зону 0.20).
* Делает 640×480 снимки; CSV не создаёт (только JPEG).
* Кнопка **`X`** — старт/стоп записи, **`Start`** — выход.

### train\_robot\_model.py — MobileNetV3‑Small

* Аугментация: `RandomHorizontalFlip`, `ColorJitter`, `BottomMask` (30‑45 %).
* Игнорирует классы `stop`, `backward` (шум).
* Сохраняет лучший чек‑пойнт по валидации + экспортирует TorchScript.

### run\_robot\_model.py — инференс 50 FPS

* Препроцесс наследует `BottomMask` и `Normalize` как при обучении.
* Управляет GPIO: `BASE_SPEED = 30 %`, `FAST_SPEED = 45 %`.
* Всего три действия ⇒ сверхбыстрая модель (≈ 5 ms на Pi 4).

---

## 📁 Структура проекта после первого запуска

```
robot/
├── dataset/
│   ├── forward_0001.jpg
│   ├── rotate_left_0042.jpg
│   └── ...
├── models/
│   ├── robot_action_cnn.pth
│   └── robot_action_cnn_script.pt
├── main.py
├── train_robot_model.py
└── run_robot_model.py
```

---

## ❓ FAQ

**Q:** Почему вы затемняете нижнюю часть кадра?
**A:** Колёса и пол под роботом дают сильный bias; маска улучшает обобщение.

**Q:** Хочу добавить класс `backward` — что менять?
**A:** Добавьте метку в `ACTIONS` (оба скрипта) и переобучите модель.

**Q:** Нужно ли GPU на Pi?
**A:** Нет — CPU Cortex‑A72 справляется; GPU не используется.

---

## ⚖️ Лицензия

MIT © 2025 — делайте форк, улучшайте и присылайте PR!

---

## 🙌 Благодарности

* [PyTorch](https://pytorch.org/) за быстрый JIT.
* [Picamera2](https://github.com/raspberrypi/picamera2) за удобный API.
* Сообщество DIY‑роботов за вдохновение.
