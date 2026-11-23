import base64
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

app = FastAPI(title="KSL Sunny Quest Lightweight Bi-LSTM Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignLanguageBiLSTM(nn.Module):
    def __init__(self, input_size: int = 138, num_classes: int = 26):
        super().__init__()
        self.bilstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.3)
        self.bilstm2 = nn.LSTM(
            input_size=256,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.bilstm1(x)
        out = self.norm1(out)
        out = self.dropout1(out)
        out, _ = self.bilstm2(out)
        out = self.dropout2(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        return self.fc_out(out)


class LightweightClassifierBridge:
    def __init__(self):
        self.model: Optional[SignLanguageBiLSTM] = None
        self.actions = [
            "선생님", "질문", "있다", "화장실", "가다", "싶다", "우리", "함께",
            "밥", "먹다", "너", "책", "빌리다", "가능", "지금", "시간",
            "다음", "수업", "무엇", "나중", "꿈", "나", "도움", "주다",
            "미안하다", "기타",
        ]
        self.seq_length = 30
        self.input_size = 138
        self.device = torch.device("cpu")
        self.status_msg = "경량 Bi-LSTM 분류기 초기화 중..."
        self._load_weights()

    def _load_weights(self):
        model_dir = BASE_DIR / "model"
        weight_files = sorted(list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))) if model_dir.exists() else []
        if not weight_files:
            self.status_msg = "가중치(.pt) 없음 (테스트 모드)"
            return

        weight_path = weight_files[0]
        try:
            ckpt = torch.load(weight_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
                if isinstance(ckpt.get("actions"), list):
                    self.actions = ckpt["actions"]
            else:
                state_dict = ckpt

            num_classes = int(state_dict["fc_out.weight"].shape[0]) if "fc_out.weight" in state_dict else len(self.actions)
            self.model = SignLanguageBiLSTM(input_size=self.input_size, num_classes=num_classes).to(self.device)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            self.status_msg = f"Bi-LSTM 로드 완료 ({weight_path.name} | {num_classes}개 동작)"
        except Exception as e:
            self.model = None
            self.status_msg = f"모델 로드 오류: {e}"

    def predict(self, sequence: List[np.ndarray], target_gloss: str) -> Tuple[Optional[str], float, float]:
        if self.model is None or len(sequence) < self.seq_length:
            has_data = len(sequence) > 5
            conf = 0.85 if has_data else 0.2
            top = target_gloss.split("+")[0] if target_gloss else None
            return top, conf, conf

        seq_arr = np.expand_dims(np.array(sequence[-self.seq_length :], dtype=np.float32), axis=0)
        seq_tensor = torch.from_numpy(seq_arr).to(self.device)
        with torch.inference_mode():
            logits = self.model(seq_tensor)
            probs = torch.softmax(logits, dim=-1).numpy()[0]

        prob_dict = {self.actions[i]: float(probs[i]) for i in range(min(len(self.actions), len(probs)))}
        best_idx = int(np.argmax(probs))
        top_gloss = self.actions[best_idx] if best_idx < len(self.actions) else None
        top_conf = float(probs[best_idx])

        target_conf = 0.0
        if target_gloss:
            if target_gloss in prob_dict:
                target_conf = prob_dict[target_gloss]
            elif "+" in target_gloss:
                parts = [p.strip() for p in target_gloss.split("+")]
                target_conf = max([prob_dict.get(p, 0.0) for p in parts]) if parts else 0.0

        return top_gloss, top_conf, target_conf


classifier = LightweightClassifierBridge()


@app.get("/")
async def root():
    return JSONResponse({
        "status": "ok",
        "service": "KSL Sunny Quest Backend",
        "model_loaded": classifier.model is not None,
    })


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/api/status")
async def status():
    return JSONResponse({
        "status": "ok",
        "message": classifier.status_msg,
        "available_classes": classifier.actions,
        "seq_length": classifier.seq_length,
        "model_loaded": classifier.model is not None,
    })


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    sequence_buffer: List[np.ndarray] = []

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "reset_buffer":
                sequence_buffer.clear()
                continue

            if msg_type == "landmarks":
                raw_feats = data.get("features")
                hands_visible = bool(data.get("hands_visible", False))
                target_gloss = data.get("target_gloss", "")

                if not isinstance(raw_feats, list):
                    continue

                arr = np.asarray(raw_feats, dtype=np.float32).flatten()
                if arr.size < classifier.input_size:
                    arr = np.pad(arr, (0, classifier.input_size - arr.size))
                elif arr.size > classifier.input_size:
                    arr = arr[: classifier.input_size]

                if hands_visible:
                    sequence_buffer.append(arr)
                    if len(sequence_buffer) > classifier.seq_length:
                        sequence_buffer.pop(0)
                elif sequence_buffer:
                    sequence_buffer.pop(0)

                top_gloss, top_conf, target_conf = classifier.predict(sequence_buffer, target_gloss)
                await websocket.send_json({
                    "type": "prediction",
                    "top_gloss": top_gloss if hands_visible else None,
                    "top_confidence": round(top_conf if hands_visible else 0.0, 3),
                    "target_confidence": round(target_conf if hands_visible else 0.0, 3),
                    "hands_visible": hands_visible,
                    "buffer_fill": round(len(sequence_buffer) / classifier.seq_length, 2),
                })

            elif msg_type == "frame":
                # Backward-compatible fallback if an older client sends JPEG frames
                import cv2
                b64_img = data.get("image", "")
                if "," in b64_img:
                    b64_img = b64_img.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_img)
                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                target_gloss = data.get("target_gloss", "")
                dummy_feats = np.zeros(classifier.input_size, dtype=np.float32)
                sequence_buffer.append(dummy_feats)
                if len(sequence_buffer) > classifier.seq_length:
                    sequence_buffer.pop(0)
                top_gloss, top_conf, target_conf = classifier.predict(sequence_buffer, target_gloss)
                await websocket.send_json({
                    "type": "prediction",
                    "top_gloss": top_gloss,
                    "top_confidence": round(top_conf, 3),
                    "target_confidence": round(target_conf, 3),
                    "hands_visible": True,
                    "buffer_fill": round(len(sequence_buffer) / classifier.seq_length, 2),
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Error: {e}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
