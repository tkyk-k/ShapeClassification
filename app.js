const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
let drawing = false;

// 初期化：白背景
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('mousemove', (e) => {
  if (!drawing) return;
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 2;
  ctx.stroke();
});

canvas.addEventListener('mouseup', () => {
  drawing = false;
});

canvas.addEventListener('mouseout', () => {
  drawing = false;
});

// クリアボタン
document.getElementById('clearBtn').addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
});

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

document.getElementById('predictBtn').addEventListener('click', async() => {
  // キャンバスの画像を取得
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  // 画像 → グレースケールに変換（例）
  const grayData = new Float32Array(canvas.width * canvas.height);
  for (let i = 0; i < canvas.width * canvas.height; i++) {
    const r = imageData.data[i * 4 + 0];
    const g = imageData.data[i * 4 + 1];
    const b = imageData.data[i * 4 + 2];
    // 白黒画像なのでR=G=B前提、平均でもよい
    grayData[i] = (r + g + b) / (3 * 255); // 正規化（0〜1）
  }

  // 例: 入力が [1, 1, 512, 512] の形状を期待する場合
  const tensor = new ort.Tensor("float32", grayData, [1, 1, 512, 512]);

  // モデル読み込み（1回だけにしたい場合は外に出す）
  const session = await ort.InferenceSession.create('trained_model.onnx');

  // 推論
  const feeds = { input: tensor }; // 'input' はモデルの入力名に合わせて
  const results = await session.run(feeds);

  // 出力確認
  const output = results[Object.keys(results)[0]];

  const S = 16;
  const B = 2;
  const num_classes = 3;
  const cellSize = canvas.width / S;

  const reshaped = [];
  const data = output.data;
for (let y = 0; y < S; y++) {
  reshaped[y] = [];
  for (let x = 0; x < S; x++) {
    reshaped[y][x] = [];
    for (let b = 0; b < B; b++) {
      const offset = ((y * S + x) * B + b) * (1 + 4 + num_classes + 1);
      const box = data.slice(offset, offset + 9);
      reshaped[y][x][b] = box;
    }
  }
}

const threshold = 0.1;
const boxes = [];

for (let y = 0; y < S; y++) {
  for (let x = 0; x < S; x++) {
    for (let b = 0; b < B; b++) {
      const box = reshaped[y][x][b];
      const rawObjectness = box[0];
      const objectness = sigmoid(rawObjectness);  // ← ここでSigmoidをかける
    
      if (objectness > threshold) {
        const [_, cx, cy, w, h, cls0, cls1, cls2, area] = box;
        const classScores = [cls0, cls1, cls2];
        const classId = classScores.indexOf(Math.max(...classScores));
        const label = ["円", "三角形", "四角形"][classId];

        const abs_cx = (x + cx) * cellSize;
        const abs_cy = (y + cy) * cellSize;
        const abs_w = w * canvas.width;
        const abs_h = h * canvas.height;

        boxes.push({
          label,
          classId,
          confidence: objectness.toFixed(2),  // Sigmoid後の値を confidence に
          cx: abs_cx,
          cy: abs_cy,
          w: abs_w,
          h: abs_h,
          area: area.toFixed(1)
        });
      }
    }
  }
}

console.log("抽出されたボックス:", boxes);
const nmsBoxes = applyNMS(boxes);
console.log("NMS後:", nmsBoxes);
drawBoxesOnCanvas(nmsBoxes);
});

// IoU計算（2つの矩形の重なり率）
function computeIoU(box1, box2) {
  const x1 = Math.max(box1.cx - box1.w / 2, box2.cx - box2.w / 2);
  const y1 = Math.max(box1.cy - box1.h / 2, box2.cy - box2.h / 2);
  const x2 = Math.min(box1.cx + box1.w / 2, box2.cx + box2.w / 2);
  const y2 = Math.min(box1.cy + box1.h / 2, box2.cy + box2.h / 2);

  const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const box1Area = box1.w * box1.h;
  const box2Area = box2.w * box2.h;

  const unionArea = box1Area + box2Area - interArea;
  return interArea / unionArea;
}

// NMS（重複ボックスを除去）
function applyNMS(boxes, iouThreshold = 0.4) {
  const result = [];
  const boxesByClass = {};

  // クラスごとに分ける
  for (const box of boxes) {
    if (!boxesByClass[box.classId]) {
      boxesByClass[box.classId] = [];
    }
    boxesByClass[box.classId].push(box);
  }

  for (const classId in boxesByClass) {
    const classBoxes = boxesByClass[classId];

    // 信頼度でソート
    classBoxes.sort((a, b) => parseFloat(b.confidence) - parseFloat(a.confidence));

    const kept = [];

    while (classBoxes.length > 0) {
      const best = classBoxes.shift();
      kept.push(best);

      // IoUが高いものを除外
      for (let i = classBoxes.length - 1; i >= 0; i--) {
        if (computeIoU(best, classBoxes[i]) > iouThreshold) {
          classBoxes.splice(i, 1);
        }
      }
    }

    result.push(...kept);
  }

  return result;
}


function drawBoxesOnCanvas(boxes) {
  for (const box of boxes) {
    // 図形の種類に応じた色を決定
    let color = 'red';
    if (box.label === '円') color = 'lime';
    else if (box.label === '三角形') color = 'deepskyblue';
    else if (box.label === '四角形') color = 'orange';

    // 枠の描画
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(box.cx - box.w / 2, box.cy - box.h / 2, box.w, box.h);

    // ラベルの描画
    ctx.fillStyle = color;
    ctx.font = '16px sans-serif';
    const labelText = `${box.label} (${box.confidence})`;
    ctx.fillText(labelText, box.cx - box.w / 2, box.cy - box.h / 2 - 4);
  }
}
