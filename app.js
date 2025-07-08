const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
let drawing = false;

// 黒背景で初期化
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// 描画イベント
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
canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mouseout', () => drawing = false);

// クリアボタン
document.getElementById('clearBtn').addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
});

// 活性化関数
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
function softmax(arr) {
  const maxVal = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

// 推論処理
document.getElementById('predictBtn').addEventListener('click', async () => {
  // 画像取得 → グレースケール変換
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const grayData = new Float32Array(canvas.width * canvas.height);
  for (let i = 0; i < grayData.length; i++) {
    const r = imageData.data[i * 4];
    const g = imageData.data[i * 4 + 1];
    const b = imageData.data[i * 4 + 2];
    grayData[i] = (r + g + b) / (3 * 255);
  }

  const tensor = new ort.Tensor("float32", grayData, [1, 1, 512, 512]);
  const session = await ort.InferenceSession.create('trained_model.onnx');
  const results = await session.run({ input: tensor });
  const output = results[Object.keys(results)[0]];

  const S = 16, B = 2, num_classes = 3;
  const cellSize = canvas.width / S;
  const data = output.data;
  const boxes = [];
  const threshold = 0.1;

  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      for (let b = 0; b < B; b++) {
        const offset = ((y * S + x) * B + b) * (1 + 4 + num_classes + 1);
        const box = data.slice(offset, offset + 9);
        const objectness = sigmoid(box[0]);

        if (objectness > threshold) {
          const [_, cx, cy, w, h, cls0, cls1, cls2, areaNorm] = box;
          const probs = softmax([cls0, cls1, cls2]);
          const classId = probs.indexOf(Math.max(...probs));
          const label = ["円", "三角形", "四角形"][classId];

          const abs_cx = (x + cx) * cellSize;
          const abs_cy = (y + cy) * cellSize;
          const abs_w = w * canvas.width;
          const abs_h = h * canvas.height;
          const abs_area = Math.round(areaNorm * canvas.width * canvas.height);

          boxes.push({
            label,
            classId,
            confidence: Math.round(objectness * 100),
            cx: abs_cx,
            cy: abs_cy,
            w: abs_w,
            h: abs_h,
            area: abs_area,
            classScores: probs.map(p => Math.round(p * 100))
          });
        }
      }
    }
  }

  const nmsBoxes = applyNMS(boxes);
  drawBoxesOnCanvas(nmsBoxes);
});

// IoU計算
function computeIoU(box1, box2) {
  const x1 = Math.max(box1.cx - box1.w / 2, box2.cx - box2.w / 2);
  const y1 = Math.max(box1.cy - box1.h / 2, box2.cy - box2.h / 2);
  const x2 = Math.min(box1.cx + box1.w / 2, box2.cx + box2.w / 2);
  const y2 = Math.min(box1.cy + box1.h / 2, box2.cy + box2.h / 2);
  const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const unionArea = box1.w * box1.h + box2.w * box2.h - interArea;
  return interArea / unionArea;
}

// NMS
function applyNMS(boxes, iouThreshold = 0.4) {
  const result = [];
  const boxesByClass = {};

  for (const box of boxes) {
    if (!boxesByClass[box.classId]) boxesByClass[box.classId] = [];
    boxesByClass[box.classId].push(box);
  }

  for (const classId in boxesByClass) {
    const classBoxes = boxesByClass[classId];
    classBoxes.sort((a, b) => b.confidence - a.confidence);
    while (classBoxes.length > 0) {
      const best = classBoxes.shift();
      result.push(best);
      for (let i = classBoxes.length - 1; i >= 0; i--) {
        if (computeIoU(best, classBoxes[i]) > iouThreshold) {
          classBoxes.splice(i, 1);
        }
      }
    }
  }

  return result;
}

// 描画
function drawBoxesOnCanvas(boxes) {
  for (const box of boxes) {
    let color = 'red';
    if (box.label === '円') color = 'lime';
    else if (box.label === '三角形') color = 'deepskyblue';
    else if (box.label === '四角形') color = 'orange';

    // 枠描画
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(box.cx - box.w / 2, box.cy - box.h / 2, box.w, box.h);

    // ラベル描画
    ctx.fillStyle = color;
    ctx.font = '14px sans-serif';

    const texts = [
      `${box.label} (${box.confidence}%)`,
      `面積: ${box.area}px²`,
      `円:${box.classScores[0]}% 三角:${box.classScores[1]}% 四角:${box.classScores[2]}%`
    ];

    // ボックスの上だと隠れるので、下側 or 中央へ描画
    const startY = Math.min(canvas.height - 5, box.cy + box.h / 2 + 16);

    texts.forEach((t, i) => {
      ctx.fillText(t, box.cx - box.w / 2, startY + i * 16);
    });
  }
}
