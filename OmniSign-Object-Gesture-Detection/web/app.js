/**
 * OmniSign Live - Neural Object & Gesture Detection Client
 */

// State
let isStreaming = false;
let videoStream = null;
let ws = null;
let streamInterval = null;
let httpLoopActive = false;
let isHttpProcessing = false;
let lastFrameTime = performance.now();
let fps = 0;
let confidenceThreshold = 0.30;

// Hand skeleton connections for 21 MediaPipe landmarks
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [9, 13], [13, 14], [14, 15], [15, 16],// Ring
  [13, 17], [17, 18], [18, 19], [19, 20],// Pinky
  [0, 17]                               // Palm base
];

// DOM Elements
const video = document.getElementById("webcamVideo");
const canvas = document.getElementById("detectionCanvas");
const ctx = canvas.getContext("2d");
const toggleCameraBtn = document.getElementById("toggleCameraBtn");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdValue = document.getElementById("thresholdValue");
const latencyValue = document.getElementById("latencyValue");
const fpsValue = document.getElementById("fpsValue");
const heroLabel = document.getElementById("heroLabel");
const heroConfidence = document.getElementById("heroConfidence");
const heroMeterFill = document.getElementById("heroMeterFill");
const detectionList = document.getElementById("detectionList");
const liveDetectionCount = document.getElementById("liveDetectionCount");
const liveResolution = document.getElementById("liveResolution");
const connectionStatus = document.getElementById("connectionStatus");

// Offscreen capture canvas
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

// Tabs
const tabButtons = document.querySelectorAll(".tab-btn");
const tabPanes = document.querySelectorAll(".tab-pane");

tabButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    tabButtons.forEach(b => b.classList.remove("active"));
    tabPanes.forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    const targetId = btn.getAttribute("data-tab");
    document.getElementById(targetId).classList.add("active");

    if (targetId === "samples-tab") {
      loadDatasetGallery();
    }
  });
});

// Slider Handler
thresholdSlider.value = 30;
thresholdValue.textContent = "30%";
thresholdSlider.addEventListener("input", (e) => {
  confidenceThreshold = parseInt(e.target.value) / 100;
  thresholdValue.textContent = `${e.target.value}%`;
});

// Toggle Camera
toggleCameraBtn.addEventListener("click", async () => {
  if (!isStreaming) {
    await startCamera();
  } else {
    stopCamera();
  }
});

async function startCamera() {
  try {
    const constraints = {
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
    };
    videoStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = videoStream;
    video.style.display = "block";
    videoPlaceholder.style.display = "none";

    await new Promise(resolve => {
      video.onloadedmetadata = () => {
        video.play();
        resolve();
      };
    });

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    liveResolution.textContent = `${canvas.width}x${canvas.height}`;

    isStreaming = true;
    toggleCameraBtn.classList.add("stop");
    toggleCameraBtn.innerHTML = `
      <svg class="btn-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12"/></svg>
      <span>Stop Camera</span>
    `;

    initStream();
  } catch (err) {
    console.error("Camera access error:", err);
    alert("Could not access camera: " + err.message);
  }
}

function stopCamera() {
  if (videoStream) {
    videoStream.getTracks().forEach(track => track.stop());
    videoStream = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
  if (streamInterval) {
    clearInterval(streamInterval);
    streamInterval = null;
  }
  httpLoopActive = false;

  isStreaming = false;
  video.style.display = "none";
  videoPlaceholder.style.display = "flex";
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  toggleCameraBtn.classList.remove("stop");
  toggleCameraBtn.innerHTML = `
    <svg class="btn-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
    <span>Start Camera</span>
  `;

  resetTelemetry();
}

function initStream() {
  // Try WebSocket first
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("[WebSocket] Connected successfully");
      connectionStatus.textContent = "Live WS Connected";

      streamInterval = setInterval(() => {
        if (!isStreaming || !video.videoWidth || ws.readyState !== WebSocket.OPEN) return;

        captureCanvas.width = 320;
        captureCanvas.height = Math.round(320 * (video.videoHeight / video.videoWidth));
        captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

        const frameBase64 = captureCanvas.toDataURL("image/jpeg", 0.65);
        ws.send(JSON.stringify({
          frame: frameBase64,
          threshold: confidenceThreshold
        }));
      }, 50);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleInferenceResult(data);
      updateFPS();
    };

    ws.onerror = (err) => {
      console.warn("[WebSocket] WS error, switching to HTTP loop fallback", err);
      startHttpLoop();
    };

    ws.onclose = () => {
      console.log("[WebSocket] Closed");
      if (isStreaming && !httpLoopActive) {
        startHttpLoop();
      }
    };
  } catch (err) {
    console.warn("[WebSocket] Init failed, using HTTP fallback", err);
    startHttpLoop();
  }
}

function startHttpLoop() {
  if (httpLoopActive) return;
  httpLoopActive = true;
  connectionStatus.textContent = "Live Stream Active";
  console.log("[Stream] Running on real-time stream engine");

  const loop = async () => {
    if (!isStreaming || !httpLoopActive) return;

    if (!isHttpProcessing && video.videoWidth > 0) {
      isHttpProcessing = true;
      captureCanvas.width = 320;
      captureCanvas.height = Math.round(320 * (video.videoHeight / video.videoWidth));
      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

      const frameBase64 = captureCanvas.toDataURL("image/jpeg", 0.6);

      try {
        const res = await fetch("/api/detect", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            image_base64: frameBase64,
            threshold: confidenceThreshold
          })
        });
        const data = await res.json();
        handleInferenceResult(data);
        updateFPS();
      } catch (err) {
        console.error("[HTTP Stream error]", err);
      } finally {
        isHttpProcessing = false;
      }
    }

    if (isStreaming) {
      setTimeout(loop, 45);
    }
  };

  loop();
}

function updateFPS() {
  const now = performance.now();
  const delta = (now - lastFrameTime) / 1000;
  lastFrameTime = now;
  fps = Math.round(1 / (delta || 0.05));
  fpsValue.textContent = fps;
}

function handleInferenceResult(data) {
  if (!data) return;

  if (data.inference_ms !== undefined) {
    latencyValue.textContent = `${data.inference_ms} ms`;
  }

  const detections = data.detections || [];
  const landmarks = data.landmarks || [];
  liveDetectionCount.textContent = detections.length;

  // Clear HUD Canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 1. Draw Hand Skeleton Landmarks if detected
  if (landmarks.length > 0) {
    landmarks.forEach(handLms => {
      drawHandSkeleton(ctx, handLms, canvas.width, canvas.height);
    });
  }

  // 2. Draw Bounding Boxes and Update Hero
  if (detections.length > 0) {
    const topDet = detections[0];
    heroLabel.textContent = topDet.label;
    heroConfidence.textContent = `${topDet.confidence}%`;
    heroMeterFill.style.width = `${topDet.confidence}%`;

    detections.forEach((det, idx) => {
      drawBoundingBox(ctx, det, canvas.width, canvas.height, idx === 0);
    });

    renderDetectionList(detections);
  } else {
    heroLabel.textContent = "Searching...";
    heroConfidence.textContent = "0%";
    heroMeterFill.style.width = "0%";
    detectionList.innerHTML = `<div class="empty-notice">Hold sign/hand clearly in view</div>`;
  }

  // 3. Update Classifier Breakdown
  if (data.classification && data.classification.all_scores) {
    const scores = data.classification.all_scores;
    for (const [clsName, conf] of Object.entries(scores)) {
      const fillEl = document.getElementById(`clf-${clsName}`);
      const valEl = document.getElementById(`clf-val-${clsName}`);
      if (fillEl && valEl) {
        fillEl.style.width = `${conf}%`;
        valEl.textContent = `${conf}%`;
      }
    }
  }
}

function drawHandSkeleton(context, landmarks, canvasW, canvasH) {
  context.save();
  context.lineWidth = 2;
  context.strokeStyle = "rgba(56, 189, 248, 0.7)";
  context.shadowColor = "#38bdf8";
  context.shadowBlur = 8;

  // Draw bone lines
  HAND_CONNECTIONS.forEach(([i, j]) => {
    const p1 = landmarks[i];
    const p2 = landmarks[j];
    if (p1 && p2) {
      context.beginPath();
      context.moveTo(p1.x * canvasW, p1.y * canvasH);
      context.lineTo(p2.x * canvasW, p2.y * canvasH);
      context.stroke();
    }
  });

  // Draw joint points
  landmarks.forEach((p, idx) => {
    const isTip = [4, 8, 12, 16, 20].includes(idx);
    context.fillStyle = isTip ? "#34d399" : "#38bdf8";
    context.beginPath();
    context.arc(p.x * canvasW, p.y * canvasH, isTip ? 4.5 : 3, 0, 2 * Math.PI);
    context.fill();
  });

  context.restore();
}

function drawBoundingBox(context, det, canvasW, canvasH, isPrimary) {
  const box = det.box;
  const x = box.xmin * canvasW;
  const y = box.ymin * canvasH;
  const w = (box.xmax - box.xmin) * canvasW;
  const h = (box.ymax - box.ymin) * canvasH;

  const color = isPrimary ? "#38bdf8" : "#818cf8";

  context.save();
  context.strokeStyle = color;
  context.lineWidth = isPrimary ? 3 : 2;
  context.shadowColor = color;
  context.shadowBlur = isPrimary ? 12 : 6;
  context.strokeRect(x, y, w, h);

  // Label tag banner
  const text = `${det.label} ${det.confidence}%`;
  context.font = "bold 13px 'JetBrains Mono', monospace";
  const textMetrics = context.measureText(text);
  const pad = 6;
  const bannerW = textMetrics.width + pad * 2;
  const bannerH = 22;

  context.fillStyle = isPrimary ? "rgba(2, 132, 199, 0.9)" : "rgba(79, 70, 229, 0.9)";
  context.fillRect(x, Math.max(0, y - bannerH), bannerW, bannerH);

  context.fillStyle = "#ffffff";
  context.fillText(text, x + pad, Math.max(0, y - bannerH) + 15);

  context.restore();
}

function renderDetectionList(detections) {
  detectionList.innerHTML = detections.map(det => `
    <div class="detection-row">
      <div class="det-label-group">
        <div class="det-dot" style="background: ${det.source === 'hand_tracker' ? '#10b981' : '#38bdf8'}"></div>
        <div>
          <div class="det-name">${det.label}</div>
          <div class="det-coords">[${Math.round(det.box.xmin * 100)}%, ${Math.round(det.box.ymin * 100)}%] (${det.source || 'tflite'})</div>
        </div>
      </div>
      <div class="det-badge">${det.confidence}%</div>
    </div>
  `).join("");
}

function resetTelemetry() {
  latencyValue.textContent = "-- ms";
  fpsValue.textContent = "--";
  heroLabel.textContent = "Awaiting Stream";
  heroConfidence.textContent = "0%";
  heroMeterFill.style.width = "0%";
  liveDetectionCount.textContent = "0";
  detectionList.innerHTML = `<div class="empty-notice">No active detections in current frame</div>`;
}

// ----------------------------------------------------------------------
// Image Upload Inspector
// ----------------------------------------------------------------------
const dropzone = document.getElementById("dropzone");
const imageFileInput = document.getElementById("imageFileInput");
const imagePreviewWrapper = document.getElementById("imagePreviewWrapper");
const staticImgPreview = document.getElementById("staticImgPreview");
const staticCanvas = document.getElementById("staticCanvas");
const staticCtx = staticCanvas.getContext("2d");
const staticResultsContainer = document.getElementById("staticResultsContainer");

dropzone.addEventListener("click", () => imageFileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "var(--primary)";
});

dropzone.addEventListener("dragleave", () => {
  dropzone.style.borderColor = "var(--border-accent)";
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "var(--border-accent)";
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleFileUpload(e.dataTransfer.files[0]);
  }
});

imageFileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFileUpload(e.target.files[0]);
  }
});

async function handleFileUpload(file) {
  const reader = new FileReader();
  reader.onload = async (e) => {
    const dataUrl = e.target.result;
    staticImgPreview.src = dataUrl;
    imagePreviewWrapper.style.display = "block";

    staticImgPreview.onload = async () => {
      staticCanvas.width = staticImgPreview.naturalWidth || staticImgPreview.width;
      staticCanvas.height = staticImgPreview.naturalHeight || staticImgPreview.height;

      staticResultsContainer.innerHTML = `<div class="empty-notice">Running neural inference...</div>`;

      const formData = new FormData();
      formData.append("file", file);
      formData.append("threshold", confidenceThreshold);

      try {
        const response = await fetch("/api/detect", {
          method: "POST",
          body: formData
        });
        const result = await response.json();
        renderStaticResults(result);
      } catch (err) {
        staticResultsContainer.innerHTML = `<div class="empty-notice" style="color: var(--red);">Inference Error: ${err.message}</div>`;
      }
    };
  };
  reader.readAsDataURL(file);
}

function renderStaticResults(result) {
  staticCtx.clearRect(0, 0, staticCanvas.width, staticCanvas.height);

  const landmarks = result.landmarks || [];
  if (landmarks.length > 0) {
    landmarks.forEach(handLms => {
      drawHandSkeleton(staticCtx, handLms, staticCanvas.width, staticCanvas.height);
    });
  }

  const detections = result.detections || [];
  if (detections.length === 0) {
    staticResultsContainer.innerHTML = `
      <div class="empty-notice">No objects/signs detected above ${(confidenceThreshold * 100)}% confidence threshold.</div>
    `;
    return;
  }

  detections.forEach((det, idx) => {
    drawBoundingBox(staticCtx, det, staticCanvas.width, staticCanvas.height, idx === 0);
  });

  let html = `
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--primary); margin-bottom: 0.75rem;">
      Found ${detections.length} target(s) in ${result.inference_ms}ms:
    </div>
    <div class="detection-items-container">
  `;

  detections.forEach(det => {
    html += `
      <div class="detection-row">
        <div class="det-label-group">
          <div class="det-dot" style="background: ${det.source === 'hand_tracker' ? '#10b981' : '#38bdf8'}"></div>
          <div>
            <div class="det-name">${det.label}</div>
            <div class="det-coords">[px: ${det.box.px_xmin}, ${det.box.px_ymin} &rarr; ${det.box.px_xmax}, ${det.box.px_ymax}]</div>
          </div>
        </div>
        <div class="det-badge">${det.confidence}%</div>
      </div>
    `;
  });
  html += `</div>`;

  if (result.classification) {
    html += `
      <div class="section-label" style="margin-top: 1rem;">CLASSIFIER PREDICTION</div>
      <div class="classifier-breakdown" style="margin-top: 0.5rem;">
        <div style="font-weight: 700; color: #34d399; font-size: 0.95rem;">
          Top Gesture: ${result.classification.top_label} (${result.classification.top_confidence}%)
        </div>
      </div>
    `;
  }

  staticResultsContainer.innerHTML = html;
}

// ----------------------------------------------------------------------
// Dataset Gallery & Custom Photo Handler
// ----------------------------------------------------------------------
let currentGalleryCat = "all";
let galleryDataCache = null;

async function loadDatasetGallery(forceRefresh = false) {
  const gallery = document.getElementById("sampleGallery");
  if (!forceRefresh && galleryDataCache) {
    renderGalleryItems(galleryDataCache);
    return;
  }

  try {
    gallery.innerHTML = `<div class="gallery-loading">Loading dataset samples...</div>`;
    const res = await fetch("/api/samples");
    const data = await res.json();
    galleryDataCache = data;

    // Update Category Counts Badges
    const counts = data.category_counts || {};
    const countIly = document.getElementById("count-ily");
    const countTy = document.getElementById("count-ty");
    const countYes = document.getElementById("count-yes");
    const countArch = document.getElementById("count-arch");

    if (countIly) countIly.textContent = counts["i love you"] || 0;
    if (countTy) countTy.textContent = counts["thank you"] || 0;
    if (countYes) countYes.textContent = counts["yes"] || 0;
    if (countArch) countArch.textContent = counts["compressed_b[1]"] || 0;

    renderGalleryItems(data);
  } catch (err) {
    gallery.innerHTML = `<div class="empty-notice" style="color: var(--red);">Error loading gallery: ${err.message}</div>`;
  }
}

function renderGalleryItems(data) {
  const gallery = document.getElementById("sampleGallery");
  let samples = data.samples || [];

  if (currentGalleryCat !== "all") {
    samples = samples.filter(s => s.category.toLowerCase() === currentGalleryCat.toLowerCase());
  }

  if (samples.length === 0) {
    gallery.innerHTML = `<div class="empty-notice">No photos found in category "${currentGalleryCat}".</div>`;
    return;
  }

  gallery.innerHTML = samples.map(sample => {
    const isCustomTag = sample.is_custom 
      ? `<span class="class-tag accent" style="font-size: 0.65rem; padding: 0.15rem 0.4rem; position: absolute; top: 8px; right: 8px; z-index: 2;">YOUR PHOTO</span>` 
      : "";
    return `
      <div class="gallery-card" style="position: relative;" onclick="testSampleImage('${sample.url}', '${sample.name}')">
        ${isCustomTag}
        <div class="gallery-thumb-wrap">
          <img src="${sample.url}" alt="${sample.name}" loading="lazy" />
        </div>
        <div class="gallery-info">
          <span class="gallery-name" title="${sample.name}">${sample.name}</span>
          <span class="gallery-cat" style="text-transform: capitalize; font-weight: 600; color: var(--primary);">${sample.category}</span>
        </div>
      </div>
    `;
  }).join("");
}

// Category filter chip event listeners
document.addEventListener("DOMContentLoaded", () => {
  const filterContainer = document.getElementById("galleryFilters");
  if (filterContainer) {
    filterContainer.addEventListener("click", (e) => {
      const chip = e.target.closest(".filter-chip");
      if (!chip) return;
      document.querySelectorAll(".filter-chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      currentGalleryCat = chip.dataset.cat;
      if (galleryDataCache) {
        renderGalleryItems(galleryDataCache);
      } else {
        loadDatasetGallery();
      }
    });
  }

  const refreshBtn = document.getElementById("refreshGalleryBtn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      loadDatasetGallery(true);
    });
  }

  const datasetUploadInput = document.getElementById("datasetUploadInput");
  if (datasetUploadInput) {
    datasetUploadInput.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const catToUpload = currentGalleryCat !== "all" && currentGalleryCat !== "compressed_b[1]" 
        ? currentGalleryCat 
        : "thank you";

      const formData = new FormData();
      formData.append("file", file);
      formData.append("category", catToUpload);

      try {
        const res = await fetch("/api/upload_sample", {
          method: "POST",
          body: formData
        });
        const result = await res.json();
        if (result.success) {
          await loadDatasetGallery(true);
          testSampleImage(result.url, result.filename);
        }
      } catch (err) {
        alert("Upload failed: " + err.message);
      }
      datasetUploadInput.value = "";
    });
  }
});

async function testSampleImage(sampleUrl, sampleName) {
  document.querySelector('[data-tab="upload-tab"]').click();
  staticImgPreview.src = sampleUrl;
  imagePreviewWrapper.style.display = "block";

  staticImgPreview.onload = async () => {
    staticCanvas.width = staticImgPreview.naturalWidth || staticImgPreview.width;
    staticCanvas.height = staticImgPreview.naturalHeight || staticImgPreview.height;
    staticResultsContainer.innerHTML = `<div class="empty-notice">Running detection on ${sampleName}...</div>`;

    try {
      const resp = await fetch(sampleUrl);
      const blob = await resp.blob();
      const file = new File([blob], sampleName, { type: "image/jpeg" });

      const formData = new FormData();
      formData.append("file", file);
      formData.append("threshold", confidenceThreshold);

      const detResp = await fetch("/api/detect", {
        method: "POST",
        body: formData
      });
      const result = await detResp.json();
      renderStaticResults(result);
    } catch (err) {
      staticResultsContainer.innerHTML = `<div class="empty-notice" style="color: var(--red);">Inference Error: ${err.message}</div>`;
    }
  };
}
