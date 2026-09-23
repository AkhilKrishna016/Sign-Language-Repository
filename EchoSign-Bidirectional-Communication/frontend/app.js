/**
 * EchoSign — Frontend Interactive Logic
 * Character-by-Character ASL Sentence Composer & Communication System.
 * - Tracks 3D hand landmarks and recognizes single letters (A-Z).
 * - Hold gesture or tap to append letters to compose full words and sentences.
 * - User explicitly clicks "SEND SENTENCE TO MODEL" to trigger AI dialogue reply.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements - Navigation
  const navTabs = document.querySelectorAll(".nav-tab");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // DOM Elements - Camera & Canvas
  const videoFeed = document.getElementById("webcam-feed");
  const skeletonCanvas = document.getElementById("skeleton-canvas");
  const ctx = skeletonCanvas ? skeletonCanvas.getContext("2d") : null;
  const btnToggleCam = document.getElementById("btn-toggle-cam");
  const btnStartCamHero = document.getElementById("btn-start-cam-hero");
  const btnMirrorCam = document.getElementById("btn-mirror-cam");
  const camOfflineOverlay = document.getElementById("cam-offline-overlay");

  // DOM Elements - Character HUD
  const hudSymbol = document.getElementById("hud-symbol");
  const hudConfidence = document.getElementById("hud-confidence");
  const hudHandBadge = document.getElementById("hud-hand-badge");
  const hudHoldFill = document.getElementById("hud-hold-fill");
  const hudHoldPct = document.getElementById("hud-hold-pct");

  // DOM Elements - Sentence Composer
  const autoAddToggle = document.getElementById("auto-add-toggle");
  const sentenceTextPreview = document.getElementById("sentence-text-preview");
  const sentenceTokensStream = document.getElementById("sentence-tokens-stream");
  const btnAddChar = document.getElementById("btn-add-char");
  const btnAddSpace = document.getElementById("btn-add-space");
  const btnBackspace = document.getElementById("btn-backspace");
  const btnClearSentence = document.getElementById("btn-clear-sentence");
  const btnSendSentence = document.getElementById("btn-send-sentence");

  // DOM Elements - Listener & Captions
  const liveCaptionsText = document.getElementById("live-captions-text");
  const liveCaptionsGloss = document.getElementById("live-captions-gloss");
  const btnReplayTts = document.getElementById("btn-replay-tts");
  const autoTtsToggle = document.getElementById("auto-tts-toggle");
  const statusTierBadge = document.getElementById("status-tier-badge");
  const rawCvDisplay = document.getElementById("raw-cv-display");
  const reconstructedMeaningDisplay = document.getElementById("reconstructed-meaning-display");
  const confidenceMetricDisplay = document.getElementById("confidence-metric-display");
  const correctionsAppliedDisplay = document.getElementById("corrections-applied-display");
  const incomingSignCards = document.getElementById("incoming-sign-cards");
  const chatFeed = document.getElementById("chat-feed");
  const btnClearChat = document.getElementById("btn-clear-chat");

  // DOM Elements - Hearing Input
  const btnMicStt = document.getElementById("btn-mic-stt");
  const hearingTextInput = document.getElementById("hearing-text-input");
  const btnSendHearingMsg = document.getElementById("btn-send-hearing-msg");
  const micStatusLabel = document.getElementById("mic-status-label");

  // DOM Elements - Prompt Lab
  const labCustomInput = document.getElementById("lab-custom-input");
  const btnRunLabTest = document.getElementById("btn-run-lab-test");
  const btnClearLab = document.getElementById("btn-clear-lab");
  const labStatusBadge = document.getElementById("lab-status-badge");
  const labReconstructedOut = document.getElementById("lab-reconstructed-out");
  const labReplyOut = document.getElementById("lab-reply-out");
  const labGlossOut = document.getElementById("lab-gloss-out");
  const labCorrectionsOut = document.getElementById("lab-corrections-out");
  const presetButtons = document.querySelectorAll(".preset-btn");

  // DOM Elements - Dataset & Feedback
  const datasetGridView = document.getElementById("dataset-grid-view");
  const btnReloadDataset = document.getElementById("btn-reload-dataset");
  const feedbackTbody = document.getElementById("feedback-tbody");
  const btnReloadFeedback = document.getElementById("btn-reload-feedback");
  const btnClearFeedback = document.getElementById("btn-clear-feedback");
  const globalTtsPlayer = document.getElementById("global-tts-player");

  // State Variables
  let isCameraActive = false;
  let isMirrored = false;
  let mediaStream = null;
  let frameInterval = null;
  let lastAudioBase64 = null;
  let lastResponseText = "";
  let isRecordingMic = false;
  let recognitionInstance = null;

  // Character-by-Character Sentence State
  // Structure: Array of words, where each word is Array of { char: 'A', conf: 0.92 }
  let sentenceWords = [[]];
  let currentDetectedChar = null;
  let currentDetectedConf = 0.0;
  let currentHeldChar = null;
  let heldFramesCount = 0;
  const HOLD_REQUIRED_FRAMES = 6; // ~0.8s
  let cooldownFrames = 0;

  // 1. Navigation Tabs
  navTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      navTabs.forEach((t) => t.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");

      if (targetId === "dataset-tab") loadDatasetSamples();
      else if (targetId === "feedback-tab") loadFeedbackLogs();
      lucide.createIcons();
    });
  });

  // 2. Camera Controls
  async function startCamera() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      videoFeed.srcObject = mediaStream;
      await videoFeed.play();

      isCameraActive = true;
      camOfflineOverlay.style.display = "none";
      btnToggleCam.innerHTML = `<i data-lucide="camera-off"></i> <span>Stop Camera</span>`;
      btnToggleCam.classList.remove("btn-primary");
      btnToggleCam.classList.add("btn-secondary");
      lucide.createIcons();

      skeletonCanvas.width = videoFeed.videoWidth || 640;
      skeletonCanvas.height = videoFeed.videoHeight || 480;

      frameInterval = setInterval(captureAndSendFrame, 130);
    } catch (err) {
      console.error("Camera access error:", err);
      alert("Unable to access webcam. Please check browser permissions.");
    }
  }

  function stopCamera() {
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }
    if (frameInterval) {
      clearInterval(frameInterval);
      frameInterval = null;
    }
    isCameraActive = false;
    camOfflineOverlay.style.display = "flex";
    btnToggleCam.innerHTML = `<i data-lucide="camera"></i> <span>Start Camera</span>`;
    btnToggleCam.classList.add("btn-primary");
    btnToggleCam.classList.remove("btn-secondary");
    if (ctx) ctx.clearRect(0, 0, skeletonCanvas.width, skeletonCanvas.height);
    currentDetectedChar = null;
    updateHUD({ hand_present: false });
    lucide.createIcons();
  }

  if (btnToggleCam) {
    btnToggleCam.addEventListener("click", () => {
      if (isCameraActive) stopCamera();
      else startCamera();
    });
  }
  if (btnStartCamHero) btnStartCamHero.addEventListener("click", startCamera);

  if (btnMirrorCam) {
    btnMirrorCam.addEventListener("click", () => {
      isMirrored = !isMirrored;
      videoFeed.classList.toggle("mirrored", isMirrored);
      skeletonCanvas.classList.toggle("mirrored", isMirrored);
    });
  }

  // 3. Frame Capture & Real-time Letter Recognition
  const offscreenCanvas = document.createElement("canvas");
  const offscreenCtx = offscreenCanvas.getContext("2d");

  async function captureAndSendFrame() {
    if (!isCameraActive || videoFeed.readyState < 2) return;

    offscreenCanvas.width = videoFeed.videoWidth || 640;
    offscreenCanvas.height = videoFeed.videoHeight || 480;
    offscreenCtx.drawImage(videoFeed, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

    const base64Data = offscreenCanvas.toDataURL("image/jpeg", 0.7);

    try {
      const resp = await fetch("/api/recognize-frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: base64Data }),
      });

      if (!resp.ok) return;
      const data = await resp.json();
      updateHUD(data);

      if (data.landmarks && data.landmarks.length > 0) {
        renderSkeleton(data.landmarks);
      } else if (ctx) {
        ctx.clearRect(0, 0, skeletonCanvas.width, skeletonCanvas.height);
      }

      // Character stabilization & Auto-hold processing
      processCharacterHold(data);
    } catch (e) {
      console.warn("Recognition error:", e);
    }
  }

  function renderSkeleton(landmarks) {
    if (!ctx) return;
    ctx.clearRect(0, 0, skeletonCanvas.width, skeletonCanvas.height);

    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [0, 9], [9, 10], [10, 11], [11, 12],
      [0, 13], [13, 14], [14, 15], [15, 16],
      [0, 17], [17, 18], [18, 19], [19, 20],
      [5, 9], [9, 13], [13, 17],
    ];

    const w = skeletonCanvas.width;
    const h = skeletonCanvas.height;

    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    connections.forEach(([i, j]) => {
      if (landmarks[i] && landmarks[j]) {
        ctx.beginPath();
        ctx.moveTo(landmarks[i].x * w, landmarks[i].y * h);
        ctx.lineTo(landmarks[j].x * w, landmarks[j].y * h);
        ctx.stroke();
      }
    });

    landmarks.forEach((lm) => {
      ctx.fillStyle = "#818cf8";
      ctx.beginPath();
      ctx.arc(lm.x * w, lm.y * h, 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }

  function updateHUD(data) {
    if (data.hand_present && data.detected_symbol) {
      currentDetectedChar = data.detected_symbol;
      currentDetectedConf = data.confidence;
      hudSymbol.textContent = data.detected_symbol;
      hudConfidence.textContent = data.confidence.toFixed(2);
      hudHandBadge.innerHTML = `<i data-lucide="hand"></i> Hands: ${data.hands_detected || 1} (${data.handedness})`;
      hudHandBadge.style.color = "#34d399";
    } else {
      currentDetectedChar = null;
      hudSymbol.textContent = "—";
      hudConfidence.textContent = "0.00";
      hudHandBadge.innerHTML = `<i data-lucide="hand"></i> Hand: None`;
      hudHandBadge.style.color = "#94a3b8";
    }
    lucide.createIcons();
  }

  // 4. Character Hold & Auto-Append Engine
  function processCharacterHold(data) {
    if (cooldownFrames > 0) {
      cooldownFrames--;
      return;
    }

    if (!data.hand_present || !data.detected_symbol || data.confidence < 0.5) {
      heldFramesCount = 0;
      currentHeldChar = null;
      updateHoldProgress(0);
      return;
    }

    const symbol = data.detected_symbol;

    if (symbol === currentHeldChar) {
      heldFramesCount++;
    } else {
      currentHeldChar = symbol;
      heldFramesCount = 1;
    }

    const pct = Math.min(100, Math.round((heldFramesCount / HOLD_REQUIRED_FRAMES) * 100));
    updateHoldProgress(pct);

    // Auto-add when hold threshold is reached
    if (autoAddToggle && autoAddToggle.checked && heldFramesCount >= HOLD_REQUIRED_FRAMES) {
      addCharacter(symbol, data.confidence);
      heldFramesCount = 0;
      cooldownFrames = 8; // ~1s cooldown before same character can be re-triggered
      updateHoldProgress(0);
    }
  }

  function updateHoldProgress(pct) {
    if (hudHoldFill) hudHoldFill.style.width = `${pct}%`;
    if (hudHoldPct) hudHoldPct.textContent = `${pct}%`;
  }

  // 5. Sentence Builder Management
  function addCharacter(char, conf = 0.85) {
    if (!char) return;

    if (char === " " || char.toLowerCase() === "space") {
      addSpace();
      return;
    }
    if (char.toLowerCase() === "del") {
      backspace();
      return;
    }

    // Append character to last word
    let lastWord = sentenceWords[sentenceWords.length - 1];
    lastWord.push({ char: char.toUpperCase(), conf: roundConf(conf) });
    renderComposedSentence();
  }

  function addSpace() {
    let lastWord = sentenceWords[sentenceWords.length - 1];
    if (lastWord.length > 0) {
      sentenceWords.push([]);
      renderComposedSentence();
    }
  }

  function backspace() {
    let lastWord = sentenceWords[sentenceWords.length - 1];
    if (lastWord.length > 0) {
      lastWord.pop();
    } else if (sentenceWords.length > 1) {
      sentenceWords.pop();
    }
    renderComposedSentence();
  }

  function clearSentence() {
    sentenceWords = [[]];
    renderComposedSentence();
  }

  function roundConf(val) {
    return typeof val === "number" ? Math.round(val * 100) / 100 : 0.85;
  }

  function renderComposedSentence() {
    const activeWords = sentenceWords.filter((w) => w.length > 0);

    if (activeWords.length === 0 && sentenceWords[0].length === 0) {
      sentenceTextPreview.innerHTML = `
        <span class="empty-sentence-placeholder">Sign letters to begin spelling words...</span>
        <span class="blinking-cursor">|</span>
      `;
      sentenceTokensStream.innerHTML = "";
      return;
    }

    // Build plain sentence string
    const plainWords = sentenceWords.map((word) => word.map((item) => item.char).join(""));
    const plainSentence = plainWords.join(" ");

    sentenceTextPreview.innerHTML = `
      <span>${plainSentence}</span>
      <span class="blinking-cursor">|</span>
    `;

    // Render word chips with letter confidence badges
    sentenceTokensStream.innerHTML = "";
    sentenceWords.forEach((word) => {
      if (word.length === 0) return;
      const wordPill = document.createElement("div");
      wordPill.className = "word-pill";
      wordPill.innerHTML = word
        .map((item) => `<span class="letter-chip">${item.char}<small>(${item.conf})</small></span>`)
        .join("<span style='color:var(--text-dim)'>-</span>");
      sentenceTokensStream.appendChild(wordPill);
    });
  }

  // Button actions for composer
  if (btnAddChar) {
    btnAddChar.addEventListener("click", () => {
      if (currentDetectedChar) addCharacter(currentDetectedChar, currentDetectedConf);
    });
  }
  if (btnAddSpace) btnAddSpace.addEventListener("click", addSpace);
  if (btnBackspace) btnBackspace.addEventListener("click", backspace);
  if (btnClearSentence) btnClearSentence.addEventListener("click", clearSentence);

  // Keyboard Shortcuts for Sentence Composer
  document.addEventListener("keydown", (e) => {
    // Only capture if not focused inside an input or textarea
    if (document.activeElement && ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;

    if (e.code === "Space") {
      e.preventDefault();
      addSpace();
    } else if (e.key === "Backspace") {
      e.preventDefault();
      backspace();
    } else if (e.key === "Enter") {
      e.preventDefault();
      sendSentenceToModel();
    }
  });

  // 6. Explicit Send Sentence Action -> AI Dialogue Layer
  async function sendSentenceToModel() {
    const activeWords = sentenceWords.filter((w) => w.length > 0);
    if (activeWords.length === 0) {
      alert("Please sign or add some characters first before sending.");
      return;
    }

    // Format into structured prompt syntax: [FS] L(conf)-L(conf)... [FS] ...
    const structuredParts = activeWords.map((word) => {
      const fsTokens = word.map((item) => `${item.char}(${item.conf})`).join("-");
      return `[FS] ${fsTokens}`;
    });

    const structuredPayload = structuredParts.join(" ");

    btnSendSentence.disabled = true;
    btnSendSentence.innerHTML = `<i data-lucide="loader"></i> <span>Interpreting Sentence...</span>`;
    lucide.createIcons();

    try {
      const resp = await fetch("/api/understand", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ raw_input: structuredPayload }),
      });

      if (!resp.ok) throw new Error("Server error");
      const res = await resp.json();
      displayUnderstandingResult(res);

      // Clear composed sentence for next turn!
      clearSentence();
    } catch (err) {
      console.error("Language engine error:", err);
    } finally {
      btnSendSentence.disabled = false;
      btnSendSentence.innerHTML = `<i data-lucide="send"></i> <span>SEND SENTENCE TO MODEL</span>`;
      lucide.createIcons();
    }
  }

  if (btnSendSentence) {
    btnSendSentence.addEventListener("click", sendSentenceToModel);
  }

  function displayUnderstandingResult(res) {
    liveCaptionsText.textContent = `"${res.response_text}"`;
    liveCaptionsGloss.textContent = `GLOSS: ${res.gloss_reply || "—"}`;
    lastResponseText = res.response_text;
    lastAudioBase64 = res.audio_base64;

    statusTierBadge.className = "badge";
    if (res.status === "reliable") {
      statusTierBadge.classList.add("badge-reliable");
      statusTierBadge.textContent = "Reliable (≥0.75)";
    } else if (res.status === "clarify") {
      statusTierBadge.classList.add("badge-clarify");
      statusTierBadge.textContent = "Clarify (0.4–0.75)";
    } else {
      statusTierBadge.classList.add("badge-repeat");
      statusTierBadge.textContent = "Repeat (<0.4)";
    }

    rawCvDisplay.textContent = res.raw_input || "—";
    reconstructedMeaningDisplay.textContent = res.reconstructed_english || "—";
    confidenceMetricDisplay.innerHTML = `Avg: <span>${res.average_confidence}</span> | Min: <span>${res.min_confidence}</span>`;

    if (res.applied_corrections && res.applied_corrections.length > 0) {
      correctionsAppliedDisplay.textContent = res.applied_corrections.join("; ");
      correctionsAppliedDisplay.style.color = "#38bdf8";
    } else {
      correctionsAppliedDisplay.textContent = "None";
      correctionsAppliedDisplay.style.color = "#94a3b8";
    }

    appendChatMessage("signer", res.reconstructed_english || res.raw_input);
    appendChatMessage("hearing", res.response_text);

    if (autoTtsToggle && autoTtsToggle.checked && res.response_text) {
      playTtsAudio(res.audio_base64, res.response_text);
    }
  }

  function appendChatMessage(sender, text) {
    if (!text || !chatFeed) return;
    const msgDiv = document.createElement("div");
    msgDiv.className = `dialogue-item ${sender === "signer" ? "signer-msg" : "hearing-msg"}`;
    msgDiv.innerHTML = `
      <div class="msg-sender">${sender === "signer" ? "Signer (Deaf/Mute)" : "Assistant / Partner"}</div>
      <div class="msg-text">${text}</div>
    `;
    chatFeed.appendChild(msgDiv);
    chatFeed.scrollTop = chatFeed.scrollHeight;
  }

  function playTtsAudio(base64Audio, fallbackText) {
    if (base64Audio) {
      globalTtsPlayer.src = `data:audio/mp3;base64,${base64Audio}`;
      globalTtsPlayer.play().catch((e) => console.log("Audio play deferred:", e));
    } else if ("speechSynthesis" in window && fallbackText) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(fallbackText);
      utterance.rate = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  }

  if (btnReplayTts) {
    btnReplayTts.addEventListener("click", () => {
      playTtsAudio(lastAudioBase64, lastResponseText);
    });
  }

  if (btnClearChat) {
    btnClearChat.addEventListener("click", () => {
      chatFeed.innerHTML = `
        <div class="dialogue-item system-notice">
          <i data-lucide="info"></i> Transcript cleared. Ready for communication.
        </div>
      `;
      lucide.createIcons();
    });
  }

  // 7. Reverse Communication Channel: Hearing Partner -> Deaf Signer
  async function handleSendHearingMessage() {
    const text = hearingTextInput.value.trim();
    if (!text) return;

    hearingTextInput.value = "";
    try {
      const resp = await fetch("/api/speech-to-sign", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ english_text: text }),
      });

      if (!resp.ok) return;
      const data = await resp.json();
      renderVisualSignCards(data);
      appendChatMessage("hearing", text);
    } catch (e) {
      console.error("Speech-to-sign error:", e);
    }
  }

  function renderVisualSignCards(data) {
    if (!incomingSignCards) return;
    incomingSignCards.innerHTML = "";

    if (!data.visual_tokens || data.visual_tokens.length === 0) {
      incomingSignCards.innerHTML = `<div class="empty-cards-placeholder">No visual signs available.</div>`;
      return;
    }

    data.visual_tokens.forEach((token) => {
      const card = document.createElement("div");
      card.className = "visual-sign-card";
      card.innerHTML = `
        <div class="sign-card-icon">
          <i data-lucide="${token.icon || 'hand-metal'}"></i>
        </div>
        <div class="sign-card-word">${token.text}</div>
        <div class="sign-card-desc">${token.description}</div>
      `;
      incomingSignCards.appendChild(card);
    });
    lucide.createIcons();
  }

  if (btnSendHearingMsg) btnSendHearingMsg.addEventListener("click", handleSendHearingMessage);
  if (hearingTextInput) {
    hearingTextInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") handleSendHearingMessage();
    });
  }

  // 8. Speech-to-Text (STT)
  if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognitionInstance = new SpeechRecognition();
    recognitionInstance.continuous = false;
    recognitionInstance.interimResults = false;
    recognitionInstance.lang = "en-US";

    recognitionInstance.onstart = () => {
      isRecordingMic = true;
      btnMicStt.classList.add("recording");
      micStatusLabel.textContent = "Listening... Speak clearly now";
    };

    recognitionInstance.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      hearingTextInput.value = transcript;
      micStatusLabel.textContent = `Captured: "${transcript}"`;
      handleSendHearingMessage();
    };

    recognitionInstance.onerror = (e) => {
      console.error("Speech recognition error:", e);
      micStatusLabel.textContent = "Could not capture audio. Try typing below.";
      btnMicStt.classList.remove("recording");
      isRecordingMic = false;
    };

    recognitionInstance.onend = () => {
      btnMicStt.classList.remove("recording");
      isRecordingMic = false;
    };

    if (btnMicStt) {
      btnMicStt.addEventListener("click", () => {
        if (isRecordingMic) {
          recognitionInstance.stop();
        } else {
          recognitionInstance.start();
        }
      });
    }
  }

  // 9. Prompt Lab Presets
  presetButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const inputVal = btn.getAttribute("data-input");
      if (inputVal && labCustomInput) {
        labCustomInput.value = inputVal;
        runLabTest(inputVal);
      }
    });
  });

  if (btnRunLabTest) {
    btnRunLabTest.addEventListener("click", () => {
      const inputVal = labCustomInput.value.trim();
      if (inputVal) runLabTest(inputVal);
    });
  }

  if (btnClearLab) {
    btnClearLab.addEventListener("click", () => {
      labCustomInput.value = "";
      labReconstructedOut.textContent = "—";
      labReplyOut.textContent = "—";
      labGlossOut.textContent = "—";
      labCorrectionsOut.textContent = "None";
      labStatusBadge.textContent = "Ready";
      labStatusBadge.className = "badge badge-reliable";
    });
  }

  async function runLabTest(rawText) {
    try {
      const resp = await fetch("/api/understand", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ raw_input: rawText }),
      });

      if (!resp.ok) return;
      const res = await resp.json();

      labReconstructedOut.textContent = res.reconstructed_english || "—";
      labReplyOut.textContent = res.response_text || "—";
      labGlossOut.textContent = res.gloss_reply || "—";
      labCorrectionsOut.textContent = res.applied_corrections && res.applied_corrections.length > 0
        ? res.applied_corrections.join("; ")
        : "None";

      labStatusBadge.className = "badge";
      if (res.status === "reliable") {
        labStatusBadge.classList.add("badge-reliable");
        labStatusBadge.textContent = `Reliable (Avg: ${res.average_confidence})`;
      } else if (res.status === "clarify") {
        labStatusBadge.classList.add("badge-clarify");
        labStatusBadge.textContent = `Clarification Needed (Avg: ${res.average_confidence})`;
      } else {
        labStatusBadge.classList.add("badge-repeat");
        labStatusBadge.textContent = `Repeat Requested (Avg: ${res.average_confidence})`;
      }

      if (res.response_text) {
        playTtsAudio(res.audio_base64, res.response_text);
      }
    } catch (err) {
      console.error("Lab test error:", err);
    }
  }

  // 10. Dataset Samples
  async function loadDatasetSamples() {
    if (!datasetGridView) return;
    try {
      const resp = await fetch("/api/dataset-samples");
      if (!resp.ok) return;
      const data = await resp.json();

      datasetGridView.innerHTML = "";
      if (!data.samples || data.samples.length === 0) {
        datasetGridView.innerHTML = `<div class="empty-cards-placeholder">No dataset samples found.</div>`;
        return;
      }

      data.samples.forEach((sample) => {
        const card = document.createElement("div");
        card.className = "dataset-card";
        card.innerHTML = `
          <img src="/api/dataset-image/${sample.filename}" alt="ASL ${sample.label}" loading="lazy" />
          <div class="dataset-label">${sample.label}</div>
        `;
        card.addEventListener("click", () => {
          // Add this character directly to sentence builder!
          addCharacter(sample.label, 0.95);
          const studioTab = document.getElementById("tab-studio-btn");
          if (studioTab) studioTab.click();
        });
        datasetGridView.appendChild(card);
      });
    } catch (e) {
      console.error("Failed to load dataset:", e);
    }
  }

  if (btnReloadDataset) btnReloadDataset.addEventListener("click", loadDatasetSamples);

  // 11. Feedback Logs
  async function loadFeedbackLogs() {
    if (!feedbackTbody) return;
    try {
      const resp = await fetch("/api/feedback-logs?limit=50");
      if (!resp.ok) return;
      const data = await resp.json();

      feedbackTbody.innerHTML = "";
      if (!data.logs || data.logs.length === 0) {
        feedbackTbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color: var(--text-dim);">No logged turns yet.</td></tr>`;
        return;
      }

      data.logs.forEach((log) => {
        const tr = document.createElement("tr");
        const statusBadge = log.status === "repeat"
          ? `<span class="badge badge-repeat">REPEAT</span>`
          : (log.status === "clarify" ? `<span class="badge badge-clarify">CLARIFY</span>` : `<span class="badge badge-reliable">RELIABLE</span>`);

        tr.innerHTML = `
          <td style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-dim);">${log.iso_time}</td>
          <td><code class="metric-code">${log.raw_input}</code></td>
          <td>${statusBadge}</td>
          <td>Avg: ${log.average_confidence} | Min: ${log.min_confidence}</td>
          <td style="color: var(--primary); font-weight: 500;">${log.reconstructed_english || "—"}</td>
          <td style="color: var(--text-muted); font-size: 0.8rem;">${log.reason}</td>
        `;
        feedbackTbody.appendChild(tr);
      });
    } catch (e) {
      console.error("Failed to load feedback logs:", e);
    }
  }

  if (btnReloadFeedback) btnReloadFeedback.addEventListener("click", loadFeedbackLogs);
  if (btnClearFeedback) {
    btnClearFeedback.addEventListener("click", async () => {
      await fetch("/api/clear-feedback-logs", { method: "POST" });
      loadFeedbackLogs();
    });
  }

  renderComposedSentence();
  lucide.createIcons();
});
