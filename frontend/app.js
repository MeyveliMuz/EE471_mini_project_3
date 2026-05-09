// RoboMunch frontend logic.
// Backend assumed served from same origin (FastAPI mounts ./frontend at /).
// If opening index.html directly, set API_BASE to "http://127.0.0.1:8000".
const API_BASE = "";

const $ = (id) => document.getElementById(id);

const promptInput  = $("prompt-input");
const paintBtn     = $("paint-btn");
const outputImage  = $("output-image");
const imageSpinner = $("image-spinner");

const chatOutput = $("chat-output");
const chatInput  = $("chat-input");
const sendBtn    = $("send-btn");
const micBtn     = $("mic-btn");

const history = []; // [{role, content}]

/* -------------------- PAINT -------------------- */
paintBtn.addEventListener("click", async () => {
  const prompt = promptInput.value.trim();
  if (!prompt) return;
  paintBtn.disabled = true;
  imageSpinner.hidden = false;
  outputImage.removeAttribute("src");
  try {
    const res = await fetch(`${API_BASE}/api/generate-image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    outputImage.src = data.image;
  } catch (err) {
    alert("Image generation failed: " + err.message);
  } finally {
    paintBtn.disabled = false;
    imageSpinner.hidden = true;
  }
});

/* -------------------- CHAT -------------------- */
function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = "msg";
  const label = document.createElement("span");
  label.className = "label " + (role === "user" ? "you" : "munch");
  label.textContent = role === "user" ? "YOU:" : "MUNCH:";
  div.appendChild(label);
  div.appendChild(document.createTextNode(" " + text));
  chatOutput.appendChild(div);
  chatOutput.scrollTop = chatOutput.scrollHeight;
}

async function sendMessage() {
  const message = chatInput.value.trim();
  if (!message) return;
  chatInput.value = "";
  appendMessage("user", message);
  history.push({ role: "user", content: message });
  sendBtn.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: history.slice(0, -1) }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    appendMessage("assistant", data.reply);
    history.push({ role: "assistant", content: data.reply });
  } catch (err) {
    appendMessage("assistant", "(error: " + err.message + ")");
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendMessage);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); sendMessage(); }
});

/* -------------------- VOICE INPUT (MediaRecorder + /api/transcribe) -------------------- */
let mediaRecorder = null;
let recordedChunks = [];

micBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    alert("MediaDevices API not available in this browser.");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      micBtn.classList.remove("recording");
      const blob = new Blob(recordedChunks, { type: "audio/webm" });
      const fd = new FormData();
      fd.append("audio", blob, "voice.webm");
      micBtn.disabled = true;
      try {
        const res = await fetch(`${API_BASE}/api/transcribe`, {
          method: "POST",
          body: fd,
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        chatInput.value = data.text || "";
        chatInput.focus();
      } catch (err) {
        alert("Transcription failed: " + err.message);
      } finally {
        micBtn.disabled = false;
      }
    };
    mediaRecorder.start();
    micBtn.classList.add("recording");
  } catch (err) {
    alert("Microphone error: " + err.message);
  }
});
