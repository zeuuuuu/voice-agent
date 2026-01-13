// static/app.js — WEBRTC WITH PROVIDER FIX + RECORDING SUPPORT

let pc = null;
let stream = null;
let sessionId = null;
let isCallActive = false;

// ✅ Persisted provider state (SOURCE OF TRUTH)
let useElevenLabs = false;

// UI elements
const logEl = document.getElementById("log");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const endBtn = document.getElementById("endBtn");
const remoteAudio = document.getElementById("remoteAudio");

// Provider toggle UI
const providerToggle = document.getElementById("providerToggle");
const piperLabel = document.getElementById("piperLabel");
const elevenLabsLabel = document.getElementById("elevenLabsLabel");

// --------------------------------------------------
// Logging helper
// --------------------------------------------------
function log(msg) {
  console.log(msg);
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

// --------------------------------------------------
// Provider toggle handling
// --------------------------------------------------
useElevenLabs = providerToggle.checked;

function updateProviderUI() {
  if (useElevenLabs) {
    piperLabel.classList.remove("active");
    elevenLabsLabel.classList.add("active");
    log("📢 Provider set to ElevenLabs");
  } else {
    piperLabel.classList.add("active");
    elevenLabsLabel.classList.remove("active");
    log("📢 Provider set to Whisper/Piper");
  }
}

providerToggle.addEventListener("change", () => {
  useElevenLabs = providerToggle.checked;
  updateProviderUI();
});

// Initialize UI state
updateProviderUI();

// --------------------------------------------------
// Notify server that session is ending
// --------------------------------------------------
async function notifyServerEnd() {
  if (!sessionId) return;

  try {
    await fetch(`/end/${sessionId}`, { method: "POST" });
    log("✅ Session closed on server — recording should now be saved");
  } catch (e) {
    console.error("Failed to notify server of end:", e);
    log("⚠️ Could not reach server to save recording");
  }
  sessionId = null;
}

// --------------------------------------------------
// Cleanup (stop tracks, close PC, reset state)
// --------------------------------------------------
function cleanup() {
  isCallActive = false;

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  if (pc) {
    pc.close();
    pc = null;
  }

  remoteAudio.srcObject = null;

  // UI reset
  statusEl.textContent = "Idle";
  startBtn.disabled = false;
  endBtn.disabled = true;
}

// --------------------------------------------------
// Start Call
// --------------------------------------------------
startBtn.onclick = async () => {
  startBtn.disabled = true;
  statusEl.textContent = "Requesting microphone...";

  log(`🎯 Using provider: ${useElevenLabs ? "ElevenLabs" : "Whisper/Piper"}`);

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 48000,
      },
    });

    log("✅ Microphone access granted");

  } catch (e) {
    log("❌ Microphone access denied: " + e.message);
    startBtn.disabled = false;
    statusEl.textContent = "Idle";
    return;
  }

  // --------------------------------------------------
  // WebRTC Peer Connection
  // --------------------------------------------------
  pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  // Monitor connection state
  pc.onconnectionstatechange = () => {
    log(`Connection state: ${pc.connectionState}`);

    if (pc.connectionState === "closed" || pc.connectionState === "failed") {
      log("🔌 Connection closed/failed — ending session");
      endCall(); // This will trigger server /end
    }
  };

  pc.oniceconnectionstatechange = () => {
    log(`ICE state: ${pc.iceConnectionState}`);
  };

  // Send microphone audio
  stream.getTracks().forEach(track => {
    log(`📤 Sending track: ${track.kind} (${track.label})`);
    pc.addTrack(track, stream);
  });

  // Receive AI audio
  pc.ontrack = (event) => {
    if (event.track.kind === "audio") {
      log("🔊 AI voice connected");
      remoteAudio.srcObject = event.streams[0];
      remoteAudio.play().catch(err => {
        log("⚠️ Playback error: " + err.message);
      });
    }
  };

  // --------------------------------------------------
  // Offer / Answer exchange
  // --------------------------------------------------
  try {
    statusEl.textContent = "Creating offer...";
    const offer = await pc.createOffer({ offerToReceiveAudio: true });
    await pc.setLocalDescription(offer);

    log("📤 Sending offer to server");

    const res = await fetch("/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: offer.sdp,
        type: offer.type,
        use_elevenlabs: useElevenLabs,
      }),
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    sessionId = data.session_id;
    isCallActive = true;

    await pc.setRemoteDescription({
      sdp: data.sdp,
      type: data.type,
    });

    log("✅ Call connected — start speaking!");
    statusEl.textContent = "🎙️ Live — Speak now";
    endBtn.disabled = false;

  } catch (e) {
    console.error(e);
    log("❌ Connection failed: " + e.message);
    endCall(); // Ensures cleanup + server notify even on failure
  }
};

// --------------------------------------------------
// End Call (manual or automatic)
// --------------------------------------------------
async function endCall() {
  if (!isCallActive) return;

  log("🛑 Ending call...");

  cleanup();

  // Notify server to finalize recording
  await notifyServerEnd();

  log("Call fully ended");
}

// Manual end button
endBtn.onclick = () => {
  endCall();
};

// --------------------------------------------------
// Safety net: Try to notify server on page unload
// --------------------------------------------------
window.addEventListener("beforeunload", () => {
  if (sessionId) {
    // keepalive: true allows request even during unload
    navigator.sendBeacon(`/end/${sessionId}`);
    // Or fallback to fetch with keepalive
    // fetch(`/end/${sessionId}`, { method: "POST", keepalive: true });
  }
});