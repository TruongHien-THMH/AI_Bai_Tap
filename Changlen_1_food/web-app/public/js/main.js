// main.js
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snapBtn = document.getElementById('snap');

// getUserMedia
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream)
  .catch(err => console.error("Camera error:", err));

snapBtn.addEventListener('click', () => {
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(async (blob) => {
    const form = new FormData();
    form.append('image', blob, 'photo.png');

    // send to Express endpoint
    const res = await fetch('/upload', { method: 'POST', body: form });
    // We expect Express to redirect/render result page. If we want SPA behavior,
    // we'd call Flask directly (bypass Express) or have Express return JSON.
    const html = await res.text();
    document.open();
    document.write(html);
    document.close();
  }, 'image/png');
});