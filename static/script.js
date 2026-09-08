let currentVideoId = null;

const videoUpload = document.getElementById('videoUpload');
const videoStatus = document.getElementById('videoStatus');
const videoDetails = document.getElementById('videoDetails');

const chatWindow = document.getElementById('chatWindow');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');


// ---------------------------------------------------------
// Video upload
// ---------------------------------------------------------

videoUpload.addEventListener('change', async (event) => {
  const file = event.target.files[0];

  if (!file) {
    return;
  }

  currentVideoId = null;

  userInput.disabled = true;
  sendButton.disabled = true;

  userInput.placeholder =
    'Wait for the video to finish processing...';

  videoStatus.textContent =
    `Processing "${file.name}"...`;

  videoDetails.textContent = '';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(
      '/api/v1/video/ingest',
      {
        method: 'POST',
        body: formData,
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.detail || 'Video ingestion failed.'
      );
    }

    currentVideoId = data.video_id;

    videoStatus.textContent =
      `Successfully processed: ${data.filename}`;

    videoDetails.textContent =
      `Language: ${data.language || 'unknown'} | ` +
      `Segments: ${data.segment_count ?? 'N/A'} | ` +
      `Chunks: ${data.chunk_count ?? 'N/A'}`;

    userInput.disabled = false;
    sendButton.disabled = false;

    userInput.placeholder =
      'Ask a question about this video...';

    displayMessage(
      'Your video is ready. You can now ask questions about it.',
      'bot'
    );

    userInput.focus();

  } catch (error) {
    console.error(
      'Video upload failed:',
      error
    );

    videoStatus.textContent =
      `Upload failed: ${error.message}`;

    videoDetails.textContent = '';

    userInput.disabled = true;
    sendButton.disabled = true;
  }
});


// ---------------------------------------------------------
// Query handling
// ---------------------------------------------------------

sendButton.addEventListener(
  'click',
  sendMessage
);

userInput.addEventListener(
  'keydown',
  (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      sendMessage();
    }
  }
);


async function sendMessage() {
  const question = userInput.value.trim();

  if (!question) {
    return;
  }

  if (!currentVideoId) {
    displayMessage(
      'Please upload and process a video first.',
      'bot'
    );

    return;
  }

  displayMessage(
    question,
    'user'
  );

  userInput.value = '';
  userInput.disabled = true;
  sendButton.disabled = true;

  try {
    const response = await fetch(
      '/api/v1/query',
      {
        method: 'POST',

        headers: {
          'Content-Type': 'application/json',
        },

        body: JSON.stringify({
          video_id: currentVideoId,
          query: question,
          top_k: 5,
        }),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.detail || 'Query processing failed.'
      );
    }

    displayMessage(
      data.answer ||
        'I could not find an answer in the video.',
      'bot'
    );

  } catch (error) {
    console.error(
      'Query failed:',
      error
    );

    displayMessage(
      `Unable to process the question: ${error.message}`,
      'bot'
    );

  } finally {
    userInput.disabled = false;
    sendButton.disabled = false;
    userInput.focus();
  }
}


// ---------------------------------------------------------
// Chat display
// ---------------------------------------------------------

function displayMessage(message, sender) {
  const messageElement =
    document.createElement('div');

  messageElement.classList.add(
    'chat-message',
    sender === 'user'
      ? 'user-message'
      : 'bot-message'
  );

  messageElement.textContent =
    sender === 'bot'
      ? `VisionaryQ: ${message}`
      : message;

  chatWindow.appendChild(
    messageElement
  );

  chatWindow.scrollTop =
    chatWindow.scrollHeight;
}