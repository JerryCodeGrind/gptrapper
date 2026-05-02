/** Records mic for up to maxMs and resolves with a Blob. Throws Error('mic-permission-denied') on getUserMedia rejection.
 *  Disables echoCancellation/noiseSuppression/autoGainControl so quiet found-sounds
 *  (taps, scrapes, voice noises) aren't silenced by the browser's voice-call processing.
 */
export async function recordAudio(maxMs: number = 5000): Promise<Blob> {
  const constraints: MediaStreamConstraints = {
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
      sampleRate: 48000,
    },
  };
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      throw new Error('mic-permission-denied');
    }
  }

  const mimeType = MediaRecorder.isTypeSupported('audio/webm')
    ? 'audio/webm'
    : MediaRecorder.isTypeSupported('audio/ogg')
    ? 'audio/ogg'
    : '';
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  const chunks: Blob[] = [];

  return new Promise((resolve, reject) => {
    recorder.addEventListener('dataavailable', (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    });
    recorder.addEventListener('stop', () => {
      stream.getTracks().forEach((t) => t.stop());
      resolve(new Blob(chunks, { type: recorder.mimeType || 'audio/webm' }));
    });
    recorder.addEventListener('error', (e) => reject(e));
    recorder.start();
    setTimeout(() => {
      if (recorder.state !== 'inactive') recorder.stop();
    }, maxMs);
  });
}
