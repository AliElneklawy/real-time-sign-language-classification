document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video-feed');
    const predictionElement = document.getElementById('prediction');
    const confidenceElement = document.getElementById('confidence');
    
    let isProcessing = false;
    
    // Start video stream automatically when the page loads
    startVideo();
    
    // Capture a frame from the video element
    function captureFrame() {
        if (video.readyState === video.HAVE_ENOUGH_DATA) {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            
            // Draw the current video frame to the canvas
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Convert canvas to blob and send for classification
            return new Promise((resolve) => {
                canvas.toBlob(blob => {
                    resolve(blob);
                }, 'image/jpeg', 0.8);
            });
        }
        return null;
    }
    
    // Send frame to the server for classification
    async function classifyFrame(blob) {
        const formData = new FormData();
        formData.append('image', blob, 'frame.jpg');
        
        try {
            const response = await fetch('/classify', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error classifying frame:', error);
            return { status: 'error', message: error.message };
        }
    }
    
    // Process video frames for classification
    async function processVideo() {
        if (!isProcessing) return;
        
        try {
            // Capture frame from video
            const blob = await captureFrame();
            if (!blob) return;
            
            // Classify the frame
            const result = await classifyFrame(blob);
            
            // Update UI with results
            if (result.status === 'success' && result.prediction) {
                updateResults(result.prediction, result.confidence || 0);
            } else if (result.message === 'No hands detected') {
                updateResults('-', 0);
            }
            
        } catch (error) {
            console.error('Error processing video:', error);
        }
        
        // Continue processing the next frame
        if (isProcessing) {
            setTimeout(() => processVideo(), 100); // Process ~10 frames per second
        }
    }
    
    // Start video stream
    async function startVideo() {
        try {
            // Get the video feed URL from the data attribute
            const videoSrc = video.getAttribute('data-video-src');
            if (!videoSrc) {
                throw new Error('Video source not found');
            }
            
            // Set the video source
            video.src = videoSrc;
            
            // Wait for the video to start playing
            await new Promise((resolve) => {
                video.onplaying = resolve;
                // Add error handler
                video.onerror = (e) => {
                    console.error('Video error:', e);
                    throw new Error('Failed to load video feed');
                };
            });
            
            isProcessing = true;
            
            // Start processing frames
            processVideo();
            
        } catch (err) {
            console.error('Error starting video:', err);
            alert('Could not access the video feed. Please make sure the server is running and the camera is connected.');
        }
    }
    
    function updateResults(prediction, confidence) {
        predictionElement.textContent = prediction || '-';
        confidenceElement.textContent = `${(confidence * 100).toFixed(1)}%`;
    }
});
