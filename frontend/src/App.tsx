import { useState, useRef, useEffect } from 'react'
import { CameraIcon, ArrowUpTrayIcon, LanguageIcon } from '@heroicons/react/24/solid'
import './App.css'
// Import custom icons
import uploadIcon from './assets/icons/button_upload.png'
import cameraIcon from './assets/icons/button_camera.png'
import aiIcon from './assets/icons/button_AI.png'

interface TextDetection {
  text: string;
  confidence: number;
  bounding_box: number[][];
  translated_text?: string;
}

interface AnalysisResult {
  original_language: string;
  target_language: string;
  metadata: {
    image_orientation: string;
    total_paragraphs: number;
    total_lines: number;
  };
  paragraphs: Array<{
    id: number;
    coordinates: number[];
    orientation: string;
    lines: Array<{
      coordinates: number[];
      original_text: string;
      translated_text: string;
    }>;
  }>;
}

// Define backend URL based on environment
const BACKEND_URL = process.env.NODE_ENV === 'production' 
  ? '/api'  // In production, use relative path
  : `http://${window.location.hostname}:8000/api`;  // In development, use the same host with backend port

function App() {
  // Language selection state
  const [sourceLanguage, setSourceLanguage] = useState('auto')
  const [targetLanguage, setTargetLanguage] = useState('en')
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [overlayImage, setOverlayImage] = useState<string | null>(null)
  const [showOverlay, setShowOverlay] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showCamera, setShowCamera] = useState(false)
  const [detectedTexts, setDetectedTexts] = useState<TextDetection[]>([])
  const [showAIResponse, setShowAIResponse] = useState(false)
  
  // Hidden file input ref
  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  
  // Browser detection
  const isFirefox = typeof navigator !== 'undefined' && navigator.userAgent.toLowerCase().includes('firefox')
  
  // Reference for tracking double tap timing
  const lastTapRef = useRef<number>(0);
  
  // Add utility function for image optimization
  const optimizeImage = async (file: File | Blob, maxDimension: number = 1920): Promise<Blob> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        // Calculate new dimensions while maintaining aspect ratio
        let width = img.width;
        let height = img.height;
        if (width > height && width > maxDimension) {
          height = (height * maxDimension) / width;
          width = maxDimension;
        } else if (height > maxDimension) {
          width = (width * maxDimension) / height;
          height = maxDimension;
        }

        // Create canvas and resize image
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('Failed to get canvas context');
        
        ctx.drawImage(img, 0, 0, width, height);
        
        // Convert to blob with reasonable quality
        canvas.toBlob(
          (blob) => {
            if (blob) {
              console.log(`[Timing] Image optimized from ${file.size} to ${blob.size} bytes`);
              resolve(blob);
            }
          },
          'image/jpeg',
          0.85  // Adjust quality for better compression
        );
      };
      img.src = URL.createObjectURL(file);
    });
  };

  // Modify processImageForOCR to use optimized image
  const processImageForOCR = async (file: File | Blob) => {
    const processStartTime = performance.now();
    console.log('[Timing] Starting OCR process...');
    setLoading(true);
    setError(null);
    
    try {
      // Optimize image before sending
      console.log('[Timing] Starting image optimization...');
      const optimizeStartTime = performance.now();
      const optimizedImage = await optimizeImage(file);
      console.log(`[Timing] Image optimization took ${(performance.now() - optimizeStartTime).toFixed(2)}ms`);

      // Create form data with optimized image
      const formData = new FormData();
      formData.append('file', optimizedImage, `image_${Date.now()}.jpg`);
      formData.append('source_lang', sourceLanguage);

      // Send OCR request
      console.log('[Timing] Sending OCR request to backend...');
      const requestStartTime = performance.now();
      const response = await fetch(`${BACKEND_URL}/ocr`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`OCR request failed: ${response.statusText}`);
      }

      const responseTime = performance.now();
      console.log(`[Timing] Backend request took ${(responseTime - requestStartTime).toFixed(2)}ms`);

      const data = await response.json();
      
      if (data.overlay_image) {
        const overlayUrl = process.env.NODE_ENV === 'production'
          ? data.overlay_image
          : `http://${window.location.hostname}:8000${data.overlay_image}`;
        
        console.log('[Timing] Starting overlay image loading...');
        const overlayStartTime = performance.now();
        
        // Preload image with timeout
        const loadImage = async (url: string, timeout: number = 10000): Promise<void> => {
          return new Promise((resolve, reject) => {
            const img = new Image();
            const timeoutId = setTimeout(() => {
              reject(new Error('Image loading timed out'));
            }, timeout);
            
            img.onload = () => {
              clearTimeout(timeoutId);
              setOverlayImage(url);
              setShowOverlay(true);
              resolve();
            };
            
            img.onerror = () => {
              clearTimeout(timeoutId);
              reject(new Error('Failed to load image'));
            };
            
            img.src = url;
          });
        };

        try {
          await loadImage(overlayUrl);
          console.log(`[Timing] Overlay image loaded in ${(performance.now() - overlayStartTime).toFixed(2)}ms`);
          
          // Store analysis result if needed
          if (data.analysis_result) {
            setDetectedTexts(data.analysis_result);
          }
        } catch (error) {
          console.error('Error loading overlay:', error);
          setError('Failed to load overlay image. Please try again.');
        }
      } else {
        console.log('No overlay image in response');
        setOverlayImage(null);
        setShowOverlay(false);
      }
    } catch (err) {
      console.error('Error processing image:', err);
      setError(err instanceof Error ? err.message : 'Failed to process image');
      setOverlayImage(null);
      setShowOverlay(false);
    } finally {
      setLoading(false);
      console.log(`[Timing] Total OCR process took ${(performance.now() - processStartTime).toFixed(2)}ms`);
    }
  };
  
  // Handle file upload
  const handleFileUpload = async (file: File) => {
    try {
      setLoading(true)
      setError(null)
      
      // Create URL for preview
      const imageUrl = URL.createObjectURL(file)
      setSelectedImage(imageUrl)
      setOverlayImage(null)  // Reset overlay when new image is uploaded
      setShowOverlay(false)  // Show original image first
      setDetectedTexts([])   // Reset detected texts
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process image')
    } finally {
      setLoading(false)
    }
  }
  
  // Handle file input change
  const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      handleFileUpload(file)
    }
  }

  // Handle translate button click
  const handleTranslateClick = async () => {
    if (!selectedImage) {
      setError('Please select an image first')
      return
    }

    const startTime = performance.now();
    console.log(`[Timing] Translation process started at ${new Date().toISOString()}`);
    
    try {
      setLoading(true)
      // Convert data URL to Blob
      console.log('[Timing] Starting data URL to Blob conversion...');
      const blobStartTime = performance.now();
      const response = await fetch(selectedImage)
      const blob = await response.blob()
      console.log(`[Timing] Blob conversion took ${(performance.now() - blobStartTime).toFixed(2)}ms`);
      
      // Create form data
      const formData = new FormData();
      formData.append('file', blob, `image_${Date.now()}.jpg`);
      formData.append('target_lang', targetLanguage);  // Use selected target language
      
      // Send request to backend
      const apiResponse = await fetch(`${BACKEND_URL}/ocr`, {
        method: 'POST',
        body: formData,
      });

      if (!apiResponse.ok) {
        throw new Error(`API request failed: ${apiResponse.statusText}`);
      }

      const data = await apiResponse.json();
      
      if (data.overlay_image) {
        const overlayUrl = process.env.NODE_ENV === 'production'
          ? data.overlay_image
          : `http://${window.location.hostname}:8000${data.overlay_image}`;
        
        console.log('[Timing] Starting overlay image loading...');
        const overlayStartTime = performance.now();
        
        // Preload image with timeout
        const loadImage = async (url: string, timeout: number = 10000): Promise<void> => {
          return new Promise((resolve, reject) => {
            const img = new Image();
            const timeoutId = setTimeout(() => {
              reject(new Error('Image loading timed out'));
            }, timeout);
            
            img.onload = () => {
              clearTimeout(timeoutId);
              setOverlayImage(url);
              setShowOverlay(true);
              resolve();
            };
            
            img.onerror = () => {
              clearTimeout(timeoutId);
              reject(new Error('Failed to load image'));
            };
            
            img.src = url;
          });
        };

        try {
          await loadImage(overlayUrl);
          console.log(`[Timing] Overlay image loaded in ${(performance.now() - overlayStartTime).toFixed(2)}ms`);
          
          // Store analysis result if needed
          if (data.analysis_result) {
            setDetectedTexts(data.analysis_result);
          }
        } catch (error) {
          console.error('Error loading overlay:', error);
          setError('Failed to load overlay image. Please try again.');
        }
      } else {
        console.log('No overlay image in response');
        setOverlayImage(null);
        setShowOverlay(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process image')
    } finally {
      setLoading(false)
      console.log(`[Timing] Total process took ${(performance.now() - startTime).toFixed(2)}ms`);
    }
  }
  
  // Camera functions
  const startCamera = async () => {
    try {
      // Reset any previous errors
      setError(null);
      console.log('Starting camera initialization...');
      console.log('Checking navigator.mediaDevices:', !!navigator.mediaDevices);
      console.log('Checking getUserMedia:', !!navigator.mediaDevices?.getUserMedia);

      // Most basic camera request possible
      const stream = await navigator.mediaDevices?.getUserMedia({
        video: true,
        audio: false
      });

      console.log('Camera stream obtained:', !!stream);
      
      if (!videoRef.current) {
        throw new Error('Video element not found');
      }

      // Set up the video stream
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      
      console.log('Stream attached to video element');
      setShowCamera(true);
      setError(null);

    } catch (err) {
      console.error('Detailed camera error:', err);
      
      // Check if we're in a secure context
      console.log('Is secure context:', window.isSecureContext);
      console.log('Protocol:', window.location.protocol);
      
      let errorMessage = 'Could not access camera. ';
      
      if (!window.isSecureContext) {
        errorMessage = 'Camera access requires a secure connection (HTTPS). Please use HTTPS or localhost.';
      } else if (!navigator.mediaDevices) {
        errorMessage = 'Camera API is not available. Please check if your browser supports camera access.';
      } else if (err instanceof Error) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          errorMessage = 'Please allow camera access in your browser settings and try again.';
        } else if (err.name === 'NotFoundError') {
          errorMessage = 'No camera found on your device.';
        } else if (err.name === 'NotReadableError') {
          errorMessage = 'Your camera might be in use by another app.';
        } else if (err.name === 'SecurityError') {
          errorMessage = 'Camera access is blocked by your browser security settings.';
        } else {
          errorMessage = `Camera error: ${err.message}`;
        }
      }

      setError(errorMessage);
      setShowCamera(false);
    }
  };

  const stopCamera = () => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop();
          console.log('Camera track stopped:', track.label);
        });
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setShowCamera(false);
    } catch (err) {
      console.error('Error stopping camera:', err);
      // Even if there's an error stopping the camera, we should still update the UI
      setShowCamera(false);
    }
  };

  const captureImage = async () => {
    if (!videoRef.current) {
      setError('Camera initialization failed. Please try restarting the camera.');
      return;
    }

    try {
      // Ensure video is playing and has valid dimensions
      if (videoRef.current.readyState !== videoRef.current.HAVE_ENOUGH_DATA) {
        throw new Error('Camera stream is not ready yet. Please wait a moment and try again.');
      }

      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('Failed to initialize image capture. Please try again.');
      }
      
      // Draw the current video frame
      ctx.drawImage(videoRef.current, 0, 0);
      
      // Convert to blob with good quality
      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob(
          (blob) => {
            if (blob) resolve(blob);
            else reject(new Error('Failed to capture image. Please try again.'));
          },
          'image/jpeg',
          0.95
        );
      });
      
      const imageUrl = URL.createObjectURL(blob);
      setSelectedImage(imageUrl);
      stopCamera();
      setError(null);
      
    } catch (err) {
      console.error('Error capturing image:', err);
      setError(err instanceof Error ? err.message : 'Failed to capture image. Please try again.');
      // Don't stop the camera on capture error so user can try again
    }
  };

  // Add a cleanup effect for camera
  useEffect(() => {
    return () => {
      // Cleanup camera on component unmount
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Update video element with better error handling
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleSuccess = () => {
      console.log('Camera stream started successfully');
      setError(null);
    };

    const handleError = (event: Event) => {
      console.error('Video element error:', event);
      setError('Camera stream failed. Please try again.');
      stopCamera();
    };

    video.addEventListener('loadedmetadata', handleSuccess);
    video.addEventListener('error', handleError);

    return () => {
      video.removeEventListener('loadedmetadata', handleSuccess);
      video.removeEventListener('error', handleError);
    };
  }, [showCamera]); // Re-run when showCamera changes

  // Language Selection Component
  const LanguageSelectionBar = () => (
    <div className="fixed top-0 left-0 right-0 bg-gray-800 py-2 z-50">
      <div className="flex justify-center items-center space-x-4">
        <select
          value={sourceLanguage}
          onChange={(e) => setSourceLanguage(e.target.value)}
          className="block w-32 rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50 bg-gray-700 text-white"
        >
          <option value="auto">Auto Detect</option>
          <option value="ja">Japanese</option>
          <option value="en">English</option>
          <option value="ko">Korean</option>
          <option value="zh">Chinese</option>
        </select>
        <span className="text-gray-400">→</span>
        <select
          value={targetLanguage}
          onChange={(e) => setTargetLanguage(e.target.value)}
          className="block w-32 rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50 bg-gray-700 text-white"
        >
          <option value="en">English</option>
          <option value="ja">Japanese</option>
          <option value="ko">Korean</option>
          <option value="zh">Chinese</option>
        </select>
      </div>
    </div>
  );

  // Bottom Action Bar Component
  const BottomActionBar = () => (
    <div className="flex justify-center space-x-6 items-center">
      {/* Share/Upload Button */}
      <button
        onClick={() => fileInputRef.current?.click()}
        className="flex items-center justify-center hover:opacity-80 transition-opacity"
      >
        <img src={uploadIcon} alt="Upload" className="w-12 h-12" />
      </button>

      {/* Camera Button */}
      {!showCamera ? (
        <button
          onClick={startCamera}
          className="flex items-center justify-center hover:opacity-80 transition-opacity"
        >
          <img src={cameraIcon} alt="Camera" className="w-16 h-16" />
        </button>
      ) : (
        <button
          onClick={captureImage}
          className="flex items-center justify-center hover:opacity-80 transition-opacity"
        >
          <img src={cameraIcon} alt="Capture" className="w-16 h-16" />
        </button>
      )}

      {/* AI Assistant Button */}
      <button
        onClick={() => {
          if (!selectedImage) {
            setError('Please take or upload an image first')
            return
          }
          handleTranslateClick()
        }}
        className={`flex items-center justify-center transition-opacity ${
          selectedImage 
            ? 'hover:opacity-80' 
            : 'opacity-50 cursor-not-allowed'
        }`}
        disabled={loading || showCamera}
      >
        <img src={aiIcon} alt="AI Assistant" className="w-12 h-12" />
      </button>

      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileInputChange}
        accept="image/*"
        className="hidden"
      />
    </div>
  );

  console.log('App rendering, showAIResponse:', showAIResponse);

  return (
    <div className="h-[100vh] max-h-[100vh] overflow-hidden flex flex-col bg-white">
      <LanguageSelectionBar />
      
      {/* Main content area with viewport height */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Error display */}
        {error && (
          <div className="absolute top-10 left-0 right-0 bg-red-100 border border-red-400 text-red-700 px-4 py-3 z-50" role="alert">
            <span className="block sm:inline">{error}</span>
          </div>
        )}

        {/* Loading indicator */}
        {loading && (
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
          </div>
        )}

        {/* Content wrapper with dynamic heights */}
        <div className="flex flex-col h-full">
          {/* Language bar spacer */}
          <div className="h-12" />

          {/* Image display area - takes remaining space */}
          <div className="flex-1 overflow-hidden p-4">
            <div className="h-full w-full flex items-center justify-center relative">
              {/* Dashed border box - always visible */}
              <div className="absolute inset-0 border-2 border-blue-200 border-dashed rounded-lg"></div>
              
              {showCamera ? (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="h-full w-full object-contain relative z-10"
                  onError={(e) => {
                    console.error('Video element error:', e);
                    setError('Video playback failed. Please try again.');
                  }}
                />
              ) : selectedImage ? (
                <div className="relative h-full w-full z-10">
                  <div className="h-full w-full">
                    <img
                      src={showOverlay ? overlayImage || selectedImage : selectedImage}
                      alt={showOverlay ? "Translated View" : "Original View"}
                      className="h-full w-full object-contain"
                      style={{ touchAction: 'manipulation' }}
                    />
                  </div>
                  {overlayImage && (
                    <>
                      <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
                        {showOverlay ? 'Translated Text' : 'Original Text'}
                      </div>
                      <button
                        onClick={() => setShowOverlay(!showOverlay)}
                        className="absolute top-2 right-2 bg-black bg-opacity-50 text-white p-2 rounded-full hover:bg-opacity-75 transition-opacity"
                      >
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                            d={showOverlay 
                              ? "M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" 
                              : "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"} 
                          />
                        </svg>
                      </button>
                    </>
                  )}
                </div>
              ) : (
                <div className="text-center relative z-10">
                  <p className="text-gray-500 text-lg">Take a photo or upload image</p>
                </div>
              )}
            </div>
          </div>

          {/* Bottom sections - fixed height */}
          <div className="flex flex-col mt-auto">
            {/* AI Assistant buttons */}
            <div className="px-2 py-1">
              {showAIResponse ? (
                <div className="relative bg-blue-50 rounded-lg p-4 shadow-lg border border-blue-100">
                  <div className="min-h-[12rem]">
                    <p>this is a test AI message, will be replace by LLM answers. stay tuned</p>
                    <p>this is a test AI message, will be replace by LLM answers. stay tuned</p>
                    <p>this is a test AI message, will be replace by LLM answers. stay tuned</p>
                    <p>this is a test AI message, will be replace by LLM answers. stay tuned</p>
                  </div>
                  <button 
                    onClick={() => setShowAIResponse(false)}
                    className="absolute top-2 right-2 text-blue-500 hover:text-blue-700"
                  >
                    Return
                  </button>
                </div>
              ) : (
                <div className="flex justify-center space-x-2 sm:space-x-4">
                  <button
                    onClick={() => setShowAIResponse(true)}
                    className="flex-1 bg-blue-50 hover:bg-blue-100 text-blue-800 font-semibold rounded-lg transition-colors shadow-lg border border-blue-100 flex items-center justify-center text-sm sm:text-lg px-2 py-2"
                  >
                    Explain the content
                  </button>
                  <button
                    onClick={() => setShowAIResponse(true)}
                    className="flex-1 bg-blue-50 hover:bg-blue-100 text-blue-800 font-semibold rounded-lg transition-colors shadow-lg border border-blue-100 flex items-center justify-center text-sm sm:text-lg px-2 py-2"
                  >
                    Maybe you want to ask this?
                  </button>
                  <button
                    onClick={() => setShowAIResponse(true)}
                    className="flex-1 bg-blue-50 hover:bg-blue-100 text-blue-800 font-semibold rounded-lg transition-colors shadow-lg border border-blue-100 flex items-center justify-center text-sm sm:text-lg px-2 py-2"
                  >
                    Custom
                  </button>
                </div>
              )}
            </div>

            {/* Camera buttons */}
            <div className="py-2">
              <BottomActionBar />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App