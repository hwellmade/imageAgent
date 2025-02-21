import { useState, useRef, useEffect } from 'react'
import { CameraIcon, ArrowUpTrayIcon, LanguageIcon, XMarkIcon } from '@heroicons/react/24/solid'
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
  
  // Add viewport measurement states
  const [viewportHeight, setViewportHeight] = useState(0);
  const [usableHeight, setUsableHeight] = useState(0);
  
  // Add new ref for camera input
  const cameraInputRef = useRef<HTMLInputElement>(null)
  
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
      console.log('🚨 RAW BACKEND RESPONSE:', JSON.stringify(data, null, 2));
      console.log('========================');
      console.log('🔍 RESPONSE DATA:', data);
      console.log('📝 Response type:', typeof data);
      console.log('🔑 Response keys:', Object.keys(data));
      console.log('🌐 Translated overlay path:', data.translated_overlay);
      console.log('🌐 Original overlay path:', data.original_overlay);
      
      // Check if we have the overlay paths
      if (!data.translated_overlay || !data.original_overlay) {
        console.error('❌ Missing overlay paths in response');
        console.log('📋 Full response data:', JSON.stringify(data, null, 2));
        setError('Missing overlay data in response');
        return;
      }

      // Construct URLs
      const translatedUrl = process.env.NODE_ENV === 'production'
        ? data.translated_overlay
        : `http://${window.location.hostname}:8000${data.translated_overlay}`;
        
      const originalUrl = process.env.NODE_ENV === 'production'
        ? data.original_overlay
        : `http://${window.location.hostname}:8000${data.original_overlay}`;
      
      console.log('🔗 Constructed URLs:', {
        translatedUrl,
        originalUrl,
        hostname: window.location.hostname,
        BACKEND_URL
      });
      
      if (data.translated_overlay && data.original_overlay) {
        // Preload both images
        const loadImage = async (url: string, timeout: number = 10000): Promise<void> => {
          return new Promise((resolve, reject) => {
            const img = new Image();
            const timeoutId = setTimeout(() => {
              reject(new Error('Image loading timed out'));
            }, timeout);
            
            img.onload = () => {
              clearTimeout(timeoutId);
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
          console.log('[Timing] Starting overlay images loading...');
          const overlayStartTime = performance.now();
          
          // Load both images in parallel
          await Promise.all([
            loadImage(originalUrl),
            loadImage(translatedUrl)
          ]);
          
          console.log(`[Timing] Overlay images loaded in ${(performance.now() - overlayStartTime).toFixed(2)}ms`);
          
          // Set the translated overlay as the initial view
          setOverlayImage(translatedUrl);
          setShowOverlay(true);
          
          // Store analysis result if needed
          if (data.text_blocks) {
            setDetectedTexts(data.text_blocks);
          }
        } catch (error) {
          console.error('Error loading overlay images:', error);
          setError('Failed to load overlay images. Please try again.');
        }
      } else {
        console.log('No overlay images in response');
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
        const errorData = await apiResponse.json().catch(() => ({ detail: 'Unknown error' }));
        console.error('🚨 Backend Error:', {
          status: apiResponse.status,
          statusText: apiResponse.statusText,
          error: errorData
        });
        throw new Error(`OCR request failed: ${errorData.detail || apiResponse.statusText}`);
      }

      const data = await apiResponse.json();
      console.log('🔍 Response data:', data);
      
      if (data.translated_overlay && data.original_overlay) {
        const overlayUrl = process.env.NODE_ENV === 'production'
          ? data.translated_overlay
          : `http://${window.location.hostname}:8000${data.translated_overlay}`;
        
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
          if (data.text_blocks) {
            setDetectedTexts(data.text_blocks);
          }
        } catch (error) {
          console.error('Error loading overlay:', error);
          setError('Failed to load overlay image. Please try again.');
        }
      } else {
        console.log('No overlay images in response:', data);
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
      setError(null);
      console.log('Starting camera initialization...');
      
      const constraints = {
        video: {
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        },
        audio: false
      };

      const stream = await navigator.mediaDevices?.getUserMedia(constraints);
      
      if (!stream) {
        throw new Error('Failed to get camera stream');
      }

      if (!videoRef.current) {
        throw new Error('Video element not found');
      }

      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      setShowCamera(true);
      setError(null);

    } catch (err) {
      console.error('Camera error:', err);
      
      // Fall back to file input with camera
      if (fileInputRef.current) {
        fileInputRef.current.click();
      }
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

  // Effect to measure and update viewport dimensions
  useEffect(() => {
    const updateViewportDimensions = () => {
      const vh = window.innerHeight;
      const usableH = document.documentElement.clientHeight;
      
      console.log('Full viewport height:', vh);
      console.log('Usable viewport height:', usableH);
      
      setViewportHeight(vh);
      setUsableHeight(usableH);

      // Set a CSS custom property for the usable height
      document.documentElement.style.setProperty('--usable-height', `${usableH}px`);
    };

    updateViewportDimensions();
    window.addEventListener('resize', updateViewportDimensions);
    window.addEventListener('orientationchange', updateViewportDimensions);

    return () => {
      window.removeEventListener('resize', updateViewportDimensions);
      window.removeEventListener('orientationchange', updateViewportDimensions);
    };
  }, []);

  // Calculate component heights
  const imageAreaHeight = Math.floor(usableHeight * 0.6);  // 60% for image
  const buttonAreaHeight = Math.floor(usableHeight * 0.3);  // 30% total for both button areas
  const navAreaHeight = Math.floor(usableHeight * 0.1);    // 10% for nav

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
      <button
        onClick={() => cameraInputRef.current?.click()}
        className="flex items-center justify-center hover:opacity-80 transition-opacity"
      >
        <img src={cameraIcon} alt="Camera" className="w-16 h-16" />
      </button>

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

      {/* Hidden file input for uploads */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileInputChange}
        accept="image/*"
        className="hidden"
      />

      {/* Hidden file input for camera */}
      <input
        type="file"
        ref={cameraInputRef}
        onChange={handleFileInputChange}
        accept="image/*"
        capture="environment"
        className="hidden"
      />
    </div>
  );

  console.log('App rendering, showAIResponse:', showAIResponse);

  return (
    <div 
      className="flex flex-col bg-white"
      style={{ 
        height: `${usableHeight}px`,
        maxHeight: `${usableHeight}px`,
        overflow: 'hidden'
      }}
    >
      {/* Language Selection Bar - fixed height */}
      <div className="h-[2vh]">
        <LanguageSelectionBar />
      </div>
      
      {/* Main content area - takes remaining space */}
      <main className="flex-1 flex flex-col min-h-0 mt-0">
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

        {/* Image display area - 60% of space */}
        <div className="h-[60vh] w-full flex flex-col relative">
          {/* Dashed border box - always visible */}
          <div className="absolute inset-0 border border-blue-500 border-dashed"></div>
          
          {showCamera ? (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-contain bg-black z-10"
              onError={(e) => {
                console.error('Video element error:', e);
                setError('Video playback failed. Please try again.');
              }}
            />
          ) : selectedImage ? (
            <div className="absolute inset-0 w-full h-full z-10 bg-black">
              <div 
                className="w-full h-full cursor-pointer"
                onDoubleClick={() => {
                  if (overlayImage) {
                    setShowOverlay(!showOverlay);
                  }
                }}
              >
                <img
                  src={showOverlay ? overlayImage || selectedImage : selectedImage}
                  alt={showOverlay ? "Translated View" : "Original View"}
                  className="w-full h-full object-contain"
                  style={{ 
                    touchAction: 'none',
                    userSelect: 'none'
                  }}
                />
              </div>
              {overlayImage && (
                <>
                  <div className="absolute bottom-1 left-1 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
                    {showOverlay ? 'Translated Text' : 'Original Text'}
                  </div>
                  <button
                    onClick={() => setShowOverlay(!showOverlay)}
                    className="absolute top-1 right-1 bg-black bg-opacity-50 text-white p-2 rounded-full hover:bg-opacity-75 transition-opacity"
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
            <div className="absolute inset-0 flex items-center justify-center z-10">
              <p className="text-gray-500 text-lg">Take a photo or upload image</p>
            </div>
          )}
        </div>

        {/* Bottom sections - adjusted heights */}
        <div className="h-[35vh] flex flex-col justify-end pb-[env(safe-area-inset-bottom,8px)]">
          {/* AI Assistant buttons - 20% */}
          <div className="h-[20vh] px-2 flex items-center">
            {showAIResponse ? (
              <div className="absolute left-1/2 -translate-x-1/2 w-full bg-blue-50 rounded-lg shadow-lg border border-blue-100">
                <div className="h-[18vh] overflow-y-auto px-4 py-3">
                  <div className="space-y-2">
                    <p>this is a test AI message, will be replace by LLM answers. stay tuned</p>
                    <p>this is a test AI message, will be replace by LLM answers. stay tuned</p>
                    <p>this is a test AI message, will be replace by LLM answers. stay tuned</p>
                    <p>this is a test AI message, will be replace by LLM answers. stay tuned</p>
                    <p>this is a test AI message, will be replace by LLM answers. stay tuned</p>
                    <p>this is a test AI message, will be replace by LLM answers. stay tuned</p>
                  </div>
                </div>
                <button 
                  onClick={() => setShowAIResponse(false)}
                  className="absolute top-2 right-2 text-blue-500 hover:text-blue-700"
                >
                  <XMarkIcon className="w-6 h-6" />
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

          {/* Camera buttons - 13% */}
          <div className="h-[13vh] flex items-center justify-center">
            <BottomActionBar />
          </div>
          {/* 2% blank space */}
          <div className="h-[2vh]"></div>
        </div>
      </main>
    </div>
  )
}

export default App