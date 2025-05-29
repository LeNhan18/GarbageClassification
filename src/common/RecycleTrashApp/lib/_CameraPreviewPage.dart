import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:typed_data';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img_lib;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'Service/model_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await _requestCameraPermission();
  final cameras = await availableCameras();
  final firstCamera = cameras.isNotEmpty ? cameras.first : null;

  if (firstCamera != null) {
    runApp(MyApp(camera: firstCamera));
  } else {
    print("No camera found!");
  }
}

Future<void> _requestCameraPermission() async {
  try {
    PermissionStatus status = await Permission.camera.request();
    if (status.isDenied) {
      print("Camera permission denied");
      status = await Permission.camera.request();
      if (status.isDenied) {
        print("Camera permission denied again");
        return;
      }
    }
    if (status.isPermanentlyDenied) {
      print("Camera permission permanently denied");
      await openAppSettings();
    }
  } catch (e) {
    print("Error requesting camera permission: $e");
  }
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;
  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Phân Loại Rác AI',
      theme: ThemeData(
        primarySwatch: Colors.green,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        useMaterial3: true,
      ),
      home: CameraPreviewPage(camera: camera),
      debugShowCheckedModeBanner: false,
    );
  }
}

class CameraPreviewPage extends StatefulWidget {
  final CameraDescription camera;
  const CameraPreviewPage({super.key, required this.camera});

  @override
  _CameraPreviewPageState createState() => _CameraPreviewPageState();
}

class _CameraPreviewPageState extends State<CameraPreviewPage>
    with WidgetsBindingObserver {
  // Camera related variables
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;
  bool _isFlashOn = false;
  double _zoomLevel = 1.0;
  double _minZoom = 1.0;
  double _maxZoom = 1.0;
  bool _isFrontCamera = false;
  bool _isGridVisible = false;
  List<CameraDescription> cameras = [];
  bool _isCameraInitialized = false;
  
  // TFLite model related variables
  static const int MODEL_INPUT_WIDTH = 224;
  static const int MODEL_INPUT_HEIGHT = 224;
  bool _isProcessingImage = false;
  String? _predictionResultText;
  Rect? _detectionBox;
  String? _detectionLabel;
  
  // Model service
  late ModelService _modelService;

  // Model file paths
  static const String model1Path = 'Assets/ML_Models/Model1.tflite';
  static const String model2aPath = 'Assets/ML_Models/Model2A.tflite';
  static const String model2bPath = 'Assets/ML_Models/Model2B.tflite';

  // Class labels
  static const Map<int, String> _model1ClassNames = {
    0: 'Không tái chế',
    1: 'Tái chế'
  };
  
  static const Map<int, String> _model2bClassNames = {
    0: 'Pin',
    1: 'Rác hữu cơ',
    2: 'quần áo cũ',
    3: 'Giày dép cũ',
    4: 'Rác thải khác (không tái chế)'
  };

  static const Map<int, String> _model2aClassNames = {
    0: 'Bìa carton',
    1: 'Thủy tinh',
    2: 'Kim loại',
    3: 'Giấy',
    4: 'Nhựa'
  };

  Timer? _processingTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeAndLoadResources();
    // We'll use a manual button instead of automatic processing
    // to ensure better control and avoid camera issues
  }

  Future<void> _initializeAndLoadResources() async {
    await _initializeCamera(widget.camera);
    await _loadAllModels();
  }

  @override
  void dispose() {
    _processingTimer?.cancel();
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    _modelService.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Handle app lifecycle changes
    if (!mounted) return;
    
    final CameraController? cameraController = _controller;
    // If camera is not initialized, nothing to do
    if (cameraController == null || !cameraController.value.isInitialized) {
      return;
    }
    
    if (state == AppLifecycleState.inactive) {
      // When app is inactive, dispose the camera controller
      _controller?.dispose();
      _controller = null;
    } else if (state == AppLifecycleState.resumed) {
      // When app is resumed, reinitialize the camera if needed
      if (_controller == null) {
        _initializeCamera(widget.camera);
      }
    }
  }

  Future<void> _initializeCamera(CameraDescription cameraDescription) async {
    // Safely dispose of the previous controller if it exists
    if (_controller != null) {
      try {
        await _controller!.dispose();
      } catch (e) {
        print('Error disposing camera controller: $e');
      }
      _controller = null;
    }

    // Check if the widget is still mounted before creating a new controller
    if (!mounted) return;
    
    // Create a new camera controller
    _controller = CameraController(
      cameraDescription,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    // Add a listener to update the UI when camera state changes
    _controller!.addListener(() {
      if (mounted) setState(() {});
    });

    try {
      _initializeControllerFuture = _controller!.initialize();
      await _initializeControllerFuture;
      
      // Safely get zoom levels with error handling
      try {
        _minZoom = await _controller!.getMinZoomLevel();
        _maxZoom = await _controller!.getMaxZoomLevel();
        _zoomLevel = _minZoom;
      } catch (zoomError) {
        // Default values if zoom is not supported
        print("Zoom không được hỗ trợ: $zoomError");
        _minZoom = 1.0;
        _maxZoom = 1.0;
        _zoomLevel = 1.0;
      }

      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
        });
      }
      await _loadAvailableCameras();
    } catch (e) {
      print("Lỗi khởi tạo camera: $e");
      if (mounted) {
        _showErrorSnackBar('Lỗi khởi tạo camera: $e');
      }
    }
  }

  Future<void> _loadAvailableCameras() async {
    try {
      cameras = await availableCameras();
    } catch (e) {
      print("Lỗi tải danh sách camera: $e");
    }
  }

  Future<void> _loadAllModels() async {
    try {
      // Initialize the appropriate model service based on platform
      if (kIsWeb) {
        _modelService = WebModelService();
      } else {
        _modelService = NativeModelService();
      }
      
      // Load all models
      await _modelService.loadModels();
      
      print("Đã tải xong tất cả các mô hình");
    } catch (e) {
      print("Lỗi tải mô hình: $e");
      if (mounted) {
        _showErrorSnackBar('Lỗi tải mô hình: $e');
      }
    }
  }

  // Helper function to get the index of the maximum value in a list
  int _getIndexOfMax(List<double> list) {
    if (list.isEmpty) return -1;
    
    int maxIndex = 0;
    double maxValue = list[0];
    
    for (int i = 1; i < list.length; i++) {
      if (list[i] > maxValue) {
        maxValue = list[i];
        maxIndex = i;
      }
    }
    
    return maxIndex;
  }

  Future<void> _processImageAndGetPrediction(XFile imageFile) async {
    try {
      // Hiển thị thông báo đang xử lý
      if (mounted) {
        setState(() {
          _predictionResultText = 'Đang chuẩn bị ảnh...';
        });
      }
      
      // Thông tin về ảnh
      print('Xử lý ảnh: ${imageFile.path}');
      print('Kích thước ảnh: ${await imageFile.length()} bytes');
      
      // Đọc ảnh dưới dạng bytes
      final bytes = await imageFile.readAsBytes();
      print('Kích thước bytes: ${bytes.length}');
      
      if (mounted) {
        setState(() {
          _predictionResultText = 'Đang phân tích ảnh...';
        });
      }
      
      // Đảm bảo ảnh được xử lý đúng cách trước khi đưa vào mô hình
      if (bytes.isEmpty) {
        throw Exception('Ảnh không hợp lệ hoặc trống');
      }
      
      // First, determine if the item is recyclable or not
      final model1Results = await _modelService.runModel1(imageFile);
      print('Kết quả Model 1: $model1Results');
      
      // Check if the item is recyclable (class index 1) or not (class index 0)
      final isRecyclable = model1Results[1] > model1Results[0];
      print('Có thể tái chế: $isRecyclable');
      
      // Hiển thị kết quả trung gian
      if (mounted) {
        setState(() {
          _predictionResultText = isRecyclable ? 'Đang xác định loại vật liệu tái chế...' : 'Đang xác định loại chất thải không tái chế...';
        });
      }
      
      String resultText;
      if (isRecyclable) {
        // If recyclable, run model 2a to determine the type of recyclable material
        final model2aResults = await _modelService.runModel2a(imageFile);
        print('Kết quả Model 2a: $model2aResults');
        
        // Find the class with highest probability
        int maxIndex = 0;
        double maxProb = model2aResults[0];
        for (int i = 1; i < model2aResults.length; i++) {
          if (model2aResults[i] > maxProb) {
            maxProb = model2aResults[i];
            maxIndex = i;
          }
        }
        print('Chỉ số lớp tái chế cao nhất: $maxIndex, Xác suất: $maxProb');
        
        // Get the class name for the recyclable item
        final className = _model2aClassNames[maxIndex] ?? 'Không xác định';
        resultText = 'Tái chế - $className (${(maxProb * 100).toStringAsFixed(1)}%)';
        
        // Hiển thị khung và nhãn
        if (mounted) {
          final Size previewSize = MediaQuery.of(context).size;
          final double boxWidth = previewSize.width * 0.8;
          final double boxHeight = previewSize.height * 0.4;
          
          setState(() {
            _detectionBox = Rect.fromCenter(
              center: Offset(previewSize.width / 2, previewSize.height / 2),
              width: boxWidth,
              height: boxHeight,
            );
            _detectionLabel = "Tái chế\n$className";
          });
        }
      } else {
        // If not recyclable, run model 2b to determine the type of non-recyclable waste
        final model2bResults = await _modelService.runModel2b(imageFile);
        print('Kết quả Model 2b: $model2bResults');
        
        // Find the class with highest probability
        int maxIndex = 0;
        double maxProb = model2bResults[0];
        for (int i = 1; i < model2bResults.length; i++) {
          if (model2bResults[i] > maxProb) {
            maxProb = model2bResults[i];
            maxIndex = i;
          }
        }
        print('Chỉ số lớp không tái chế cao nhất: $maxIndex, Xác suất: $maxProb');
        
        // Get the class name for the non-recyclable item
        final className = _model2bClassNames[maxIndex] ?? 'Không xác định';
        resultText = 'Không tái chế - $className (${(maxProb * 100).toStringAsFixed(1)}%)';
        
        // Hiển thị khung và nhãn
        if (mounted) {
          final Size previewSize = MediaQuery.of(context).size;
          final double boxWidth = previewSize.width * 0.8;
          final double boxHeight = previewSize.height * 0.4;
          
          setState(() {
            _detectionBox = Rect.fromCenter(
              center: Offset(previewSize.width / 2, previewSize.height / 2),
              width: boxWidth,
              height: boxHeight,
            );
            _detectionLabel = "Không tái chế\n$className";
          });
        }
      }
      
      if (mounted) {
        setState(() {
          _predictionResultText = resultText;
        });
      }
    } catch (e) {
      print('Lỗi trong quá trình dự đoán: $e');
      if (mounted) {
        setState(() {
          _predictionResultText = 'Lỗi: $e';
          _detectionBox = null;
          _detectionLabel = null;
        });
        _showErrorSnackBar('Lỗi dự đoán: $e');
      }
    }
  }
  
  Future<void> _processImageFromCamera() async {
    // Don't process if already processing or camera not ready
    if (_isProcessingImage || _controller == null || !_controller!.value.isInitialized) {
      _showErrorSnackBar("Camera chưa sẵn sàng");
      return;
    }

    setState(() {
      _isProcessingImage = true;
      _predictionResultText = "Đang chụp ảnh...";
    });

    try {
      // Check if controller is still valid
      if (_controller == null || !_controller!.value.isInitialized) {
        throw Exception('Camera not initialized');
      }
      
      // Take picture and process it
      print("Taking picture...");
      final image = await _controller!.takePicture();
      print("Picture taken: ${image.path}");
      
      if (mounted) {
        setState(() {
          _predictionResultText = "Đang phân tích...";
        });
        
        // Process the image
        await _processImageAndGetPrediction(image);
      }
    } catch (e) {
      print("Lỗi xử lý ảnh từ camera: $e");
      if (mounted) {
        setState(() {
          _predictionResultText = "Lỗi xử lý ảnh: $e";
        });
        _showErrorSnackBar("Lỗi xử lý ảnh: $e");
      }
    } finally {
      if (mounted) {
        setState(() {
          _isProcessingImage = false;
        });
      }
    }
  }

  void _showErrorSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
        duration: Duration(seconds: 3),
      ),
    );
  }

  Future<void> _showPredictionDialog(String category, String detail) async {
    await showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Kết quả phân loại'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Loại: $category'),
            SizedBox(height: 8),
            Text('Chi tiết: $detail'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: Text('Đóng'),
          ),
        ],
      ),
    );
  }

  Future<void> _pickImageAndPredict() async {
    try {
      setState(() {
        _isProcessingImage = true;
        _predictionResultText = "Đang chọn ảnh...";
      });
      
      final ImagePicker picker = ImagePicker();
      final XFile? imageFile = await picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 800,  // Giới hạn kích thước để tối ưu hiệu suất
        maxHeight: 800,
        imageQuality: 90, // Chất lượng ảnh tốt
      );
      
      if (imageFile != null) {
        print("Đã chọn ảnh: ${imageFile.path}");
        setState(() {
          _predictionResultText = "Đang phân tích ảnh...";
        });
        
        // Xử lý ảnh và dự đoán
        await _processImageAndGetPrediction(imageFile);
      } else if (mounted) {
        setState(() {
          _predictionResultText = "Chưa chọn ảnh";
          _isProcessingImage = false;
        });
      }
    } catch (e) {
      print('Lỗi chọn ảnh: $e');
      if (mounted) {
        setState(() {
          _predictionResultText = "Lỗi chọn ảnh: $e";
          _isProcessingImage = false;
        });
        _showErrorSnackBar("Lỗi chọn ảnh: $e");
      }
    } finally {
      if (mounted) {
        setState(() {
          _isProcessingImage = false;
        });
      }
    }
  }

  Future<void> _switchCamera() async {
    if (!_isCameraInitialized || _controller == null || cameras.isEmpty) return;
    
    // Get the current lens direction
    final currentLensDirection = _controller!.description.lensDirection;
    CameraDescription newCameraDescription;
    
    if (currentLensDirection == CameraLensDirection.back) {
      // Switch to front camera
      newCameraDescription = cameras.firstWhere(
          (camera) => camera.lensDirection == CameraLensDirection.front,
          orElse: () => widget.camera);
    } else {
      // Switch to back camera
      newCameraDescription = cameras.firstWhere(
          (camera) => camera.lensDirection == CameraLensDirection.back,
          orElse: () => widget.camera);
    }
    
    if (newCameraDescription.name != _controller!.description.name) {
      await _initializeCamera(newCameraDescription);
      if (mounted) {
        setState(() {
          _isFrontCamera = !_isFrontCamera;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Phân Loại Rác AI'),
        actions: [
          // Flash toggle button
          IconButton(
            icon: Icon(_isFlashOn ? Icons.flash_off : Icons.flash_on),
            onPressed: () async {
              if (_controller != null) {
                if (_isFlashOn) {
                  await _controller!.setFlashMode(FlashMode.off);
                } else {
                  await _controller!.setFlashMode(FlashMode.torch);
                }
                setState(() {
                  _isFlashOn = !_isFlashOn;
                });
              }
            },
          ),
          // Camera switch button
          IconButton(
            icon: Icon(Icons.flip_camera_ios),
            onPressed: _switchCamera,
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: FutureBuilder<void>(
              future: _initializeControllerFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  if (_controller != null) {
                    return Stack(
                      children: [
                        CameraPreview(_controller!),
                        if (_detectionBox != null &&
                            _detectionLabel != null)
                          CustomPaint(
                            painter: DetectionBoxPainter(
                              boundingBox: _detectionBox!,
                              label: _detectionLabel!,
                            ),
                            size: Size.infinite,
                          ),
                        if (_isGridVisible)
                          CustomPaint(
                            painter: GridPainter(),
                            size: Size.infinite,
                          ),
                      ],
                    );
                  } else {
                    return Center(child: Text('Camera không khả dụng'));
                  }
                } else {
                  return Center(child: CircularProgressIndicator());
                }
              },
            ),
          ),
          Container(
            padding: EdgeInsets.all(16),
            color: Colors.black87,
            child: Column(
              children: [
                Text(
                  _predictionResultText ?? "Chưa có kết quả",
                  style: TextStyle(color: Colors.white, fontSize: 18),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 16),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    IconButton(
                      icon: Icon(
                        _isGridVisible ? Icons.grid_off : Icons.grid_on,
                      ),
                      onPressed: () {
                        setState(() {
                          _isGridVisible = !_isGridVisible;
                        });
                      },
                    ),
                    IconButton(
                      icon: Icon(Icons.image),
                      onPressed: _pickImageAndPredict,
                    ),
                    // Add manual prediction button
                    ElevatedButton(
                      onPressed: () {
                        if (_controller != null && _controller!.value.isInitialized) {
                          _processImageFromCamera();
                        } else {
                          _showErrorSnackBar('Camera chưa sẵn sàng');
                        }
                      },
                      child: Text('Phân loại'),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class DetectionBoxPainter extends CustomPainter {
  final Rect boundingBox;
  final String label;
  
  DetectionBoxPainter({
    required this.boundingBox,
    required this.label,
  });
  
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
    
    // Draw bounding box
    canvas.drawRect(boundingBox, paint);
    
    // Draw label background
    final textPainter = TextPainter(
      text: TextSpan(
        text: label,
        style: TextStyle(
          color: Colors.white,
          fontSize: 16,
          backgroundColor: Colors.green,
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    
    textPainter.layout();
    
    final textBackgroundRect = Rect.fromLTWH(
      boundingBox.left,
      boundingBox.top - textPainter.height - 4,
      textPainter.width + 8,
      textPainter.height + 4,
    );
    
    canvas.drawRect(
      textBackgroundRect,
      Paint()..color = Colors.green,
    );
    
    // Draw label text
    textPainter.paint(
      canvas,
      Offset(boundingBox.left + 4, boundingBox.top - textPainter.height - 2),
    );
  }
  
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}

class GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.3)
      ..strokeWidth = 1;
    
    // Draw horizontal lines
    for (int i = 1; i < 3; i++) {
      final y = size.height * (i / 3);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
    
    // Draw vertical lines
    for (int i = 1; i < 3; i++) {
      final x = size.width * (i / 3);
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
  }
  
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return false;
  }
}
