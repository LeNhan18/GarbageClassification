import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_gallery_saver/image_gallery_saver.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Yêu cầu quyền truy cập camera
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

  const MyApp({required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Camera App',
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        useMaterial3: true,
      ),
      home: CameraPreviewPage(camera: camera),
    );
  }
}

class CameraPreviewPage extends StatefulWidget {
  final CameraDescription camera;

  const CameraPreviewPage({required this.camera});

  @override
  _CameraPreviewPageState createState() => _CameraPreviewPageState();
}

class _CameraPreviewPageState extends State<CameraPreviewPage> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  bool _isFlashOn = false;
  double _zoomLevel = 1.0;
  bool _isFrontCamera = false;
  bool _isGridVisible = false;
  List<CameraDescription> cameras = [];
  bool _isInitialized = false;
  bool _isLoading = false;
  String? _predictionResult;
  Timer? _analysisTimer;
  bool _isAnalyzing = false;
  List<Rect> _detectionBoxes = [];
  List<String> _detectionLabels = [];

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      _controller = CameraController(widget.camera, ResolutionPreset.high);
      _initializeControllerFuture = _controller.initialize();
      await _initializeControllerFuture;
      if (mounted) {
        setState(() {
          _isInitialized = true;
        });
      }
      await _loadCameras();
      _startAnalysis();
    } catch (e) {
      print("Error initializing camera: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Lỗi khởi tạo camera: $e',
              style: GoogleFonts.roboto(),
            ),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  void _startAnalysis() {
    _analysisTimer?.cancel();
    _analysisTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
      if (!_isAnalyzing && _isInitialized) {
        _analyzeFrame();
      }
    });
  }

  Future<void> _analyzeFrame() async {
    if (!_isInitialized || !_controller.value.isInitialized) return;

    setState(() {
      _isAnalyzing = true;
    });

    try {
      final XFile file = await _controller.takePicture();
      final directory = await getTemporaryDirectory();
      final String filePath = '${directory.path}/frame_${DateTime.now()}.png';
      await File(filePath).writeAsBytes(await file.readAsBytes());

      final result = await _sendImageForPrediction(filePath);
      
      // Cập nhật các ô nhận diện
      if (result != null && result['boxes'] != null) {
        setState(() {
          _detectionBoxes = (result['boxes'] as List).map((box) {
            return Rect.fromLTWH(
              box['x'] * _controller.value.previewSize!.width,
              box['y'] * _controller.value.previewSize!.height,
              box['width'] * _controller.value.previewSize!.width,
              box['height'] * _controller.value.previewSize!.height,
            );
          }).toList();
          _detectionLabels = (result['labels'] as List).cast<String>();
        });
      }
      
      // Xóa file tạm sau khi phân tích
      await File(filePath).delete();
    } catch (e) {
      print("Error analyzing frame: $e");
    } finally {
      if (mounted) {
        setState(() {
          _isAnalyzing = false;
        });
      }
    }
  }

  Future<Map<String, dynamic>?> _sendImageForPrediction(String imagePath) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('YOUR_API_ENDPOINT'),
      );

      request.files.add(
        await http.MultipartFile.fromPath('image', imagePath),
      );

      final response = await request.send();
      final responseData = await response.stream.bytesToString();
      final result = json.decode(responseData);

      if (mounted) {
        setState(() {
          _predictionResult = result['prediction'];
        });
      }

      return result;
    } catch (e) {
      print("Error sending image: $e");
      return null;
    }
  }

  @override
  void dispose() {
    _analysisTimer?.cancel();
    _controller.dispose();
    super.dispose();
  }

  void _switchCamera() async {
    final lensDirection = _controller.description.lensDirection;
    CameraDescription newCamera;

    if (lensDirection == CameraLensDirection.front) {
      newCamera = cameras.firstWhere(
            (camera) => camera.lensDirection == CameraLensDirection.back,
      );
    } else {
      newCamera = cameras.firstWhere(
            (camera) => camera.lensDirection == CameraLensDirection.front,
      );
    }

    await _controller.dispose();
    _controller = CameraController(newCamera, ResolutionPreset.high);
    await _controller.initialize();
    setState(() {
      _isFrontCamera = !_isFrontCamera;
    });
    _startAnalysis();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [Colors.deepPurple.shade900, Colors.blue.shade900],
              ),
            ),
          ),
          FutureBuilder<void>(
            future: _initializeControllerFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done && !snapshot.hasError) {
                return Center(
                  child: Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(30),
                      child: AspectRatio(
                        aspectRatio: _controller.value.aspectRatio,
                        child: Stack(
                          children: [
                            CameraPreview(_controller),
                            if (_isGridVisible)
                              CustomPaint(
                                size: Size.infinite,
                                painter: GridPainter(),
                              ),
                            // Thêm các ô nhận diện
                            ..._detectionBoxes.asMap().entries.map((entry) {
                              final index = entry.key;
                              final box = entry.value;
                              return Positioned(
                                left: box.left,
                                top: box.top,
                                width: box.width,
                                height: box.height,
                                child: Container(
                                  decoration: BoxDecoration(
                                    border: Border.all(
                                      color: Colors.green,
                                      width: 2,
                                    ),
                                  ),
                                  child: Stack(
                                    children: [
                                      Positioned(
                                        top: -20,
                                        left: 0,
                                        child: Container(
                                          padding: const EdgeInsets.symmetric(
                                            horizontal: 8,
                                            vertical: 4,
                                          ),
                                          color: Colors.green.withOpacity(0.8),
                                          child: Text(
                                            _detectionLabels.length > index
                                                ? _detectionLabels[index]
                                                : '',
                                            style: GoogleFonts.roboto(
                                              color: Colors.white,
                                              fontSize: 12,
                                            ),
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              );
                            }).toList(),
                            Container(
                              decoration: BoxDecoration(
                                gradient: LinearGradient(
                                  begin: Alignment.topCenter,
                                  end: Alignment.bottomCenter,
                                  colors: [
                                    Colors.black.withOpacity(0.3),
                                    Colors.transparent,
                                    Colors.black.withOpacity(0.3),
                                  ],
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                );
              } else if (snapshot.hasError) {
                return Center(
                  child: Text(
                    'Lỗi: ${snapshot.error}',
                    style: GoogleFonts.roboto(color: Colors.white),
                  ),
                );
              }
              return const Center(
                child: CircularProgressIndicator(
                  color: Colors.white,
                  strokeWidth: 3,
                ),
              );
            },
          ),
          if (_isAnalyzing)
            Container(
              color: Colors.black.withOpacity(0.5),
              child: const Center(
                child: CircularProgressIndicator(
                  color: Colors.white,
                ),
              ),
            ),
          if (_predictionResult != null)
            Positioned(
              top: 100,
              left: 0,
              right: 0,
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 20),
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.7),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  'Kết quả: $_predictionResult',
                  style: GoogleFonts.roboto(
                    color: Colors.white,
                    fontSize: 18,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      IconButton(
                        icon: const Icon(Icons.arrow_back_ios, color: Colors.white),
                        onPressed: () => Navigator.pop(context),
                      ),
                      Text(
                        'Camera',
                        style: GoogleFonts.roboto(
                          color: Colors.white,
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      IconButton(
                        icon: Icon(
                          _isGridVisible ? Icons.grid_on : Icons.grid_off,
                          color: Colors.white,
                        ),
                        onPressed: () => setState(() => _isGridVisible = !_isGridVisible),
                      ),
                    ],
                  ),
                  Slider(
                    value: _zoomLevel,
                    min: 1.0,
                    max: 5.0,
                    activeColor: Colors.white,
                    inactiveColor: Colors.white30,
                    onChanged: _isInitialized
                        ? (value) async {
                            setState(() => _zoomLevel = value);
                            await _controller.setZoomLevel(value);
                          }
                        : null,
                  ),
                ],
              ),
            ),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Container(
              padding: const EdgeInsets.all(20),
              margin: const EdgeInsets.only(bottom: 20),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.7),
                borderRadius: BorderRadius.circular(30),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  IconButton(
                    icon: Icon(
                      _isFlashOn ? Icons.flash_on : Icons.flash_off,
                      color: Colors.white,
                      size: 28,
                    ),
                    onPressed: _isInitialized
                        ? () async {
                            setState(() => _isFlashOn = !_isFlashOn);
                            await _controller.setFlashMode(
                              _isFlashOn ? FlashMode.torch : FlashMode.off,
                            );
                          }
                        : null,
                  ),
                  const SizedBox(width: 20),
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: Colors.white.withOpacity(0.3),
                          spreadRadius: 2,
                          blurRadius: 5,
                        ),
                      ],
                    ),
                    child: IconButton(
                      iconSize: 35,
                      icon: const Icon(Icons.camera, color: Colors.black),
                      onPressed: _isInitialized ? _switchCamera : null,
                    ),
                  ),
                  const SizedBox(width: 20),
                  IconButton(
                    icon: Icon(
                      _isFrontCamera ? Icons.camera_front : Icons.camera_rear,
                      color: Colors.white,
                      size: 28,
                    ),
                    onPressed: _isInitialized ? _switchCamera : null,
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white30
      ..strokeWidth = 1;

    // Draw vertical lines
    for (int i = 1; i < 3; i++) {
      final x = size.width * i / 3;
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }

    // Draw horizontal lines
    for (int i = 1; i < 3; i++) {
      final y = size.height * i / 3;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
